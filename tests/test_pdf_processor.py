import pytest

from src.pdf_processor import PDFProcessingError, extract_pdf
from tests.conftest import make_pdf_bytes
from utils.helpers import clean_text, safe_filename, validate_pdf_bytes


def test_extracts_pages_with_numbers(pdf_a):
    result = extract_pdf(pdf_a, "paper_a.pdf")
    assert result.total_pages == 2
    assert [p.page_number for p in result.pages] == [1, 2]
    assert all(p.document_name == "paper_a.pdf" for p in result.pages)
    assert "PlantVillage" in result.pages[0].text
    assert "ResNet50" in result.pages[1].text


def test_blank_pages_are_skipped_but_counted():
    result = extract_pdf(make_pdf_bytes(["Real text on the first page of this paper.", ""]), "x.pdf")
    assert result.total_pages == 2
    assert len(result.pages) == 1 and result.empty_pages == 1
    assert not result.is_probably_scanned


def test_image_only_pdf_flagged_as_scanned():
    result = extract_pdf(make_pdf_bytes(["", ""]), "scan.pdf")
    assert result.is_probably_scanned and not result.has_text


def test_corrupt_bytes_raise_friendly_error():
    with pytest.raises(PDFProcessingError):
        extract_pdf(b"this is not a pdf", "bad.pdf")


def test_missing_file_raises(tmp_path):
    with pytest.raises(PDFProcessingError):
        extract_pdf(tmp_path / "nope.pdf")


def test_clean_text():
    assert clean_text("meth-\nod   with   spaces\n\n\n\nnext") == "method with spaces\n\nnext"


def test_safe_filename_blocks_path_traversal():
    assert safe_filename("../../etc/pass wd.pdf") == "pass wd.pdf"


def test_validate_pdf_bytes(pdf_a):
    validate_pdf_bytes(pdf_a, "ok.pdf")
    for data, name in [(b"", "a.pdf"), (b"hello", "a.pdf"), (pdf_a, "a.txt")]:
        with pytest.raises(ValueError):
            validate_pdf_bytes(data, name)
