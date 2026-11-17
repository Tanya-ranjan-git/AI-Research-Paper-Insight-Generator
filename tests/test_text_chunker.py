import pytest

from src.pdf_processor import ExtractionResult, PageContent, extract_pdf
from src.text_chunker import chunk_extraction


def _long_extraction():
    text = " ".join(f"Sentence number {i} about neural networks." for i in range(120))
    return ExtractionResult("long.pdf", [PageContent("long.pdf", 7, text)], total_pages=1)


def test_metadata_on_every_chunk(pdf_a):
    chunks = chunk_extraction(extract_pdf(pdf_a, "a.pdf"))
    assert chunks
    for c in chunks:
        assert c.metadata["document_name"] == "a.pdf"
        assert c.metadata["page"] in (1, 2)
        assert c.metadata["chunk_id"].startswith("a.pdf::chunk-")
        assert "a.pdf" in c.metadata["source"]


def test_chunk_ids_unique_and_page_preserved():
    chunks = chunk_extraction(_long_extraction(), chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 3
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)
    assert {c.metadata["page"] for c in chunks} == {7}


def test_chunk_size_respected_and_overlap_present():
    chunks = chunk_extraction(_long_extraction(), chunk_size=300, chunk_overlap=60)
    assert all(len(c.page_content) <= 300 for c in chunks)
    # consecutive chunks share some text because of overlap
    shared = [set(a.page_content.split()) & set(b.page_content.split())
              for a, b in zip(chunks, chunks[1:])]
    assert all(len(s) > 3 for s in shared)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_extraction(_long_extraction(), chunk_size=100, chunk_overlap=100)
