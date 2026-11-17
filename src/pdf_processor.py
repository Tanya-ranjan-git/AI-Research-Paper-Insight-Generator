"""PDF text extraction with PyMuPDF. Keeps document name and page number for every page."""
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from config import settings
from utils.helpers import clean_text


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be opened or read."""


@dataclass
class PageContent:
    document_name: str
    page_number: int  # 1-based
    text: str


@dataclass
class ExtractionResult:
    document_name: str
    pages: list[PageContent] = field(default_factory=list)
    total_pages: int = 0
    empty_pages: int = 0

    @property
    def has_text(self) -> bool:
        return len(self.pages) > 0

    @property
    def is_probably_scanned(self) -> bool:
        return self.total_pages > 0 and not self.has_text

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


def extract_pdf(source: str | Path | bytes, document_name: str | None = None) -> ExtractionResult:
    """Extract text page by page from a file path or raw bytes.

    Pages with almost no text are skipped (counted in empty_pages). If *every* page is
    empty the result has has_text == False so the UI can warn about scanned PDFs.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            doc = pymupdf.open(stream=bytes(source), filetype="pdf")
            name = document_name or "uploaded.pdf"
        else:
            path = Path(source)
            if not path.exists():
                raise PDFProcessingError(f"File not found: {path.name}")
            doc = pymupdf.open(str(path))
            name = document_name or path.name
    except PDFProcessingError:
        raise
    except Exception as exc:  # fitz raises various error types
        raise PDFProcessingError(f"Could not open PDF: {type(exc).__name__}") from exc

    result = ExtractionResult(document_name=name)
    try:
        if doc.needs_pass:
            raise PDFProcessingError(f"'{name}' is password protected.")
        result.total_pages = doc.page_count
        for index in range(doc.page_count):
            text = clean_text(doc[index].get_text("text"))
            if len(text) < settings.MIN_CHARS_PER_PAGE:
                result.empty_pages += 1
                continue
            result.pages.append(PageContent(name, index + 1, text))
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(f"Failed while reading '{name}': {type(exc).__name__}") from exc
    finally:
        doc.close()
    return result


def scanned_pdf_message(name: str) -> str:
    return (
        f"No extractable text found in '{name}'. It is probably a scanned (image-only) PDF. "
        "This app does not include OCR yet - please upload a text-based PDF or run OCR first "
        "(e.g. with OCRmyPDF)."
    )
