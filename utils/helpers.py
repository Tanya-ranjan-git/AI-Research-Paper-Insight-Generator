"""Small shared helpers."""
import re
from pathlib import Path

from config import settings


def clean_text(text: str) -> str:
    """Normalize whitespace: fix hyphenated line breaks, collapse spaces/blank lines."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # "meth-\nod" -> "method"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def safe_filename(name: str) -> str:
    """Strip any path components and unsafe characters from an uploaded file name."""
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    return name or "document.pdf"


def validate_pdf_bytes(data: bytes, filename: str) -> None:
    """Raise ValueError if the upload is obviously not an acceptable PDF."""
    if not filename.lower().endswith(".pdf"):
        raise ValueError(f"'{filename}' is not a PDF file.")
    if not data:
        raise ValueError(f"'{filename}' is empty.")
    if len(data) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"'{filename}' exceeds {settings.MAX_FILE_SIZE_MB} MB.")
    if not data.lstrip().startswith(b"%PDF"):
        raise ValueError(f"'{filename}' does not look like a valid PDF.")


def format_citation(doc_name: str, page: int) -> str:
    return f"[{doc_name}, Page {page}]"
