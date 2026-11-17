"""Split page text into overlapping chunks, attaching metadata to each one."""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.pdf_processor import ExtractionResult


def chunk_extraction(
    extraction: ExtractionResult,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Chunk each page separately so every chunk keeps an exact page number."""
    size = chunk_size or settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: list[Document] = []
    chunk_counter = 0
    for page in extraction.pages:
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if not piece:
                continue
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "document_name": page.document_name,
                        "page": page.page_number,
                        "chunk_id": f"{page.document_name}::chunk-{chunk_counter}",
                        "source": f"{page.document_name} (page {page.page_number})",
                    },
                )
            )
            chunk_counter += 1
    return docs
