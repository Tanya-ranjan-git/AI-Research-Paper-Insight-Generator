"""Structured summaries of a single paper."""
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from config import settings
from src.rag_pipeline import safe_invoke
from src.vector_store import VectorStoreManager
from utils.prompts import SUMMARY_PROMPT, SUMMARY_SECTIONS


def build_paper_text(vm: VectorStoreManager, document_name: str,
                     max_chars: int | None = None) -> str:
    """Rebuild page-labelled paper text from stored chunks.

    Overlapping chunks would duplicate text, so we keep it simple: join chunks per page.
    If the paper is longer than max_chars we keep the start (abstract/intro/method) and
    the end (results/conclusion) and drop the middle.
    """
    max_chars = max_chars or settings.MAX_CONTEXT_CHARS_PER_PAPER
    chunks = vm.get_document_chunks(document_name)
    if not chunks:
        return ""
    pages: dict[int, list[str]] = {}
    for c in chunks:
        pages.setdefault(c.metadata["page"], []).append(c.page_content)
    blocks = [f"[Page {p}]\n" + "\n".join(parts) for p, parts in sorted(pages.items())]
    full = "\n\n".join(blocks)
    if len(full) <= max_chars:
        return full
    head_budget, tail_budget = int(max_chars * 0.6), int(max_chars * 0.4)
    return full[:head_budget] + "\n\n[... middle of the paper omitted for length ...]\n\n" + full[-tail_budget:]


class PaperSummarizer:
    def __init__(self, vector_manager: VectorStoreManager, llm: BaseChatModel):
        self.vm = vector_manager
        self.llm = llm

    def summarize(self, document_name: str) -> str:
        text = build_paper_text(self.vm, document_name)
        if not text:
            raise ValueError(f"No indexed text found for '{document_name}'.")
        chain = SUMMARY_PROMPT | self.llm | StrOutputParser()
        return safe_invoke(chain, {
            "document_name": document_name,
            "paper_text": text,
            "sections": "\n".join(f"{i}. {s}" for i, s in enumerate(SUMMARY_SECTIONS, 1)),
        }).strip()
