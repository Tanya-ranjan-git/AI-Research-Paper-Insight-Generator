"""Compare two or more papers: structured table + short analysis."""
import json
import re
from dataclasses import dataclass

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from config import settings
from src.rag_pipeline import LLMCallError, safe_invoke
from src.summarizer import build_paper_text
from src.vector_store import VectorStoreManager
from utils.prompts import COMPARE_DIMENSIONS, COMPARE_PROMPT


@dataclass
class ComparisonResult:
    table: pd.DataFrame   # rows = dimensions, columns = paper names
    analysis: str


def parse_json_response(raw: str) -> dict:
    """Parse JSON from an LLM reply, tolerating code fences or surrounding text."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def build_comparison_table(data: dict, paper_names: list[str]) -> pd.DataFrame:
    table = data.get("table", {}) if isinstance(data, dict) else {}
    rows = {}
    for dim in COMPARE_DIMENSIONS:
        cell = table.get(dim, {}) if isinstance(table.get(dim, {}), dict) else {}
        rows[dim] = [str(cell.get(name, "Not mentioned")) for name in paper_names]
    return pd.DataFrame.from_dict(rows, orient="index", columns=paper_names)


class PaperComparator:
    def __init__(self, vector_manager: VectorStoreManager, llm: BaseChatModel):
        self.vm = vector_manager
        self.llm = llm

    def compare(self, paper_names: list[str]) -> ComparisonResult:
        if len(paper_names) < 2:
            raise ValueError("Select at least two papers to compare.")
        # Split the context budget so several papers still fit
        budget = max(8000, settings.MAX_CONTEXT_CHARS_PER_PAPER // len(paper_names))
        blocks = []
        for name in paper_names:
            text = build_paper_text(self.vm, name, max_chars=budget)
            if not text:
                raise ValueError(f"No indexed text found for '{name}'.")
            blocks.append(f"=== PAPER: {name} ===\n{text}")

        chain = COMPARE_PROMPT | self.llm | StrOutputParser()
        inputs = {
            "papers_text": "\n\n".join(blocks),
            "dimensions": json.dumps(COMPARE_DIMENSIONS),
            "paper_names": json.dumps(paper_names),
        }
        raw = safe_invoke(chain, inputs)
        try:
            data = parse_json_response(raw)
        except json.JSONDecodeError:
            raise LLMCallError("Gemini returned a comparison in an unreadable format. Please try again.") from None
        return ComparisonResult(
            table=build_comparison_table(data, paper_names),
            analysis=str(data.get("analysis", "")).strip() or "No analysis returned.",
        )
