"""RAG pipeline: retrieve from FAISS -> build prompt -> Gemini -> answer + citations (LangChain LCEL)."""
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from config import settings
from src.vector_store import VectorStoreManager
from utils.helpers import format_citation
from utils.prompts import NOT_FOUND_MESSAGE, RAG_PROMPT


class LLMConfigError(Exception):
    """Gemini API key missing or LLM could not be created."""


class LLMCallError(Exception):
    """Gemini call failed (message is safe to show - contains no secrets)."""


def build_llm(api_key: str | None = None) -> BaseChatModel:
    key = api_key or settings.get_gemini_api_key()
    if not key:
        raise LLMConfigError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your Gemini API key "
            "(free key: https://aistudio.google.com/apikey)."
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL, google_api_key=key,
        temperature=settings.GEMINI_TEMPERATURE,
    )


def safe_invoke(chain, inputs: dict) -> str:
    """Run a chain; convert provider errors into messages that never leak the API key."""
    try:
        return chain.invoke(inputs)
    except Exception as exc:
        text = str(exc).lower()
        if "api key" in text or "api_key" in text or "permission" in text or "401" in text or "403" in text:
            msg = "Gemini rejected the API key. Check GOOGLE_API_KEY in your .env file."
        elif "429" in text or "quota" in text or "rate" in text or "resource_exhausted" in text:
            msg = "Gemini rate limit or quota reached. Wait a moment and try again."
        elif "404" in text or "not found" in text:
            msg = f"Gemini model '{settings.GEMINI_MODEL}' was not found. Change GEMINI_MODEL in .env."
        else:
            msg = f"Gemini request failed ({type(exc).__name__})."
        raise LLMCallError(msg) from None


@dataclass
class SourceExcerpt:
    document_name: str
    page: int
    text: str
    score: float
    chunk_id: str

    @property
    def citation(self) -> str:
        return format_citation(self.document_name, self.page)


@dataclass
class RAGAnswer:
    answer: str
    sources: list[SourceExcerpt] = field(default_factory=list)
    found: bool = True


def format_context(excerpts: list[SourceExcerpt]) -> str:
    return "\n\n".join(f"{e.citation}\n{e.text}" for e in excerpts)


def format_history(history: list[dict] | None, max_turns: int = 4) -> str:
    if not history:
        return "(none)"
    lines = []
    for m in history[-max_turns * 2:]:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content'][:600]}")
    return "\n".join(lines)


class RAGPipeline:
    def __init__(self, vector_manager: VectorStoreManager, llm: BaseChatModel | None = None):
        self.vm = vector_manager
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = build_llm()
        return self._llm

    def retrieve(self, question: str, k: int | None = None,
                 document_names: list[str] | None = None) -> list[SourceExcerpt]:
        results = self.vm.similarity_search(question, k=k, document_names=document_names)
        return [
            SourceExcerpt(
                document_name=d.metadata["document_name"], page=d.metadata["page"],
                text=d.page_content, score=float(score), chunk_id=d.metadata["chunk_id"],
            )
            for d, score in results
        ]

    def ask(self, question: str, history: list[dict] | None = None, k: int | None = None,
            document_names: list[str] | None = None) -> RAGAnswer:
        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question.")
        excerpts = self.retrieve(question, k=k, document_names=document_names)
        if not excerpts:
            return RAGAnswer(answer=NOT_FOUND_MESSAGE + " (No documents are indexed yet.)", found=False)

        chain = RAG_PROMPT | self.llm | StrOutputParser()
        answer = safe_invoke(chain, {
            "history": format_history(history),
            "context": format_context(excerpts),
            "question": question,
        })
        found = NOT_FOUND_MESSAGE.lower() not in answer.lower()
        return RAGAnswer(answer=answer.strip(), sources=excerpts, found=found)
