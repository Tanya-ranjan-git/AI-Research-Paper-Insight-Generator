import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.paper_comparator import PaperComparator, parse_json_response
from src.pdf_processor import extract_pdf
from src.rag_pipeline import LLMCallError, LLMConfigError, RAGPipeline, build_llm, safe_invoke
from src.summarizer import PaperSummarizer, build_paper_text
from src.text_chunker import chunk_extraction
from src.vector_store import VectorStoreManager
from utils.prompts import NOT_FOUND_MESSAGE


@pytest.fixture
def vm(tmp_path, embeddings, pdf_a, pdf_b):
    manager = VectorStoreManager(embeddings, persist_dir=tmp_path / "vs")
    manager.add_documents(chunk_extraction(extract_pdf(pdf_a, "crops.pdf")))
    manager.add_documents(chunk_extraction(extract_pdf(pdf_b, "protein.pdf")))
    return manager


def test_retrieval_returns_correct_doc_and_page(vm):
    pipe = RAGPipeline(vm, llm=FakeListChatModel(responses=["x"]))
    top = pipe.retrieve("What accuracy does ResNet50 reach on PlantVillage?", k=2)
    assert top[0].document_name == "crops.pdf" and top[0].page == 2
    assert top[0].citation == "[crops.pdf, Page 2]"
    assert 0 < top[0].score <= 1.0001


def test_document_filter(vm):
    pipe = RAGPipeline(vm, llm=FakeListChatModel(responses=["x"]))
    res = pipe.retrieve("model results", k=5, document_names=["protein.pdf"])
    assert res and all(r.document_name == "protein.pdf" for r in res)


def test_multi_document_listing_and_remove(vm):
    assert set(vm.list_documents()) == {"crops.pdf", "protein.pdf"}
    assert vm.remove_document("crops.pdf") > 0
    assert set(vm.list_documents()) == {"protein.pdf"}
    assert all(r.document_name == "protein.pdf" for r in
               RAGPipeline(vm, llm=FakeListChatModel(responses=["x"])).retrieve("ResNet50 PlantVillage"))


def test_persistence_roundtrip(vm, embeddings, tmp_path):
    reloaded = VectorStoreManager(embeddings, persist_dir=tmp_path / "vs")
    assert set(reloaded.list_documents()) == {"crops.pdf", "protein.pdf"}


def test_clear_and_rebuild(vm, pdf_a):
    chunks = chunk_extraction(extract_pdf(pdf_a, "crops.pdf"))
    assert vm.rebuild(chunks) == len(chunks)
    assert set(vm.list_documents()) == {"crops.pdf"}
    vm.clear()
    assert vm.list_documents() == {} and vm.similarity_search("anything") == []


def test_ask_returns_answer_with_sources(vm):
    llm = FakeListChatModel(responses=["It reaches 97 percent [crops.pdf, Page 2]."])
    out = RAGPipeline(vm, llm=llm).ask("What accuracy does ResNet50 reach?")
    assert out.found and "97" in out.answer and out.sources


def test_ask_not_found_flag(vm):
    out = RAGPipeline(vm, llm=FakeListChatModel(responses=[NOT_FOUND_MESSAGE])).ask("Who won the 1998 World Cup?")
    assert out.found is False


def test_ask_with_empty_index_skips_llm(tmp_path, embeddings):
    empty = VectorStoreManager(embeddings, persist_dir=tmp_path / "empty")
    out = RAGPipeline(empty, llm=None).ask("anything")   # llm None would raise if it were called
    assert not out.found and not out.sources


def test_empty_question_rejected(vm):
    with pytest.raises(ValueError):
        RAGPipeline(vm, llm=FakeListChatModel(responses=["x"])).ask("   ")


def test_prompt_contains_context_and_citation_labels(vm):
    captured = {}

    class Spy(FakeListChatModel):
        def _generate(self, messages, *a, **k):
            captured["text"] = "\n".join(str(m.content) for m in messages)
            return super()._generate(messages, *a, **k)

    RAGPipeline(vm, llm=Spy(responses=["ok"])).ask("ResNet50 accuracy PlantVillage")
    assert "[crops.pdf, Page 2]" in captured["text"]
    assert "ONLY the context" in captured["text"]


def test_missing_api_key_gives_setup_message(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from config import settings
    monkeypatch.setattr(settings, "get_gemini_api_key", lambda: None)
    with pytest.raises(LLMConfigError, match="GOOGLE_API_KEY"):
        build_llm()


def test_safe_invoke_does_not_leak_key():
    class Boom:
        def invoke(self, _):
            raise RuntimeError("403 API key AIzaSySECRET123 not valid")
    with pytest.raises(LLMCallError) as e:
        safe_invoke(Boom(), {})
    assert "AIzaSySECRET123" not in str(e.value)


def test_summarizer_uses_paper_text(vm):
    llm = FakeListChatModel(responses=["## Research Objective\nDetect diseases [crops.pdf, Page 1]."])
    text = build_paper_text(vm, "crops.pdf")
    assert "[Page 1]" in text and "[Page 2]" in text
    assert "Research Objective" in PaperSummarizer(vm, llm).summarize("crops.pdf")
    with pytest.raises(ValueError):
        PaperSummarizer(vm, llm).summarize("missing.pdf")


def test_build_paper_text_truncates_long_papers(vm):
    assert "omitted for length" in build_paper_text(vm, "crops.pdf", max_chars=200)


def test_parse_json_with_code_fences():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Here you go: {"a": 2} thanks') == {"a": 2}


def test_comparator_builds_table(vm):
    payload = {"table": {"Datasets": {"crops.pdf": "PlantVillage [crops.pdf, Page 1]",
                                      "protein.pdf": "Protein Data Bank [protein.pdf, Page 1]"}},
               "analysis": "Both use deep models."}
    result = PaperComparator(vm, FakeListChatModel(responses=[json.dumps(payload)])).compare(["crops.pdf", "protein.pdf"])
    assert list(result.table.columns) == ["crops.pdf", "protein.pdf"]
    assert "PlantVillage" in result.table.loc["Datasets", "crops.pdf"]
    assert result.table.loc["Limitations", "crops.pdf"] == "Not mentioned"   # missing dims filled in
    assert "Both use" in result.analysis


def test_comparator_needs_two_papers_and_handles_bad_json(vm):
    with pytest.raises(ValueError):
        PaperComparator(vm, FakeListChatModel(responses=["x"])).compare(["crops.pdf"])
    with pytest.raises(LLMCallError):
        PaperComparator(vm, FakeListChatModel(responses=["not json at all"])).compare(["crops.pdf", "protein.pdf"])


def test_real_embeddings_if_available(tmp_path):
    """Optional: runs only when the real MiniLM model can be loaded (needs internet or cache)."""
    try:
        from src.embeddings import get_embeddings
        emb = get_embeddings()
        vec = emb.embed_query("hello")
    except Exception:
        pytest.skip("all-MiniLM-L6-v2 not available in this environment")
    assert len(vec) == 384
