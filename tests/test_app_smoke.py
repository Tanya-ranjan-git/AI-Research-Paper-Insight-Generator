"""Headless UI smoke test with Streamlit AppTest (fake embeddings + fake Gemini)."""
import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from streamlit.testing.v1 import AppTest

from config import settings
from src.pdf_processor import extract_pdf
from src.text_chunker import chunk_extraction
from src.vector_store import VectorStoreManager

COMPARE_JSON = json.dumps({"table": {"Datasets": {"crops.pdf": "PlantVillage", "protein.pdf": "PDB"}},
                           "analysis": "Both are deep learning papers."})


@pytest.fixture
def app(tmp_path, monkeypatch, embeddings, pdf_a, pdf_b):
    monkeypatch.setattr(settings, "VECTOR_STORE_DIR", tmp_path / "vs")
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "up")
    monkeypatch.setattr(settings, "get_gemini_api_key", lambda: "dummy")
    vm = VectorStoreManager(embeddings, persist_dir=tmp_path / "vs")
    vm.add_documents(chunk_extraction(extract_pdf(pdf_a, "crops.pdf")))
    vm.add_documents(chunk_extraction(extract_pdf(pdf_b, "protein.pdf")))

    import src.embeddings as emb_mod
    import src.rag_pipeline as rag_mod
    monkeypatch.setattr(emb_mod, "get_embeddings", lambda *a, **k: embeddings)

    class RoutedFake(FakeListChatModel):
        """Pick a canned reply based on which prompt arrived."""
        def _generate(self, messages, *a, **k):
            text = "\n".join(str(m.content) for m in messages)
            if "Return JSON" in text:
                self.responses = [COMPARE_JSON]
            elif "structured summary" in text:
                self.responses = ["## Research Objective\nDetect leaf diseases."]
            else:
                self.responses = ["ResNet50 reaches 97 percent [crops.pdf, Page 2]."]
            self.i = 0
            return super()._generate(messages, *a, **k)

    monkeypatch.setattr(rag_mod, "build_llm", lambda *a, **k: RoutedFake(responses=["x"]))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30)
    at.run()
    return at


def go(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_all_pages_render(app):
    assert not app.exception
    for page in ["📤 Upload Papers", "💬 Ask Questions", "📝 Summarize", "⚖️ Compare Papers", "🗂️ Document Library"]:
        go(app, page)


def test_chat_flow_shows_answer_and_sources(app):
    go(app, "💬 Ask Questions")
    app.chat_input[0].set_value("What accuracy does ResNet50 reach?").run()
    assert not app.exception
    assert any("97 percent" in m.value for m in app.markdown)
    assert any("crops.pdf, Page 2" in m.value for m in app.markdown)
    assert len(app.session_state["chat"]) == 2


def test_summary_and_comparison(app):
    go(app, "📝 Summarize")
    app.button[0].click().run()
    assert not app.exception and any("Research Objective" in m.value for m in app.markdown)
    go(app, "⚖️ Compare Papers")
    app.multiselect[0].set_value(["crops.pdf", "protein.pdf"]).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("deep learning" in m.value for m in app.markdown)
