"""Streamlit UI for the AI Research Paper Insight Generator. Run: streamlit run app.py"""
import re

import streamlit as st

from config import settings
from src.pdf_processor import PDFProcessingError, extract_pdf, scanned_pdf_message
from src.paper_comparator import PaperComparator
from src.rag_pipeline import LLMCallError, LLMConfigError, RAGPipeline, build_llm
from src.summarizer import PaperSummarizer
from src.text_chunker import chunk_extraction
from src.vector_store import VectorStoreManager
from utils.helpers import safe_filename, validate_pdf_bytes
from utils.prompts import SUMMARY_SECTIONS

st.set_page_config(page_title="AI Research Paper Insight Generator", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); padding: 14px 18px; border-radius: 12px;}
      .cite {display:inline-block; padding:2px 10px; margin:2px 4px 2px 0; border-radius:12px;
             background: rgba(76,110,245,0.15); font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = ["🏠 Home", "📤 Upload Papers", "💬 Ask Questions", "📝 Summarize", "⚖️ Compare Papers", "🗂️ Document Library"]


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)...")
def load_vector_manager() -> VectorStoreManager:
    from src.embeddings import get_embeddings
    return VectorStoreManager(get_embeddings())


def get_vm() -> VectorStoreManager | None:
    try:
        return load_vector_manager()
    except Exception as exc:
        st.error(
            "Could not load the embedding model. Check your internet connection (needed on first run) "
            f"and that sentence-transformers is installed. Details: {type(exc).__name__}"
        )
        return None


def get_llm():
    """Create Gemini client once per session; show a setup message if the key is missing."""
    if "llm" not in st.session_state:
        try:
            st.session_state["llm"] = build_llm()
        except LLMConfigError as exc:
            st.warning(str(exc))
            return None
        except Exception as exc:
            st.error(f"Could not create the Gemini client ({type(exc).__name__}).")
            return None
    return st.session_state["llm"]


def init_state() -> None:
    st.session_state.setdefault("chat", [])        # [{"role", "content", "sources", "found"}]
    st.session_state.setdefault("summaries", {})   # {doc_name: markdown}
    st.session_state.setdefault("comparison", None)


def cite_badges(citations: list[str]) -> str:
    return "".join(f'<span class="cite">{c}</span>' for c in citations)


# ---------------------------------------------------------------- processing helpers
def process_uploads(vm: VectorStoreManager, files) -> None:
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    existing = set(vm.list_documents())
    for f in files:
        name = safe_filename(f.name)
        with st.status(f"Processing {name}", expanded=True) as status:
            try:
                data = f.getvalue()
                validate_pdf_bytes(data, name)
                if name in existing:
                    status.update(label=f"{name}: already indexed - skipped", state="complete")
                    continue
                st.write("Extracting text with PyMuPDF...")
                extraction = extract_pdf(data, name)
                if extraction.is_probably_scanned:
                    status.update(label=f"{name}: no text found", state="error")
                    st.error(scanned_pdf_message(name))
                    continue
                st.write("Chunking text...")
                chunks = chunk_extraction(extraction)
                st.write(f"Embedding {len(chunks)} chunks and storing in FAISS...")
                vm.add_documents(chunks)
                (settings.UPLOAD_DIR / name).write_bytes(data)
                msg = f"{name}: {extraction.total_pages} pages, {len(chunks)} chunks indexed"
                if extraction.empty_pages:
                    msg += f" ({extraction.empty_pages} page(s) had no text)"
                status.update(label=msg, state="complete")
            except (ValueError, PDFProcessingError) as exc:
                status.update(label=f"{name}: failed", state="error")
                st.error(str(exc))
            except Exception as exc:
                status.update(label=f"{name}: failed", state="error")
                st.error(f"Unexpected error while processing {name}: {type(exc).__name__}")


def rebuild_index(vm: VectorStoreManager) -> tuple[int, list[str]]:
    all_chunks, problems = [], []
    for path in sorted(settings.UPLOAD_DIR.glob("*.pdf")):
        try:
            ex = extract_pdf(path)
            if ex.has_text:
                all_chunks.extend(chunk_extraction(ex))
            else:
                problems.append(scanned_pdf_message(path.name))
        except PDFProcessingError as exc:
            problems.append(f"{path.name}: {exc}")
    return vm.rebuild(all_chunks), problems


# ---------------------------------------------------------------- pages
def page_home(vm):
    st.title("📚 AI Research Paper Insight Generator")
    st.caption("Upload research papers, ask grounded questions, summarize and compare - with page-level citations.")
    docs = vm.list_documents() if vm else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers indexed", len(docs))
    c2.metric("Text chunks", sum(d["chunks"] for d in docs.values()))
    c3.metric("Questions asked", sum(1 for m in st.session_state["chat"] if m["role"] == "user"))
    c4.metric("Gemini key", "Configured" if settings.get_gemini_api_key() else "Missing")
    st.subheader("How it works")
    st.markdown(
        "1. **Upload** PDFs → text is extracted page by page (PyMuPDF)\n"
        "2. Text is **chunked** and **embedded** (Sentence Transformers, all-MiniLM-L6-v2)\n"
        "3. Vectors are stored in **FAISS**\n"
        "4. Your question retrieves the most similar chunks → **Gemini** answers using only that context\n"
        "5. Every answer shows **[Document, Page X]** citations and the source excerpts"
    )
    if not settings.get_gemini_api_key():
        st.info("Add your Gemini API key to `.env` (see `.env.example`) to enable Q&A, summaries and comparison.")
    if not docs:
        st.info("No papers yet - open **Upload Papers** in the sidebar to get started.")


def page_upload(vm):
    st.header("📤 Upload Research Papers")
    files = st.file_uploader("Select one or more PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Process & index", type="primary", disabled=not files or vm is None):
        process_uploads(vm, files)
        st.success("Done. Head to **Ask Questions** to chat with your papers.")
    st.caption("Scanned (image-only) PDFs are detected and reported; OCR is not included.")


def page_chat(vm):
    st.header("💬 Ask Questions")
    docs = vm.list_documents() if vm else {}
    if not docs:
        st.info("Upload and index at least one paper first.")
        return
    top = st.columns([3, 1, 1])
    scope = top[0].multiselect("Search in (empty = all papers)", list(docs))
    k = top[1].slider("Chunks (top-k)", 2, 10, settings.TOP_K)
    if top[2].button("🧹 Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    question = st.chat_input("Ask about your papers...")
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    llm = get_llm()
    if llm is None:
        return
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving context and asking Gemini..."):
                result = RAGPipeline(vm, llm).ask(
                    question, history=st.session_state["chat"], k=k, document_names=scope or None)
        except (LLMCallError, ValueError) as exc:
            st.error(str(exc))
            return
        st.markdown(result.answer)
        render_sources(result.sources)
    st.session_state["chat"] += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": result.answer, "sources": result.sources, "found": result.found},
    ]


def render_sources(sources):
    seen, cites = set(), []
    for s in sources:
        if s.citation not in seen:
            seen.add(s.citation)
            cites.append(s.citation)
    st.markdown("**Retrieved from:** " + cite_badges(cites), unsafe_allow_html=True)
    with st.expander(f"Inspect source excerpts ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**{i}. {s.citation}**  ·  similarity {s.score:.2f}")
            st.text(s.text)


def page_summarize(vm):
    st.header("📝 Paper Summarization")
    docs = vm.list_documents() if vm else {}
    if not docs:
        st.info("Upload and index at least one paper first.")
        return
    name = st.selectbox("Choose a paper", list(docs))
    st.caption("Sections: " + " · ".join(SUMMARY_SECTIONS))
    if st.button("Generate summary", type="primary"):
        llm = get_llm()
        if llm:
            try:
                with st.spinner("Summarizing with Gemini..."):
                    st.session_state["summaries"][name] = PaperSummarizer(vm, llm).summarize(name)
            except (LLMCallError, ValueError) as exc:
                st.error(str(exc))
    if name in st.session_state["summaries"]:
        st.markdown(st.session_state["summaries"][name])
        st.download_button("Download summary (.md)", st.session_state["summaries"][name],
                           file_name=f"{name}_summary.md")


def page_compare(vm):
    st.header("⚖️ Compare Research Papers")
    docs = vm.list_documents() if vm else {}
    if len(docs) < 2:
        st.info("Upload at least two papers to compare them.")
        return
    chosen = st.multiselect("Select two or more papers", list(docs))
    if st.button("Compare", type="primary", disabled=len(chosen) < 2):
        llm = get_llm()
        if llm:
            try:
                with st.spinner("Comparing papers with Gemini..."):
                    st.session_state["comparison"] = (chosen, PaperComparator(vm, llm).compare(chosen))
            except (LLMCallError, ValueError) as exc:
                st.error(str(exc))
    if st.session_state["comparison"]:
        names, result = st.session_state["comparison"]
        st.subheader("Comparison table")
        st.dataframe(result.table, width="stretch")
        st.subheader("Comparative analysis")
        st.markdown(result.analysis)
        st.download_button("Download table (.csv)", result.table.to_csv().encode("utf-8"),
                           file_name="paper_comparison.csv")


def page_library(vm):
    st.header("🗂️ Document Library")
    docs = vm.list_documents() if vm else {}
    if not docs:
        st.info("No documents indexed.")
    for name, info in docs.items():
        c = st.columns([4, 2, 2, 1])
        c[0].markdown(f"**{name}**")
        c[1].write(f"{info['chunks']} chunks")
        c[2].write(f"{len(info['pages'])} pages with text")
        if c[3].button("🗑️", key=f"rm_{name}", help="Remove this paper"):
            vm.remove_document(name)
            (settings.UPLOAD_DIR / name).unlink(missing_ok=True)
            st.session_state["summaries"].pop(name, None)
            st.session_state["comparison"] = None
            st.rerun()
    st.divider()
    b1, b2 = st.columns(2)
    if b1.button("🔄 Rebuild index from uploaded files", disabled=vm is None):
        with st.spinner("Rebuilding FAISS index..."):
            n, problems = rebuild_index(vm)
        st.success(f"Index rebuilt with {n} chunks.")
        for p in problems:
            st.warning(p)
    if b2.button("🧨 Clear entire index", disabled=vm is None):
        vm.clear()
        st.session_state["summaries"], st.session_state["comparison"] = {}, None
        st.success("Vector index cleared. Uploaded files are kept - use Rebuild to re-index them.")
        st.rerun()


def main():
    init_state()
    st.sidebar.title("📚 Paper Insights")
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(f"Model: {settings.GEMINI_MODEL}\n\nEmbeddings: all-MiniLM-L6-v2")
    vm = get_vm()
    {
        PAGES[0]: page_home, PAGES[1]: page_upload, PAGES[2]: page_chat,
        PAGES[3]: page_summarize, PAGES[4]: page_compare, PAGES[5]: page_library,
    }[page](vm)


main()
