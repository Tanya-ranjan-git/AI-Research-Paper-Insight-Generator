# 📚 AI Research Paper Insight Generator: RAG Application

Upload research papers (PDF), ask questions, and get answers **grounded in the papers** with page-level
citations. Also generates structured summaries and side-by-side comparisons of multiple papers.

## Features
- **Multi-PDF upload** with page-by-page text extraction (PyMuPDF); scanned PDFs are detected with a clear message
- **Overlapping chunking** with metadata on every chunk: document name, page, chunk ID, source
- **Embeddings** with Sentence Transformers (`all-MiniLM-L6-v2`) stored in a local **FAISS** index (persisted to disk)
- **RAG Q&A** with LangChain + **Gemini**; answers cite `[Document Name, Page X]`; says so when the answer isn't in the papers
- **Source inspection**: expand the exact retrieved excerpts and similarity scores
- **Summaries** in 8 sections (objective, problem, methodology, dataset, findings, results, limitations, future scope)
- **Comparison** of 2+ papers: table (9 dimensions) + short comparative analysis
- **Library management**: remove a paper, rebuild or clear the index, clear conversation

## Tech stack
Python · PyMuPDF · LangChain · Sentence Transformers · FAISS · Google Gemini API · Streamlit · pytest

## RAG workflow
```
PDF upload -> PyMuPDF text per page -> clean + chunk (1000 chars, 150 overlap, metadata)
  -> MiniLM embeddings (384-d, normalised) -> FAISS index (saved in data/vector_store/)

Question -> embed question -> FAISS top-k similar chunks -> prompt = rules + chunks labelled
[Doc, Page X] + question -> Gemini -> answer with citations + expandable source excerpts
```

## Architecture
```
app.py                  Streamlit UI (6 pages) - only UI logic
config/settings.py      Settings + .env loading
src/pdf_processor.py    PDF -> pages (name, page number, text)
src/text_chunker.py     pages -> overlapping chunks + metadata
src/embeddings.py       Sentence Transformers via LangChain
src/vector_store.py     FAISS index: add / search / filter / delete / persist
src/rag_pipeline.py     retrieve -> prompt -> Gemini (LCEL chain) -> RAGAnswer
src/summarizer.py       structured paper summary
src/paper_comparator.py multi-paper table + analysis (JSON from Gemini)
utils/prompts.py        all prompt templates
utils/helpers.py        text cleaning, file validation, citation format
tests/                  pytest suite
```

## Installation (Windows, VS Code)
Requires Python 3.10+ (3.11/3.12 recommended). In the VS Code terminal (PowerShell):
```powershell
cd AI-Research-Paper-Insight-Generator
python -m venv venv
venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
copy .env.example .env             # then open .env and paste your Gemini API key
```
Get a free key at <https://aistudio.google.com/apikey>. The first run downloads the embedding model (~90 MB).
(macOS/Linux: `source venv/bin/activate` and `cp .env.example .env`.)

## Run
```powershell
streamlit run app.py
```
Open the URL shown (usually http://localhost:8501). Upload PDFs -> **Process & index** -> ask questions.

## Sample questions
- What is the main objective of this paper?
- Which dataset was used and how large is it?
- What evaluation metrics are reported, and what were the results?
- What limitations do the authors mention?
- How does the methodology of Paper A differ from Paper B? (select both papers)
- Who won the 1998 World Cup? *(should answer that it is not in the papers - tests anti-hallucination)*

## Tests
```powershell
python -m pytest -v
```
Tests use deterministic fake embeddings and a fake LLM, so they run offline and need no API key. They cover PDF
extraction, chunking and metadata, FAISS add/search/filter/delete/persist, the RAG chain, citations,
summarizer, comparator, API-key leak protection, and a headless Streamlit UI smoke test.
Live Gemini calls and the real MiniLM model are **not** covered by automated tests - verify them by running the app.

## Screenshots
Add your own after running the app (save in `screenshots/` and link here):
| Page | Screenshot |
|---|---|
| Dashboard | _add `screenshots/home.png`_ |
| Q&A with citations | _add `screenshots/chat.png`_ |
| Comparison table | _add `screenshots/compare.png`_ |

## Limitations
- No OCR: scanned PDFs are rejected with a message
- Chunk size is in characters, not tokens; tables/equations/figures extract poorly
- Summaries/comparisons send paper text to Gemini (long papers keep start + end, drop the middle)
- Citations come from the model: labels are provided in the prompt, but are not programmatically verified
- Single-user, local app (no authentication)

## Future improvements
OCR (OCRmyPDF/Tesseract), hybrid BM25 + vector search, reranking, token-aware or section-aware chunking,
citation verification, streaming answers, user accounts and cloud deployment.

## Publish to GitHub
```powershell
git init -b main
git add .
git status                         # confirm .env and venv/ are NOT listed
git commit -m "Initial commit: AI Research Paper Insight Generator"
git remote add origin https://github.com/<your-username>/AI-Research-Paper-Insight-Generator.git
git push -u origin main
```
Create the empty repository on github.com first (no README/license). Never commit `.env`.

## License
MIT - see [LICENSE](LICENSE). Replace `<YOUR NAME>` in the license file.
