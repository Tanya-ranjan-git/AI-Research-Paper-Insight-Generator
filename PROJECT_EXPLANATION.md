# Project Explanation & Interview Notes

## 1. What problem does it solve?
Reading many long research papers to find a method, a dataset or a result is slow. This app lets you ask
questions in plain English and get answers **from your own PDFs**, with `[Document, Page X]` citations, plus
one-click summaries and multi-paper comparisons.

## 2. Why RAG?
A plain LLM answers from training memory: it hasn't seen your papers and may invent details (hallucination).
RAG first **retrieves** the relevant passages from your documents and gives them to the LLM, so the answer is
grounded and citeable. Fine-tuning would be costly and wouldn't give page citations.

## 3. Traditional LLM app vs RAG
| Plain LLM | RAG |
|---|---|
| Question -> LLM -> answer from memory | Question -> search your documents -> LLM answers from those passages |
| No private knowledge, can hallucinate | Uses your data, can cite pages, can say "not found" |

## 4. Why PyMuPDF?
Fast, accurate text extraction page by page (`page.get_text()`), which is what lets me keep page numbers.
(`import pymupdf` is used because the older `import fitz` name is deprecated.)

## 5. What are embeddings?
A model turns text into a list of numbers (here 384) so that texts with similar *meaning* get nearby vectors.
"How accurate is the model?" lands near a chunk saying "reaches 97% accuracy" even without identical words.

## 6. Why Sentence Transformers (all-MiniLM-L6-v2)?
Small (~90 MB), fast on CPU, free and local (no API cost, papers aren't sent out for embedding), good quality
for semantic search. Vectors are normalised so inner product equals cosine similarity.

## 7. What is FAISS and how does search work?
FAISS (Facebook AI Similarity Search) stores vectors and finds the nearest ones quickly. I embed the question,
FAISS returns the top-k chunks with the highest similarity score (`MAX_INNER_PRODUCT` on normalised vectors =
cosine, higher is better). It runs locally and is saved to `data/vector_store/`.

## 8. LangChain's role
Glue: `RecursiveCharacterTextSplitter` (chunking), `HuggingFaceEmbeddings`, the `FAISS` vector store wrapper,
`ChatPromptTemplate`, `ChatGoogleGenerativeAI`, and an LCEL chain `prompt | llm | StrOutputParser()`.
Swapping Gemini for another LLM means changing one line (`build_llm`).

## 9. How does Gemini generate answers?
`RAGPipeline.ask()` builds a prompt = rules ("use ONLY the context, cite labels, say not found otherwise") +
the retrieved chunks, each prefixed `[Doc, Page X]` + recent chat history + the question. Gemini writes the answer
from that context only. Temperature is low (0.2) for factual output.

## 10. How are page-level citations implemented?
Page-by-page extraction -> each page chunked separately -> every chunk carries `document_name` and `page` metadata ->
FAISS returns chunks with metadata -> `SourceExcerpt.citation` formats `[Doc, Page X]` -> the prompt labels each
excerpt -> the UI shows the retrieved labels and full excerpts. *Honest caveat:* the labels shown under the answer
come from retrieval (reliable); the inline citations in the answer text are written by the model and not verified.

## 11. Multi-document retrieval
All papers share one FAISS index; each chunk knows its document. The UI can restrict search to selected papers via
a metadata filter (`document_names`). Removing a paper deletes only its chunk IDs.

## 12. Challenges faced (real ones from this build)
- Keeping accurate page numbers -> chunk per page instead of whole document.
- Scanned PDFs return empty text -> detect (`is_probably_scanned`) and show a clear message.
- Deleting one paper from FAISS -> use stable chunk IDs (`doc::chunk-N`) so they can be deleted.
- Gemini returning JSON for the comparison table with code fences -> tolerant `parse_json_response` + fallback error.
- Not leaking secrets -> `safe_invoke` maps provider errors to safe messages; tested.
- Testing without network/API -> fake embeddings and fake LLM in tests.

## 13. Limitations
No OCR; character-based chunking; tables/equations extract poorly; long papers are truncated (start + end kept) for
summary/compare; citations in answer text aren't verified; no authentication; top-k retrieval can miss info that
spans many sections.

## 14. Improvements
OCR, hybrid BM25+vector search, reranker, section-aware chunking, citation verification, streaming, auth, cloud deploy.

## Workflow (start to finish)
1. Upload -> `validate_pdf_bytes` -> `extract_pdf` (pages) -> `chunk_extraction` (chunks + metadata)
2. `VectorStoreManager.add_documents` embeds and stores chunks in FAISS, saves to disk
3. Question -> `RAGPipeline.ask` -> `retrieve` -> `format_context` -> `RAG_PROMPT | llm` -> `RAGAnswer`
4. UI renders answer, citation badges, expandable excerpts; history in `st.session_state["chat"]`

## Important functions
| Function / class | File | Responsibility |
|---|---|---|
| `extract_pdf` | src/pdf_processor.py | Bytes/path -> `ExtractionResult` of pages with numbers; flags scanned PDFs |
| `chunk_extraction` | src/text_chunker.py | Pages -> overlapping `Document` chunks with metadata |
| `get_embeddings` | src/embeddings.py | Cached MiniLM model, normalised vectors |
| `VectorStoreManager` | src/vector_store.py | Add, search (with filter), remove, clear, rebuild, persist |
| `RAGPipeline.ask` | src/rag_pipeline.py | Retrieve -> prompt -> Gemini -> answer + sources |
| `safe_invoke` | src/rag_pipeline.py | Run chain; convert errors to key-safe messages |
| `build_paper_text` | src/summarizer.py | Page-labelled paper text, truncated to budget |
| `PaperSummarizer.summarize` | src/summarizer.py | 8-section Markdown summary |
| `PaperComparator.compare` | src/paper_comparator.py | JSON from Gemini -> DataFrame table + analysis |
| `main` and `page_*` | app.py | Streamlit navigation and pages |

## 20 interview questions
1. **Explain the project in one line.** A RAG app that answers questions about uploaded research PDFs with page citations and also summarizes and compares papers.
2. **What is RAG?** Retrieve relevant text from my documents first, then have the LLM answer using only that text.
3. **Why not just paste the PDF into Gemini?** Long papers/many papers exceed or waste context, cost more, and give no controlled citations; retrieval sends only relevant chunks. (For summary/compare I do send whole paper text, truncated, because those need global view.)
4. **How do you reduce hallucination?** Strict prompt (use only context, say "not found"), low temperature, retrieved excerpts shown to the user, empty retrieval short-circuits before calling the LLM.
5. **What is chunking and why?** Splitting text into pieces small enough to embed precisely and fit in the prompt; each piece should be about one idea.
6. **Why overlap?** So a sentence cut at a chunk boundary still appears whole in a neighbouring chunk. I use 1000 chars with 150 overlap.
7. **How did you choose chunk size?** Reasonable default (about 150-200 words); it's configurable in `.env`. I haven't benchmarked alternatives.
8. **Why chunk per page?** To keep exact page numbers for citations; trade-off: ideas spanning pages get split.
9. **What is an embedding?** A fixed-length vector (384 numbers) representing meaning.
10. **Why normalise vectors?** Then inner product equals cosine similarity, which compares direction (meaning), not length.
11. **How does FAISS find chunks?** Compares the query vector with stored vectors and returns the top-k highest similarity. With a small index this is exact search.
12. **What is top-k?** Number of chunks retrieved (default 5). Too low misses info; too high adds noise and cost.
13. **Why FAISS and not a hosted DB?** Local, free, no server, enough for a personal tool. A hosted DB would suit multi-user.
14. **How does LangChain help?** Provides standard components (splitter, embeddings, vector store, prompts, LLM wrappers) and LCEL chaining, making parts swappable.
15. **What is LCEL?** LangChain Expression Language: composing steps with `|`, e.g. `prompt | llm | StrOutputParser()`.
16. **How do you get a structured table from an LLM?** Prompt for strict JSON with fixed keys, parse it tolerantly, fill missing cells with "Not mentioned", show as a DataFrame.
17. **How do you handle API errors and secrets?** Key only in `.env` (git-ignored), `.env.example` in repo, errors mapped to safe messages, a test verifies keys aren't leaked.
18. **What is Streamlit session state?** A per-user dict that survives reruns; I store chat history, summaries and the last comparison there. Heavy objects (index, model) use `st.cache_resource`.
19. **How did you test it?** pytest with fake embeddings and a fake LLM for offline, deterministic tests, plus a headless Streamlit `AppTest`. Live Gemini and the real MiniLM model I verify manually.
20. **What would you improve first?** Hybrid search plus a reranker for retrieval quality, then OCR, then citation verification.

## 1-minute explanation
"I built an AI Research Paper Insight Generator, a Retrieval-Augmented Generation app in Python. Users upload
research PDFs; I extract text page by page with PyMuPDF, split it into overlapping chunks that keep the document
name and page number, embed them with Sentence Transformers, and store them in a local FAISS index. When a user
asks a question, I retrieve the most similar chunks and pass them to Gemini through a LangChain chain with a strict
prompt: answer only from the context and cite the page, or say the information isn't available. The Streamlit UI
shows the answer with page citations and lets you inspect the exact excerpts. It also generates structured
summaries and compares multiple papers in a table. I covered the core modules with automated tests, and I'm aware
of limits like no OCR and unverified model-written citations, which I'd address with OCR and a reranker."
