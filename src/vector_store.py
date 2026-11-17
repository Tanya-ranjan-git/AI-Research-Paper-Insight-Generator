"""FAISS vector store (via LangChain) with persistence, per-document delete and filtering."""
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import settings

INDEX_NAME = "index"


class VectorStoreManager:
    """Owns the FAISS index. Similarity metric: inner product on normalised vectors (= cosine)."""

    def __init__(self, embeddings: Embeddings, persist_dir: Path | None = None):
        self.embeddings = embeddings
        self.persist_dir = Path(persist_dir or settings.VECTOR_STORE_DIR)
        self.store: FAISS | None = None
        self.load()

    # ---- persistence -------------------------------------------------
    def load(self) -> bool:
        if not (self.persist_dir / f"{INDEX_NAME}.faiss").exists():
            return False
        try:
            # The index files are created by this app on this machine, so loading pickle is safe.
            self.store = FAISS.load_local(
                str(self.persist_dir), self.embeddings, index_name=INDEX_NAME,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
            return True
        except Exception:
            self.store = None
            return False

    def save(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if self.store is not None:
            self.store.save_local(str(self.persist_dir), index_name=INDEX_NAME)

    # ---- mutation ----------------------------------------------------
    def add_documents(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        ids = [d.metadata["chunk_id"] for d in docs]
        if self.store is None:
            self.store = FAISS.from_documents(
                docs, self.embeddings, ids=ids,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        else:
            self.store.add_documents(docs, ids=ids)
        self.save()
        return len(docs)

    def remove_document(self, document_name: str) -> int:
        """Delete every chunk belonging to one document."""
        if self.store is None:
            return 0
        ids = [i for i, d in self.store.docstore._dict.items()
               if d.metadata.get("document_name") == document_name]
        if ids:
            self.store.delete(ids)
            if not self.store.docstore._dict:
                self.clear()
            else:
                self.save()
        return len(ids)

    def clear(self) -> None:
        self.store = None
        if self.persist_dir.exists():
            for f in self.persist_dir.glob(f"{INDEX_NAME}.*"):
                f.unlink()

    def rebuild(self, docs: list[Document]) -> int:
        """Throw away the index and rebuild it from the given chunks."""
        self.clear()
        return self.add_documents(docs)

    # ---- queries -----------------------------------------------------
    def list_documents(self) -> dict[str, dict]:
        """{document_name: {"chunks": n, "pages": sorted list}}"""
        out: dict[str, dict] = {}
        if self.store is None:
            return out
        for d in self.store.docstore._dict.values():
            info = out.setdefault(d.metadata["document_name"], {"chunks": 0, "pages": set()})
            info["chunks"] += 1
            info["pages"].add(d.metadata["page"])
        for info in out.values():
            info["pages"] = sorted(info["pages"])
        return out

    def get_document_chunks(self, document_name: str) -> list[Document]:
        if self.store is None:
            return []
        docs = [d for d in self.store.docstore._dict.values()
                if d.metadata.get("document_name") == document_name]
        return sorted(docs, key=lambda d: (d.metadata["page"], d.metadata["chunk_id"]))

    def similarity_search(self, query: str, k: int | None = None,
                          document_names: list[str] | None = None) -> list[tuple[Document, float]]:
        """Return [(chunk, cosine_similarity)] best first, optionally restricted to documents."""
        if self.store is None or not query.strip():
            return []
        k = k or settings.TOP_K
        kwargs = {}
        if document_names:
            kwargs["filter"] = {"document_name": list(document_names)}
            kwargs["fetch_k"] = max(50, k * 10)
        return self.store.similarity_search_with_score(query, k=k, **kwargs)
