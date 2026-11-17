"""Sentence Transformers embeddings, wrapped for LangChain."""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


@lru_cache(maxsize=1)
def get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """Load the embedding model once and reuse it (loading takes seconds).

    Vectors are L2-normalised, so inner product == cosine similarity.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name or settings.EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
