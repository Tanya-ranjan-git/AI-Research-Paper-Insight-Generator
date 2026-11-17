import hashlib
import sys
from pathlib import Path

import pymupdf
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embeddings (offline, no model download).

    Texts that share words get similar vectors, which is enough to test retrieval logic.
    """
    DIM = 256

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.DIM, dtype="float32")
        for word in text.lower().split():
            word = word.strip(".,;:()[]\"'")
            if word:
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def make_pdf_bytes(pages: list[str]) -> bytes:
    """Build a small text PDF; an empty string makes a blank page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


PAPER_A = [
    "Deep learning for crop disease detection. The objective of this paper is to detect leaf "
    "diseases using convolutional neural networks trained on the PlantVillage dataset.",
    "Methodology: we fine-tune a ResNet50 model with data augmentation. The model reaches "
    "97 percent accuracy on the PlantVillage test split. A limitation is poor performance on field images.",
]
PAPER_B = [
    "Transformer models for protein folding. This study predicts protein structure using attention "
    "mechanisms trained on the Protein Data Bank.",
    "Results: the transformer achieves a median GDT score of 85 on benchmark targets. Future work "
    "includes extending the approach to RNA molecules.",
]


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


@pytest.fixture
def pdf_a():
    return make_pdf_bytes(PAPER_A)


@pytest.fixture
def pdf_b():
    return make_pdf_bytes(PAPER_B)
