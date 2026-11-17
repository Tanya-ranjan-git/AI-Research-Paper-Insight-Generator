"""Central configuration. All tunable values live here; secrets come from .env only."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

# Embeddings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking (characters, not tokens)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))

# Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))

# Limits
MAX_FILE_SIZE_MB = 50
MIN_CHARS_PER_PAGE = 20          # below this a page is treated as "no text" (likely scanned)
MAX_CONTEXT_CHARS_PER_PAPER = 60000  # cap text sent to Gemini for summary/compare


def get_gemini_api_key() -> str | None:
    """Return the API key from the environment (never log or print it)."""
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key or key == "your_gemini_api_key_here":
        return None
    return key
