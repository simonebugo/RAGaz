"""Shared defaults: keep indexing and querying in the same embedding space."""

from pathlib import Path

# Resolve paths from this package rather than the current terminal directory.
# Both entry points must open the same database and collection.
PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_DIR / "data" / "raw"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma"
COLLECTION_NAME = "ragaz_e5_small_v1"

# Tokenization, document encoding, and query encoding must use the same model.
# Changing the embedding model requires rebuilding the stored vectors.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
E5_MAX_TOKENS = 512
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
TOP_K = 3

# Changing the generator does not require re-embedding the PDFs.
LOCAL_MODEL_NAME = "qwen/qwen3.8-27b"
LOCAL_BASE_URL = "http://127.0.0.1:1234/v1"
OPENAI_MODEL_NAME = "gpt-6-luna"
DEFAULT_QUESTION = "Why can increasing chunk overlap increase redundancy?"
