"""Stage 3: adapt Sentence Transformers to LangChain's embedding interface."""

import os

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from ragaz.config import EMBEDDING_MODEL_NAME


class E5Embeddings(Embeddings):
    """Encode passages and questions in the same 384-dimensional E5 space.

    Construct this object once and reuse it. Sentence Transformers runs this
    embedding model locally, independently of the generator in LM Studio.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str | None = None) -> None:
        """Load one encoder/tokenizer pair on the explicit or selected device."""
        # Prefer an explicit device, then the environment. With neither set,
        # Sentence Transformers selects an available device, including CPU.
        device = device or os.getenv("EMBEDDING_DEVICE") or None
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode N raw passages and return N vectors as Python lists."""
        if not texts:
            return []
        passages = []
        for text in texts:
            # E5 requires this prefix even for non-English passages.
            passages.append("passage: " + text)
        embeddings = self.model.encode(passages, normalize_embeddings=True)
        # Normalization gives unit-length vectors. tolist() converts the NumPy
        # container to the nested lists expected by LangChain, preserving values.
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encode one raw question and return a single vector."""
        query = "query: " + text
        # A string produces shape (384,); N strings produce shape (N, 384).
        # Callers provide raw text: this adapter owns both E5 prefixes.
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()
