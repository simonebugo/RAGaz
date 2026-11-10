"""Stage 4: share one Chroma configuration between indexing and retrieval."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from ragaz.config import CHROMA_DIR, COLLECTION_NAME


def open_vector_store(
    embedding_model: Embeddings,
    db_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    *,
    create_if_missing: bool = False,
) -> Chroma:
    """Open a persisted collection; allow creation only during indexing.

    A collection stores chunk IDs, text, metadata, and embeddings. Supply the
    embedding object again when reopening: new questions must be encoded with
    the same model and preprocessing as the stored passages.
    """
    if not create_if_missing and not db_dir.exists():
        raise FileNotFoundError(
            f"Vector database not found at {db_dir}. "
            "Run 'python -m scripts.index_documents' first."
        )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(db_dir),
        # Cosine distance is 1 - cosine similarity: smaller means closer.
        collection_configuration={"hnsw": {"space": "cosine"}},
        create_collection_if_not_exists=create_if_missing,
    )
