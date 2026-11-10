"""Offline pipeline: PDF pages -> chunks -> E5 embeddings -> persisted Chroma.

Run: python -m scripts.index_documents
Index when preparing the knowledge base, not for every question.
"""

from dotenv import load_dotenv

from ragaz.chunk_documents import chunk_documents
from ragaz.config import PDF_DIR, PROJECT_DIR
from ragaz.embeddings import E5Embeddings
from ragaz.load_documents import load_documents
from ragaz.vector_store import open_vector_store


def main() -> None:
    """Build the example collection using the shared model and chunk settings."""
    load_dotenv(PROJECT_DIR / ".env")

    # Stages 1-2: load page Documents, then create smaller Documents that retain
    # the source/page metadata needed to trace evidence later.
    documents = load_documents(PDF_DIR)
    embedding_model = E5Embeddings()
    tokenizer = embedding_model.model.tokenizer
    chunks = chunk_documents(documents, tokenizer)
    if not chunks:
        raise ValueError("No text chunks were extracted. Check the PDFs or use OCR.")

    # Preserve the original IDs: the SAME unchanged corpus updates existing
    # entries rather than creating duplicates. Global positions are not a full
    # incremental-ingestion strategy; changes to files or splitting can shift
    # IDs or leave old chunks behind. See the README before changing the corpus.
    chunk_ids = []
    for index, chunk in enumerate(chunks):
        source = chunk.metadata["source"]
        page = chunk.metadata["page"]
        chunk_ids.append(f"{source}:{page}:{index}")

    # Stages 3-4: the integration calls embed_documents internally and persists
    # text, metadata, and vectors. The adapter already adds the 'passage: ' prefix.
    vector_store = open_vector_store(embedding_model, create_if_missing=True)
    vector_store.add_documents(documents=chunks, ids=chunk_ids)

    # get() returns a dictionary. Count its IDs, not its fields; include=[] avoids
    # fetching all stored text just to count the entries.
    stored_ids = vector_store.get(include=[])["ids"]
    print(f"Loaded pages: {len(documents)}")
    print(f"Prepared chunks: {len(chunks)}")
    print(f"Stored chunks: {len(stored_ids)}")


if __name__ == "__main__":
    main()
