"""Inspect extraction and complete E5 input sizes before building the index.

Run: python -m scripts.inspect_chunks
Only the tokenizer is needed here; embedding model weights are not loaded.
"""

from transformers import AutoTokenizer

from ragaz.chunk_documents import chunk_documents
from ragaz.config import EMBEDDING_MODEL_NAME, PDF_DIR
from ragaz.load_documents import load_documents
from ragaz.tokenization import count_tokens


def main() -> None:
    """Report page/chunk counts and fail if a complete chunk exceeds E5's limit."""
    documents = load_documents(PDF_DIR)
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    chunks = chunk_documents(documents, tokenizer)
    if not chunks:
        raise ValueError("No text chunks were extracted. Check the PDFs or use OCR.")

    # Count exactly what E5 will receive, including the prefix and special tokens.
    # chunk_documents already rejects any complete input above the model limit.
    token_counts = []
    for chunk in chunks:
        token_counts.append(count_tokens("passage: " + chunk.page_content, tokenizer))

    empty_pages = [doc.metadata for doc in documents if not doc.page_content.strip()]
    print(f"Loaded pages: {len(documents)}")
    print(f"Pages without extracted text: {len(empty_pages)}")
    print(f"Prepared chunks: {len(chunks)}")
    print(f"Largest complete chunk input: {max(token_counts)} tokens")
    print("All chunk inputs fit within E5's 512-token limit.")
    for metadata in empty_pages:
        print(f"Page requiring inspection or OCR: {metadata}")


if __name__ == "__main__":
    main()
