# Manage one PDF at a time for the Streamlit interface. Chroma stores one record
# per chunk, and the source filename groups the chunks that belong to a PDF.
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from transformers import PreTrainedTokenizerBase

from ragaz.chunk_documents import chunk_documents
from ragaz.load_documents import load_pdf
from shutil import rmtree
from tempfile import mkdtemp
from threading import Lock

from ragaz.config import UPLOAD_DIR

# Serialize upload and removal operations within the Streamlit process.
_index_lock = Lock()


def prepare_pdf_chunks(pdf_path: Path, tokenizer: PreTrainedTokenizerBase,) -> tuple[list[Document], list[str]]:
    pages = load_pdf(pdf_path)
    chunks = chunk_documents(pages, tokenizer)

    # A searchable PDF needs at least one chunk of extractable text.
    if not chunks:
        raise ValueError(f"No text chunks extracted from {pdf_path.name}.")

    # The page number comes from the PDF loader. The zero-based chunk counter
    # starts over for each PDF, keeping its IDs independent of the other files.
    chunk_ids = []
    for index, chunk in enumerate(chunks):
        page = chunk.metadata["page"]
        chunk_ids.append(f"{pdf_path.name}:{page}:{index}")

    return chunks, chunk_ids

def index_pdf( pdf_path: Path, tokenizer: PreTrainedTokenizerBase, vector_store: Chroma,) -> int:
    # Check the filename in metadata so this also finds records created by the
    # batch indexer, which uses a different chunk numbering scheme.
    existing_ids = vector_store.get(where={"source": pdf_path.name},include=[])["ids"]

    if existing_ids:
        raise ValueError(f"{pdf_path.name} is already indexed.")

    chunks, chunk_ids = prepare_pdf_chunks(pdf_path, tokenizer)
    # The Chroma integration calls E5 and saves each chunk with its ID and vector.
    vector_store.add_documents(documents=chunks, ids=chunk_ids)

    return len(chunk_ids)

def save_and_index_pdf(filename: str, content: bytes, tokenizer: PreTrainedTokenizerBase, vector_store: Chroma) -> int:
    # Validate the uploaded name before using it to build a local path.
    name = Path(filename)
    if name.name != filename or name.suffix.lower() != ".pdf":
        raise ValueError("Choose a PDF with a valid file name.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # A separate directory for each upload keeps the original filename and
    # gives each copy its own destination on disk.
    upload_dir = Path(mkdtemp(dir=UPLOAD_DIR))

    try:
        pdf_path = upload_dir / filename
        pdf_path.write_bytes(content)

        # Keep the duplicate check and the Chroma write inside the same lock.
        with _index_lock:
            return index_pdf(pdf_path, tokenizer, vector_store)
    except Exception:
        # Discard the copy created by this attempt and propagate the error.
        rmtree(upload_dir)
        raise


def list_indexed_pdfs(vector_store: Chroma) -> list[str]:
    metadatas = vector_store.get(include=["metadatas"])["metadatas"]

    # A set turns the per-chunk source values into one entry per PDF.
    filenames = {
        metadata["source"]
        for metadata in metadatas
        if metadata and "source" in metadata
    }
    return sorted(filenames)


def remove_pdf(filename: str, vector_store: Chroma) -> int:
    # Look up every chunk ID for the selected source before deleting the records.
    with _index_lock:
        ids = vector_store.get(
            where={"source": filename},
            include=[],
        )["ids"]

        if ids:
            vector_store.delete(ids=ids)

    return len(ids)