"""Stage 1: extract PDF text while preserving its source and page number."""

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_documents(pdf_dir: Path) -> list[Document]:
    """Return one Document per page, in a predictable file/page order.

    A LangChain Document is text plus metadata, not necessarily an entire file.
    This loader extracts existing text; scanned pages need a separate OCR step.
    """
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    documents: list[Document] = []

    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)

        # Human-readable page numbers let us trace retrieved chunks to the PDF.
        for page_number, page in enumerate(reader.pages, start=1):
            document = Document(
                page_content=page.extract_text() or "",
                metadata={
                    "source": pdf_path.name,
                    "page": page_number,
                },
            )
            documents.append(document)

    # Return data; the caller decides whether to display it in a terminal or UI.
    return documents
