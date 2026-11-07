from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document


def load_documents(pdf_dir: Path) -> list[Document]:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"Nessun PDF trovato in: {pdf_dir}")

    documents = []

    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):
            document = Document(
                page_content=page.extract_text() or "",
                metadata={
                    "source": pdf_path.name,
                    "page": page_number,
                },
            )
            documents.append(document)

        print(f"{pdf_path.name}: {len(reader.pages)} pagine")

    return documents


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    pdf_dir = project_dir / "data" / "raw"

    documents = load_documents(pdf_dir)

    empty_pages = []

    for doc in documents:
        text = doc.page_content.strip()
        if not text:
            empty_pages.append(doc.metadata)

    print("\nTotale pagine caricate:", len(documents))
    print("Pagine senza testo:", len(empty_pages))

    if empty_pages:
        print("Pagine da controllare:", empty_pages)