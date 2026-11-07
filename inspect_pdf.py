from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document


project_dir = Path(__file__).resolve().parent
pdf_path = project_dir / "data" / "raw" / "transformer.pdf"

reader = PdfReader(pdf_path)
text = reader.pages[0].extract_text() or ""

document = Document(
    page_content=text,
    metadata={
        "source": pdf_path.name,
        "page": 1,
    },
)

print("page number:", len(reader.pages))
print("Metadatas:", document.metadata)
print("extracted characters:", len(document.page_content))
print("\nPreview of the text:")
print(document.page_content[:1200])