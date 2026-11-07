from pathlib import Path
from load_documents import load_documents
if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    documents = load_documents(project_dir / "data" / "raw")
    output_dir = project_dir / "data" / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    for doc in documents:
        source = doc.metadata["source"]
        page = doc.metadata["page"]
        header = f"FILE: {source} | PAGINA: {page}\n\n"
        blocks.append(header + doc.page_content)

    output_text = "\n\n----------\n\n".join(blocks)
    output_path = output_dir / "pages.txt"
    output_path.write_text(output_text, encoding="utf-8")
    print("\nPagine esportate:", len(documents))
    print("File creato:", output_path)