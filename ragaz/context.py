"""Stage 6: turn retrieved Documents into readable, numbered evidence."""

from langchain_core.documents import Document


def format_documents(documents: list[Document]) -> str:
    """Preserve retrieval order and attach source/page references.

    Reference numbers belong to this answer, not to permanent database entries.
    Different passages can come from the same page and still need distinct IDs.
    """
    blocks = []
    for number, document in enumerate(documents, start=1):
        source = document.metadata["source"]
        page = document.metadata["page"]
        blocks.append(
            f"[{number}] Source: {source} | Page: {page}\n{document.page_content}"
        )

    # Keep a visible boundary between passages instead of merging their text.
    return "\n\n".join(blocks)

