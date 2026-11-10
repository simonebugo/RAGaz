"""Online pipeline: ask a question against the already indexed knowledge base.

Run: python -m scripts.search_chroma "Your question about the PDFs"
"""

import argparse

from dotenv import load_dotenv

from ragaz.config import DEFAULT_QUESTION, PROJECT_DIR, TOP_K
from ragaz.embeddings import E5Embeddings
from ragaz.generation import build_local_llm, build_openai_llm
from ragaz.pipeline import answer_question
from ragaz.vector_store import open_vector_store


def main() -> None:
    """Create reusable resources, ask once, and display the answer and sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--k", type=int, default=TOP_K, help="Number of passages to retrieve."
    )
    parser.add_argument(
        "--provider", choices=("local", "openai"), default="local",
        help="Use local Qwen by default, or opt into the OpenAI API.",
    )
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("The question must not be empty.")
    if args.k < 1:
        parser.error("--k must be a positive integer.")

    load_dotenv(PROJECT_DIR / ".env")
    embedding_model = E5Embeddings()
    vector_store = open_vector_store(embedding_model)

    # A retriever returns Documents, not (Document, distance) tuples. Retrieval
    # happens inside answer_question; invoking it here too would repeat the work.
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": args.k}
    )
    if args.provider == "local":
        llm = build_local_llm()
    else:
        llm = build_openai_llm()

    result = answer_question(query=args.question, retriever=retriever, llm=llm)
    print(result["answer"])

    # Match the numbering used in the prompt. A plausible-looking reference
    # still needs to be checked against its passage for factual support.
    print("\nRetrieved sources:")
    for number, document in enumerate(result["documents"], start=1):
        print(
            f"[{number}] {document.metadata['source']} "
            f"| Page: {document.metadata['page']}"
        )


if __name__ == "__main__":
    main()
