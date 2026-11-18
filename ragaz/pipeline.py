"""Online RAG: retrieve evidence, assemble a prompt, generate an answer."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from ragaz.context import format_documents
from ragaz.prompts import build_rag_prompt


def answer_question(query: str, retriever: BaseRetriever, llm: BaseChatModel) -> dict:
    """Answer one independent question and return its source Documents.

    The caller creates and reuses the retriever and model clients. This function
    has no PDF loading, indexing, or terminal/UI dependencies. A future interface
    can display both the answer and the retrieved source passages.
    """
    query = query.strip()
    if not query:
        raise ValueError("The question must not be empty.")

    # Stage 5: the retriever embeds the question and searches the saved vectors.
    # This is the only retrieval call made for this question.
    retrieved_documents = retriever.invoke(query)
    if not retrieved_documents:
        return {
            "answer": "No passages were retrieved. Index documents before asking a question.",
            "documents": [],
        }

    # Stages 6-7: attach source references and fill the message placeholders.
    context = format_documents(retrieved_documents)
    prompt = build_rag_prompt()
    formatted_prompt = prompt.invoke(
        {"context": context, "question": query}
    )

    # Stage 8: send the assembled messages to the generator. AIMessage.text
    # exposes text even when the provider represents it as content blocks.
    response = llm.invoke(formatted_prompt)
    return {
        "answer": response.text,
        "documents": retrieved_documents,
    }
