"""Stage 2: split page Documents into smaller, retrievable text passages."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizerBase

from ragaz.config import CHUNK_OVERLAP, CHUNK_SIZE, E5_MAX_TOKENS
from ragaz.tokenization import count_tokens


def chunk_documents(
    documents: list[Document],
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split within each page and retain its source/page metadata.

    The recursive splitter prefers paragraph and line boundaries before using
    smaller separators. It does not understand meaning or resolve coreferences.
    Length is measured in tokens. Overlap is a target, not a promise that every
    pair of chunks will share exactly that many tokens.
    """
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    # The splitter measures text alone; E5 also receives a prefix and special
    # tokens. Validate that complete input before indexing any chunks.
    for chunk in chunks:
        token_count = count_tokens(f"passage: {chunk.page_content}", tokenizer)
        if token_count > E5_MAX_TOKENS:
            raise ValueError(
                f"Chunk exceeds E5's {E5_MAX_TOKENS}-token limit: "
                f"{chunk.metadata}, tokens={token_count}. Reduce chunk_size."
            )
    return chunks
