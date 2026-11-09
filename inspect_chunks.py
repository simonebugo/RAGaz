from transformers import AutoTokenizer
from pathlib import Path
from load_documents import load_documents
from inspect_tokens import count_tokens
from chunk_documents import chunk_documents

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    pdf_dir = project_dir / "data" / "raw"
    documents = load_documents(pdf_dir)
    model_name = "intfloat/multilingual-e5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chunks = chunk_documents(documents, tokenizer, chunk_size=300, chunk_overlap=40)
    token_counts = []
    chunks_over_limit = 0
    for chunk in chunks:
        token_count = count_tokens('passage: ' +chunk.page_content, tokenizer)
        token_counts.append(token_count)
        if token_count > 512:
            chunks_over_limit += 1
            print(f"chunk metadata: {chunk.metadata}")
            print(f"token count: {token_count}")

    print(f"number of initial pages: {len(documents)}")
    print(f"total number of chunks: {len(chunks)}")
    print(f"max number of tokens in a chunk: {max(token_counts)}")
    print(f"number of chunks exceeding 512 tokens: {chunks_over_limit}")
