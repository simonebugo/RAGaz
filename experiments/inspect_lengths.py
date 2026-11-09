from transformers import AutoTokenizer
from pathlib import Path
from ragaz.load_documents import load_documents
from ragaz.tokenization import count_tokens

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    pdf_dir = project_dir / "data" / "raw"
    documents = load_documents(pdf_dir)
    model_name = "intfloat/multilingual-e5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_counts = []
    pages_over_limit = 0
    for doc in documents:
        content = f"passage: {doc.page_content}"
        token_num = count_tokens(content, tokenizer)
        token_counts.append(token_num)
        if token_num > 512:
            pages_over_limit += 1
        print(f"Document: {doc.metadata['source']}, Page: {doc.metadata['page']}, Token count: {token_num}")

    
    print(f"total number of pages: {len(token_counts)}")
    print(f"Number of pages exceeding 512 tokens: {pages_over_limit}")
    print(f"max number of tokens in a page: {max(token_counts)}")

