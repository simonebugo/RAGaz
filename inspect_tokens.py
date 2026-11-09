from transformers import AutoTokenizer


def count_tokens(text,tokenizer) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    return len(token_ids)

if __name__ == "__main__":
    model_name = "intfloat/multilingual-e5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "passage: retriever retrieves relevant passages that are useful for answering the query."
    token_count = count_tokens(text, tokenizer)
    print(f"Token count for the text: {token_count}")