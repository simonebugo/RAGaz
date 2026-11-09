
def count_tokens(text,tokenizer) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    return len(token_ids)