"""Measure complete model inputs without silently truncating their text."""

from transformers import PreTrainedTokenizerBase


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Count token IDs, including special tokens added by the tokenizer.

    Callers include any prefix, such as 'passage: ', in text. Counting a long
    input is safe; sending it to the embedding model is a separate operation.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    return len(token_ids)
