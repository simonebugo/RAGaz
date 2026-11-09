from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, tokenizer, chunk_size=300, chunk_overlap=40):
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return chunks
