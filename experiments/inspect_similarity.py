from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model_name = "intfloat/multilingual-e5-small"
    model = SentenceTransformer(model_name,device="cpu")
    query = "query: How can a rag system find relevant informations to answer a question?"
    passages = [
        "passage: A RAG system retrieves relevant passages from a knowledge base to answer a question.",
        "passage: Il chunk overlap ripete una parte del testo tra chunk consecutivi per preservare il contesto.",
        "passage: the lions are the kings of the jungle and they live in Africa."]
    query_embedding = model.encode(query,normalize_embeddings=True)
    passage_embeddings = model.encode(passages,normalize_embeddings=True)
    print(f"query embedding's shape: {query_embedding.shape} while passage embeddings's shape: {passage_embeddings.shape}")
    similarities = model.similarity(query_embedding,passage_embeddings)
    print(f"similarities: {similarities}, similarities's shape: {similarities.shape}")

    