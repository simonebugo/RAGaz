from sentence_transformers import SentenceTransformer
import numpy as np

if __name__ == "__main__":
    model_name = "intfloat/multilingual-e5-small"
    model = SentenceTransformer(model_name, device="cpu")
    text = "passage: retriever retrieves relevant passages that are useful for answering the query."
    model_embedding = model.encode(text,normalize_embeddings=True)
    print(f"results's type: {type(model_embedding)}")
    print(f"results's shape: {model_embedding.shape}")
    print(f"results's first 5 elements: {model_embedding[:5]}")
    print(f"normalized embedding's norm: {np.linalg.norm(model_embedding)}")