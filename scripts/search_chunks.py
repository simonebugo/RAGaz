from pathlib import Path
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from ragaz.load_documents import load_documents
from ragaz.chunk_documents import chunk_documents

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    pdf_dir = project_dir / "data" / "raw"
    documents = load_documents(pdf_dir)
    model_name = "intfloat/multilingual-e5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chunks = chunk_documents(documents, tokenizer, chunk_size=300, chunk_overlap=40)
    passages = []
    for chunk in chunks:
        passage = f"passage: {chunk.page_content}"
        passages.append(passage)
    transformer_model = SentenceTransformer(model_name, device = "mps")
    chunks_embeddings = transformer_model.encode(passages, normalize_embeddings=True, show_progress_bar=True)

    query_text = "query: Qual è il ruolo del meccanismo di attenzione nei Transformer?"
    query_embedding = transformer_model.encode(query_text, normalize_embeddings=True)
    similarities = transformer_model.similarity(query_embedding, chunks_embeddings)

    scores = similarities[0]

    for index,chunk in enumerate(chunks):
        if chunk.metadata['source'] == "transformer.pdf" and chunk.metadata['page'] == 2:
            print(f"index: {index}, score: {scores[index].item():.4f} ")
            print(f" text: {chunk.page_content}\n")

    top_results = scores.topk(5)
    top_scores = top_results.values.tolist()
    top_indices = top_results.indices.tolist()
    for score, chunk_index in zip(top_scores, top_indices):
        chunk = chunks[chunk_index]
        print(f"Score: {score:.4f}, Document: {chunk.metadata['source']}, Page: {chunk.metadata['page']}")
        print(f"Chunk Content: {chunk.page_content}\n")
        print("-" * 80)
   

