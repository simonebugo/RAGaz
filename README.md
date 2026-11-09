# RAGaz
First approach with a RAG system integrated with LLM (Qwen and GPT)

## Organizzazione

- `ragaz/`: funzioni riutilizzabili per caricamento, chunking e conteggio dei token.
- `experiments/`: esperimenti per ispezionare PDF, token, chunk ed embedding.
- `scripts/`: comandi operativi per esportazione e ricerca.
- `data/raw/`: PDF originali.
- `data/extracted/`: testo esportato dagli script, escluso da Git.

## Esecuzione

Aprire il terminale nella cartella principale RAGaz e attivare l'ambiente:

```bash
source .venv/bin/activate
```

Eseguire i programmi come moduli, senza estensione `.py`:

```bash
python -m ragaz.load_documents
python -m experiments.inspect_pdf
python -m experiments.inspect_tokens
python -m experiments.inspect_lengths
python -m experiments.inspect_chunks
python -m experiments.inspect_embeddings
python -m experiments.inspect_similarity
python -m scripts.export_pages
```

Il comando `scripts.search_chunks` verrà implementato nel prossimo esercizio.
