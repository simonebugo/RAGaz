# RAGaz: an educational RAG pipeline

A small Python project that follows a document from PDF extraction to a
source-grounded answer. It uses LangChain, multilingual E5 embeddings, Chroma,
and Qwen served locally by LM Studio. An optional OpenAI generator is also kept.

The local path has been exercised end to end.

## 1. Understand the two workflows

**Indexing prepares the knowledge base:**

```text
PDF files -> page Documents -> overlapping chunks -> E5 vectors -> Chroma on disk
```

**Question answering reuses that knowledge base:**

```text
Question -> E5 query vector -> retrieve chunks from Chroma
         -> context with numbered references -> prompt -> Qwen -> answer + sources
```

Indexing stores chunk text, source filename, page number, ID, and vector.
Question answering computes a new vector for the question, not for all PDFs.
The generator receives retrieved text through the prompt; it does not read the
database directly. This is retrieval-augmented generation, not fine-tuning.

## 2. Read the code in pipeline order

| Stage | File | What to learn |
| --- | --- | --- |
| Shared settings | `ragaz/config.py` | Keep paths, model, and chunk settings consistent. |
| 1. Extract | `ragaz/load_documents.py` | One `Document` per page, with traceable metadata. |
| 2. Split | `ragaz/chunk_documents.py` | Token-aware recursive splitting and overlap. |
| Input measurement | `ragaz/tokenization.py` | Count prefixes and special tokens without truncation. |
| 3. Embed | `ragaz/embeddings.py` | Adapt E5 to `embed_documents` and `embed_query`. |
| 4. Store | `ragaz/vector_store.py` | Persist and reopen the same Chroma collection. |
| 5. Retrieve | `ragaz/pipeline.py` | Retrieve evidence once per question. |
| 6. Format | `ragaz/context.py` | Preserve source/page references in the context. |
| 7. Prompt | `ragaz/prompts.py` | Separate application instructions from evidence. |
| 8. Generate | `ragaz/generation.py` | Configure the local or optional cloud generator. |
| Orchestrate | `ragaz/pipeline.py` | Return an answer and its source Documents. |

Reusable functions return data. Scripts handle terminal output. A future
notebook or Streamlit UI can call `answer_question` and display
`result["answer"]` and `result["documents"]` without copying the RAG logic.

```text
RAGaz/
  ragaz/                 Reusable pipeline components
  scripts/
    inspect_chunks.py    Check extracted pages and complete chunk input sizes
    index_documents.py   Build or update the example collection
    search_chroma.py     Ask a question against the saved collection
  data/
    raw/                 Original PDF files
    chroma/              Generated local database
  .env.example           Documented configuration template
  requirements.txt       Dependency versions used by this project
```

The earlier one-off inspection exercises, text-export command, and manual
in-memory search have been removed. Their essential lessons are explained in
the pipeline modules and the single chunk-inspection command. Existing PDF,
database, and text-export data have been preserved.

## 3. Prepare the environment

Use Python 3.12. From the project root, create and activate a virtual environment
if one does not already exist:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell instead.
The pins document the working development environment; installation on other
platforms also depends on available wheels for those versions.

For a new setup, copy `.env.example` to `.env` and adjust the settings. Preserve
an existing `.env` rather than replacing it. The entry points load this file.

The embedding device is selected automatically unless `EMBEDDING_DEVICE` is set.
Use `mps` for a supported Apple GPU, `cuda` for a supported NVIDIA setup, or
`cpu` for CPU inference. The embedding model runs through Sentence Transformers,
independently of the generator loaded in LM Studio.

For local generation, start the LM Studio server and load the configured Qwen
model. Defaults are `http://127.0.0.1:1234/v1` and `qwen/qwen3.8-27b`.
Set `LM_STUDIO_BASE_URL` and `LM_STUDIO_MODEL` to match another installation.
The generator's hardware requirements depend on the model you choose.

## 4. Inspect the PDFs and chunks

Place text-based `.pdf` files in `data/raw/`, then run:

```bash
python -m scripts.inspect_chunks
```

For the original three PDFs, the baseline is 105 pages, 279 chunks, and a largest
complete E5 input of 304 tokens. E5 accepts at most 512 tokens per input. The
default splitter budget is 300 tokens with a target overlap of 40 tokens.

Complete inputs include `passage: ` and tokenizer special tokens. Validation
rejects overlong chunks before indexing, avoiding silent truncation. E5 uses
`query: ` for questions. The adapter adds both prefixes even for non-English
text; callers provide unprefixed strings.

A tokenizer warning about an intermediate block exceeding 512 tokens can occur
while the splitter measures text before splitting it. Complete final chunks are
checked separately. Nonempty extraction alone does not guarantee readable text;
scanned pages need OCR, and diagrams or formulas may extract poorly.

## 5. Index the knowledge base

```bash
python -m scripts.index_documents
```

This creates the collection when necessary and persists it in `data/chroma/`.
It keeps the original IDs (`source:page:global_chunk_index`) for compatibility
with the existing database. Re-running an unchanged corpus with the same
settings updates those IDs without multiplying entries, but recomputes their
embeddings.

**Current limitation:** this is not yet incremental document management. Adding
or removing PDFs, editing text, or changing chunk settings can change IDs or
leave obsolete chunks behind. A fresh collection or an explicit rebuild is
needed for such changes; no automatic deletion or migration is performed here.
Changing the embedding model also requires a fresh index. Stable document IDs,
updates, deletion, and selecting a subset of PDFs are the next ingestion steps.

## 6. Ask a question

```bash
python -m scripts.search_chroma "Why can increasing chunk overlap increase redundancy?"
python -m scripts.search_chroma "Perché aumentare il chunk overlap può aumentare la ridondanza?"
python -m scripts.search_chroma "What is the role of attention in Transformers?" --k 5
```

With no argument, the script runs the example overlap question. The answer
follows the question's language. Code comments, source labels, and command-line
messages are in English. Run entry points as modules from the project root;
directly running a file inside `scripts/` can break imports.

Qwen is the default. Optionally configure `OPENAI_API_KEY` in `.env` and select
cloud generation explicitly:

```bash
python -m scripts.search_chroma "Why does overlap introduce redundancy?" --provider openai
```

That option sends the question and retrieved context to the OpenAI API and
requires API credit. It uses the same local E5 vectors and Chroma collection.

## 7. Evaluate the evidence, not just the output

- Check whether retrieved passages contain an answer. A related topic is not
  necessarily useful evidence.
- Check references against `result["documents"]`. Reference numbers are local
  to one answer and are not permanent chunk IDs.
- Test a question whose answer is absent from the PDFs and inspect whether the
  model acknowledges insufficient evidence.
- Compare overlapping passages for duplicated content and incomplete sentences.
- Keep the attention/Transformer question as a known weak retrieval case: the
  baseline retrieved related passages ahead of more direct explanations.

The default retriever returns top-k passages even for unrelated questions; there
is no calibrated relevance threshold. Prompt instructions and citations do not
guarantee correctness, and source notes may contain mistakes. Improving
retrieval and assessing groundedness remain separate tasks.

If you inspect `similarity_search_with_score` directly, this collection returns
cosine **distance**: lower is closer, and distance is `1 - cosine_similarity`.
These scores are not probabilities.

## Next interface step

Build the first Streamlit screen around the existing collection and
`answer_question`. Keep model resources alive across questions. Add PDF upload
and persistence after document identity and update handling are implemented.
No Streamlit application is included in this cleanup.
