# RAGaz: RAG pipeline

RAGaz is a modular Retrieval-Augmented Generation (RAG) system for querying PDF
collections through natural-language questions and generating answers with
references to the retrieved sources.

Built with LangChain, multilingual E5 embeddings, and Chroma, the pipeline covers
document ingestion, token-aware chunking, persistent vector storage, semantic
retrieval, and answer generation. It integrates local Qwen inference through
LM Studio and an optional OpenAI API backend.

Each stage is implemented as a reusable component, making the retrieval and
generation workflow explicit, inspectable, and adaptable to future interfaces.

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

Reusable functions return data. Scripts handle terminal output. The Streamlit
app calls `answer_question` and displays `result["answer"]` and
`result["documents"]`.

```text
RAGaz/
  app.py                 Streamlit interface
  run.sh                 Launch Streamlit with the project environment
  ragaz/                 Reusable pipeline components
  scripts/
    inspect_chunks.py    Check extracted pages and complete chunk input sizes
    index_documents.py   Build or update the collection from local PDFs
    search_chroma.py     Ask a question against the saved collection
  data/
    raw/                 Local PDFs for command-line indexing
    uploads/             PDFs uploaded through Streamlit
    chroma/              Generated local database
  .env.example           Documented configuration template
  requirements.txt       Dependency versions used by this project
```

The earlier one-off inspection exercises, text-export command, and manual
in-memory search have been removed. Their essential lessons are explained in
the pipeline modules and the single chunk-inspection command.

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

Create `data/raw/` and place your text-based PDFs there. Then run the inspector.

```bash
python -m scripts.inspect_chunks
```

The original test corpus produced 105 pages and 279 chunks, with a largest
complete E5 input of 304 tokens. Your counts depend on the PDFs you supply.
E5 accepts at most 512 tokens per input. The default splitter budget is 300
tokens with a target overlap of 40 tokens.

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
Changing the embedding model also requires a fresh index. To add or remove
individual PDFs, use the Streamlit app.

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

The local Qwen workflow has been exercised end to end. The optional OpenAI
integration requires API credit and has not yet completed a successful
generation request in this project.

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

## 8. Use the Streamlit app

### Start the app locally

Complete the [environment setup](#3-prepare-the-environment). On macOS or Linux,
run this from the project root.

```bash
./run.sh
```

The launcher uses `.venv/bin/python` directly. On Windows, activate the virtual
environment and run `python -m streamlit run app.py` from the project root.
Open the local URL printed in the terminal. The first page load in a server
process initializes E5 and opens the persisted Chroma collection. On a fresh
setup, the app creates an empty collection. Press `Ctrl+C` in the terminal to
stop the app.

### Add a PDF, ask a question, and check the sources

1. In the **Documents** sidebar, choose a text-based PDF from your computer
   with **Upload a PDF**. The PDF loader extracts the existing text layer.
   Scanned pages need OCR before upload.
2. Click **Save and index**. The app archives the original under `data/uploads/`
   and saves the chunks and their vectors in `data/chroma/`. **Indexed PDFs**
   lists the filenames currently in the collection. Each indexed PDF needs a
   unique filename within that collection.
3. Start the LM Studio server and load the configured model. Ask about a detail
   in the uploaded PDF. For a PDF about RAG, try `What is chunk overlap?`. Click
   **Ask** to submit the question. In **Retrieved sources**, open the passage
   whose number matches a citation in the answer. For example, `[1]` refers to
   the first retrieved passage, and its label shows the PDF filename and page.
   Read the passage to check whether it supports the claim.
4. Restart Streamlit and check **Indexed PDFs** again. Repeat the question to see
   that the uploaded document is still searchable. To try removal, choose a test
   PDF in that list and click **Remove from search**. The app deletes its chunks
   from Chroma, and the archived original is available for a later upload.

### Follow the code

- [app.py](app.py) defines the widgets and calls the pipeline. Streamlit reruns
  the script after widget interactions. The upload button starts indexing, and
  the question form starts retrieval on submit. `st.cache_resource` keeps the E5
  model, Chroma connection, and LM Studio client in memory across reruns. Chroma
  persists the index on disk between server restarts.
- [ragaz/indexing.py](ragaz/indexing.py) prepares chunks for one PDF at a time.
  An ID such as `rag.pdf:2:5` identifies the chunk at index `5` in `rag.pdf`, on
  page `2`. Each PDF's chunk counter starts at zero. The `source` metadata groups
  chunks by filename, which lets the sidebar list or remove a whole PDF even
  when its chunks were created by the batch indexer.
- [ragaz/pipeline.py](ragaz/pipeline.py) retrieves the top passages, numbers them
  in the prompt, and calls LM Studio. The app uses that same order when rendering
  **Retrieved sources**, which shows every passage supplied to the model.
