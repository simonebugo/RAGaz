# Streamlit reruns this script from top to bottom after widget interactions.
# Button and form handlers control when indexing, removal, and retrieval happen.
import streamlit as st
from dotenv import load_dotenv
from pypdf.errors import PdfReadError

from ragaz.config import PROJECT_DIR, TOP_K
from ragaz.embeddings import E5Embeddings
from ragaz.generation import build_local_llm
from ragaz.pipeline import answer_question
from ragaz.vector_store import open_vector_store
from ragaz.indexing import list_indexed_pdfs, remove_pdf, save_and_index_pdf



# Reuse the E5 model, Chroma connection, and LM Studio client across reruns.
@st.cache_resource
def get_rag_resources():
    load_dotenv(PROJECT_DIR / ".env")

    embeddings = E5Embeddings()
    store = open_vector_store(embeddings, create_if_missing=True)

    # Split uploaded PDFs with the tokenizer that belongs to the embedding model.
    return store, embeddings.model.tokenizer, build_local_llm()


st.title("RAGaz")

with st.sidebar:
    st.header("Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    # The uploader makes the file available in memory. The button starts the
    # operation that writes the PDF to disk and adds its chunks to Chroma.
    if st.button("Save and index", disabled=uploaded_file is None):
        try:
            with st.spinner("Saving and indexing..."):
                store, tokenizer, _ = get_rag_resources()
                chunk_count = save_and_index_pdf(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    tokenizer,
                    store,
                )
        except (OSError, PdfReadError, ValueError) as error:
            st.error(str(error))
        else:
            st.success(f"Indexed {uploaded_file.name} ({chunk_count} chunks).")

    # Read the current collection when the sidebar renders, including the initial
    # page load and the rerun after a document is added or removed.
    store, _, _ = get_rag_resources()
    pdf_names = list_indexed_pdfs(store)

    # Carry the confirmation across the removal rerun and display it once.
    removal_message = st.session_state.pop("removal_message", None)
    if removal_message:
        st.success(removal_message)

    selected_pdf = st.selectbox("Indexed PDFs", pdf_names, index=None)

    if st.button("Remove from search", disabled=selected_pdf is None):
        removed_count = remove_pdf(selected_pdf, store)
        st.session_state["removal_message"] = (
            f"Removed {selected_pdf} ({removed_count} chunks)."
        )
        # Rebuild the selector from the updated collection.
        st.rerun()

# The form sends the question on submit. The submitted flag is true for that run.
with st.form("question_form"):
    question = st.text_input("Question")
    submitted = st.form_submit_button("Ask")

if submitted:
    question = question.strip()

    if question:
        with st.spinner("searching in pdf and prepararing answer ..."):
            store, _, llm = get_rag_resources()
            retriever = store.as_retriever( search_type="similarity",search_kwargs={"k": TOP_K})
            # The pipeline retrieves the top passages and sends their numbered
            # text to LM Studio together with the question.
            result = answer_question(query=question, retriever=retriever, llm=llm)

        st.markdown(result["answer"])
        documents = result["documents"]
        if documents:
            st.subheader("Retrieved sources")

            # Match the numbering in the prompt. Showing every retrieved passage
            # lets the reader inspect the evidence the model received.
            for number, document in enumerate(documents, start=1):
                source = document.metadata["source"]
                page = document.metadata["page"]

                with st.expander(f"[{number}] {source} · Page {page}"):
                    st.text(document.page_content)
    else:
        st.warning("Insert a question.")
