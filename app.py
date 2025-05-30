# app.py

import streamlit as st
import os
import shutil
import time # For simulating streaming and adding small delays

# Import RAG components from your src directory
from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, LANGUAGE_MODEL, TOP_N_RETRIEVAL
from src.data_loader import load_documents, split_documents
from src.rag_chain import build_and_populate_vector_store # format_retrieved_knowledge is used internally now
from src.llm_inference import generate_llm_response

# --- Configuration and Constants ---
PROJECT_TITLE = "Interactive RAG System 📚🤖"
PROJECT_DESCRIPTION = """
This application demonstrates a Retrieval Augmented Generation (RAG) system.
Enter a query below to get answers based on a knowledge base of cat facts! 😺
"""
PROJECT_GITHUB_LINK = "https://github.com/yasbenall/ragsystem"

# --- Caching RAG System Components ---

@st.cache_resource(show_spinner="Initializing RAG System...")
def initialize_rag_system():
    # ... (rest of your initialize_rag_system function is fine) ...
    docs_directory = "docs"
    if not os.path.exists(docs_directory):
        st.error(f"Error: The 'docs' directory ('{docs_directory}') was not found. Please create it and add your source documents (e.g., cat-facts.txt).")
        st.stop()
        return None

    documents = load_documents(docs_directory)
    if not documents:
        st.error("No documents were loaded. Please check the 'docs' directory and its contents.")
        st.stop()
        return None

    from src.config import CHUNK_SIZE, CHUNK_OVERLAP
    chunks = split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not chunks:
        st.error("No chunks were generated from the documents.")
        st.stop()
        return None

    vector_store = build_and_populate_vector_store(chunks)
    if vector_store.count() == 0:
        st.warning("Vector store is empty after initialization. New data will be added on first run if available.")
    else:
        st.success(f"Vector store initialized with {vector_store.count()} entries.")
    return vector_store


def run_rag_pipeline_streamlit(query: str, vector_store):
    # ... (your run_rag_pipeline_streamlit function is fine) ...
    st.subheader("🧠 Retrieved Knowledge")
    retrieved_knowledge_results = vector_store.retrieve(query, top_n=TOP_N_RETRIEVAL)

    if not retrieved_knowledge_results:
        st.info("No relevant knowledge found in the vector store for your query.")
        knowledge_as_string = "No relevant context found in the knowledge base."
        yield {"type": "retrieval", "data": "No relevant knowledge found."}
    else:
        formatted_retrieved_knowledge_display = []
        llm_context_chunks = []
        with st.expander("📚 View Retrieved Chunks", expanded=False):
            for i, (chunk_text, metadata, similarity_score) in enumerate(retrieved_knowledge_results):
                display_text = f"**Chunk {i+1} (Source: {metadata.get('source', 'N/A')}, Page: {metadata.get('page', 'N/A')}, Score: {similarity_score:.2f})**\n\n> {chunk_text.strip()}\n\n---\n"
                st.markdown(display_text)
                formatted_retrieved_knowledge_display.append(display_text) # For potential direct yield if needed
                llm_context_chunks.append(f"Source: {metadata.get('source', 'N/A')}, Page: {metadata.get('page', 'N/A')}\nContent: {chunk_text.strip()}")
        
        knowledge_as_string = "\n\n".join(llm_context_chunks)
        yield {"type": "retrieval", "data": formatted_retrieved_knowledge_display}


    st.subheader("💬 Chatbot Response")
    instruction_prompt = f'''You are a helpful AI assistant.
Use ONLY the following pieces of context to answer the question. Your answer should be concise and directly address the question.
If the context does not contain the answer, state that the information is not available in the provided documents.
Do not make up any new information or answer from your general knowledge.

Context from documents:
{knowledge_as_string}

Question: {query}
Answer:'''
    response_placeholder = st.empty()
    full_response = ""
    stream = generate_llm_response(instruction_prompt, query)
    for chunk_response in stream:
        if chunk_response and chunk_response.get('message') and chunk_response['message'].get('content'):
            full_response += chunk_response['message']['content']
            response_placeholder.markdown(full_response + "▌")
            time.sleep(0.02)
    response_placeholder.markdown(full_response)
    yield {"type": "llm_response", "data": full_response}

# --- Main UI Layout ---
st.set_page_config(page_title=PROJECT_TITLE, layout="wide")

st.title(PROJECT_TITLE)
st.markdown(PROJECT_DESCRIPTION)
st.markdown(f"[View Project on GitHub]({PROJECT_GITHUB_LINK})")
st.markdown("---")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("🛠️ Controls")
    # Get the current vector_store instance (it's cached by Streamlit)
    # This line should be outside the button's if block if vector_store is needed before button press,
    # but for the button action, we get it fresh.
    # current_vector_store will be retrieved when initialize_rag_system() is called below.

    if st.button("Clear Vector Store Cache & Rebuild", type="primary"):
        # Step 1: Attempt to reset the currently active ChromaDB client
        # The `vector_store` variable below holds the instance from the last `initialize_rag_system` call
        active_vector_store = initialize_rag_system.__wrapped__() # Get a non-cached version for direct manipulation if needed
                                                                # Or better, operate on the global `vector_store` if sure it's current.
                                                                # For simplicity, let's assume `vector_store` from the main scope IS the one to reset.

        if 'vector_store_instance' not in st.session_state:
             st.session_state.vector_store_instance = initialize_rag_system()

        current_vs_instance = st.session_state.vector_store_instance

        if current_vs_instance and hasattr(current_vs_instance, '_reset_client'):
            st.info("Attempting to reset existing ChromaDB client before directory deletion...")
            try:
                current_vs_instance._reset_client() # Call the new method
                st.success("ChromaDB client reset successfully.")
                time.sleep(0.5) # Give a moment for file handles to be released
            except Exception as e:
                st.warning(f"Could not reset ChromaDB client, directory deletion might still fail: {e}")
        
        # Step 2: Delete the directory
        if os.path.exists(CHROMA_PERSIST_DIR):
            try:
                shutil.rmtree(CHROMA_PERSIST_DIR)
                st.success(f"Successfully deleted persistent vector store directory: '{CHROMA_PERSIST_DIR}'.")
            except PermissionError as e: # Catch PermissionError specifically
                st.error(f"Error clearing vector store directory (PermissionError): {e}")
                st.error(f"This usually means files in '{CHROMA_PERSIST_DIR}' are still in use.")
                st.error("A common cause is the ChromaDB client not fully releasing file locks. Although a reset was attempted, it might not have been sufficient on its own. Try restarting the Streamlit app completely if this persists.")
                # Do NOT clear st.cache_resource here, as the cleanup failed.
                st.rerun() # Rerun to show the error.
                # Use return or st.stop() if you want to halt further processing in the script for this click.
                # For now, it will fall through and try to clear cache etc., which is not ideal on error.
                # Let's add a return here to stop.
                st.stop() # Stop execution for this button click if rmtree fails
            except Exception as e:
                st.error(f"An unexpected error occurred while clearing vector store directory: {e}")
                st.rerun()
                st.stop() # Stop execution for this button click

        # Step 3: Clear Streamlit's resource cache for the RAG system
        # This should only happen if rmtree was successful or the directory didn't exist.
        st.cache_resource.clear()
        st.info("RAG system resource cache cleared. Will re-initialize on next action.")

        # Step 4: Ensure the directory is recreated for the new run
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        st.success(f"Recreated directory '{CHROMA_PERSIST_DIR}' for the new vector store.")
        
        # Step 5: Rerun the app to trigger re-initialization
        st.rerun()

    st.markdown("---")
    st.caption(f"Embedding Model: `{EMBEDDING_MODEL}`")
    st.caption(f"Language Model: `{LANGUAGE_MODEL}`")
    st.caption(f"Top N Retrieval: `{TOP_N_RETRIEVAL}`")

# Initialize RAG System (cached)
# Store the initialized vector_store in session_state to ensure we can access it in button callbacks
if 'vector_store_instance' not in st.session_state or not st.session_state.vector_store_instance:
    st.session_state.vector_store_instance = initialize_rag_system()

vector_store = st.session_state.vector_store_instance


if vector_store:
    st.header("❓ Ask a Question")
    query = st.text_input("Enter your question about cat facts:", placeholder="e.g., How long do cats sleep?")

    if query:
        for result in run_rag_pipeline_streamlit(query, vector_store):
            pass
else:
    st.error("RAG System could not be initialized. Please check logs and configurations.")

st.markdown("---")
st.caption("Built with Streamlit and Ollama.")