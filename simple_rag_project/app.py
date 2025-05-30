# app.py

import streamlit as st
import os
import shutil
import time # For simulating streaming

# Import RAG components from your src directory
from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, LANGUAGE_MODEL, TOP_N_RETRIEVAL
from src.data_loader import load_documents, split_documents
from src.rag_chain import build_and_populate_vector_store, format_retrieved_knowledge
from src.llm_inference import generate_llm_response # Assuming this can be used directly

# --- Configuration and Constants ---
PROJECT_TITLE = "Interactive RAG System 📚🤖"
PROJECT_DESCRIPTION = """
This application demonstrates a Retrieval Augmented Generation (RAG) system.
Enter a query below to get answers based on a knowledge base of cat facts! 😺
"""
PROJECT_GITHUB_LINK = "https://github.com/yasbenall/ragsystem" # Replace with your actual link if different

# --- Caching RAG System Components ---

@st.cache_resource(show_spinner="Initializing RAG System...")
def initialize_rag_system():
    """
    Loads documents, splits them, and builds/populates the vector store.
    This function is cached to run only once unless the cache is cleared.
    """
    # 1. Load Documents
    # Assuming 'docs' folder is in the same directory as app.py or main.py path is adjusted
    # For Streamlit, paths are relative to where streamlit run app.py is executed (project root)
    docs_directory = "docs" # Or "simple_rag_project/docs" if running from parent of simple_rag_project
    if not os.path.exists(docs_directory):
        st.error(f"Error: The 'docs' directory ('{docs_directory}') was not found. Please create it and add your source documents (e.g., cat-facts.txt).")
        st.stop()
        return None

    documents = load_documents(docs_directory)
    if not documents:
        st.error("No documents were loaded. Please check the 'docs' directory and its contents.")
        st.stop() # Stop execution if no documents
        return None

    # 2. Split Documents into Chunks
    # Ensure CHUNK_SIZE and CHUNK_OVERLAP are defined in your src.config
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP
    chunks = split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not chunks:
        st.error("No chunks were generated from the documents.")
        st.stop() # Stop execution if no chunks
        return None

    # 3. Build and Populate Vector Store
    vector_store = build_and_populate_vector_store(chunks)
    if vector_store.count() == 0:
        st.warning("Vector store is empty after initialization. New data will be added on first run if available.")
        # Don't stop here, as it might be the first run and needs to populate.
    else:
        st.success(f"Vector store initialized with {vector_store.count()} entries.")
    return vector_store

def run_rag_pipeline_streamlit(query: str, vector_store):
    """
    Modified RAG pipeline for Streamlit to yield results for dynamic display.
    Yields retrieved knowledge first, then streams LLM response.
    """
    st.subheader("🧠 Retrieved Knowledge")
    retrieved_knowledge_results = vector_store.retrieve(query, top_n=TOP_N_RETRIEVAL)

    if not retrieved_knowledge_results:
        st.info("No relevant knowledge found in the vector store for your query.")
        knowledge_as_string = "No relevant context found in the knowledge base."
        yield {"type": "retrieval", "data": "No relevant knowledge found."}
    else:
        formatted_retrieved_knowledge = []
        with st.expander("📚 View Retrieved Chunks", expanded=False):
            for i, (chunk_text, metadata, similarity_score) in enumerate(retrieved_knowledge_results):
                st.markdown(f"**Chunk {i+1} (Source: {metadata.get('source', 'N/A')}, Page: {metadata.get('page', 'N/A')}, Score: {similarity_score:.2f})**")
                st.caption(f"> {chunk_text.strip()}")
                st.markdown("---")
                formatted_retrieved_knowledge.append(f"Source: {metadata.get('source', 'N/A')}, Page: {metadata.get('page', 'N/A')}\nContent: {chunk_text.strip()}")
        # Yield the formatted retrieved knowledge for the LLM
        knowledge_as_string = "\n\n".join(formatted_retrieved_knowledge)
        yield {"type": "retrieval", "data": formatted_retrieved_knowledge}


    st.subheader("💬 Chatbot Response")
    # Construct the instruction prompt for the LLM
    instruction_prompt = f'''You are a helpful AI assistant.
Use ONLY the following pieces of context to answer the question. Your answer should be concise and directly address the question.
If the context does not contain the answer, state that the information is not available in the provided documents.
Do not make up any new information or answer from your general knowledge.

Context from documents:
{knowledge_as_string}

Question: {query}
Answer:'''

    # Use a placeholder for the LLM response
    response_placeholder = st.empty()
    full_response = ""

    stream = generate_llm_response(instruction_prompt, query)
    for chunk_response in stream:
        full_response += chunk_response['message']['content']
        response_placeholder.markdown(full_response + "▌") # Add a cursor effect
        time.sleep(0.02) # Small delay for streaming effect

    response_placeholder.markdown(full_response) # Final response without cursor
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
    if st.button("Clear Vector Store Cache & Rebuild", type="primary"):
        if os.path.exists(CHROMA_PERSIST_DIR):
            try:
                shutil.rmtree(CHROMA_PERSIST_DIR)
                st.success(f"Cleared persistent vector store at '{CHROMA_PERSIST_DIR}'.")
                # Important: Clear Streamlit's cache for the RAG system
                st.cache_resource.clear()
                st.info("RAG system cache cleared. Will re-initialize on next action.")
                # Trigger a rerun to force re-initialization
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing vector store: {e}")
        else:
            st.info("No persistent vector store found to clear.")
        # Ensure the directory is recreated if it was removed
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    st.markdown("---")
    st.caption(f"Embedding Model: `{EMBEDDING_MODEL}`")
    st.caption(f"Language Model: `{LANGUAGE_MODEL}`")
    st.caption(f"Top N Retrieval: `{TOP_N_RETRIEVAL}`")

# Initialize RAG System (cached)
vector_store = initialize_rag_system()

if vector_store: # Proceed only if RAG system initialized successfully
    # --- Query Input and Response Display ---
    st.header("❓ Ask a Question")
    query = st.text_input("Enter your question about cat facts:", placeholder="e.g., How long do cats sleep?")

    if query:
        # Clear previous results if any
        # (Handled by Streamlit's rerendering on new input naturally)

        # Run RAG pipeline and display results
        # The function will now use st elements directly to display output.
        for result in run_rag_pipeline_streamlit(query, vector_store):
            # The run_rag_pipeline_streamlit function now handles its own display
            pass
else:
    st.error("RAG System could not be initialized. Please check the logs and configurations.")

st.markdown("---")
st.caption("Built with Streamlit and Ollama.")