# app.py

import streamlit as st
import os
import shutil
import time # For simulating streaming and adding small delays

# --- Import RAG components ---
from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, LANGUAGE_MODEL, TOP_N_RETRIEVAL, USE_LANGCHAIN
# Original implementation imports
from src.data_loader import load_documents, split_documents
from src.rag_chain import build_and_populate_vector_store
from src.llm_inference import generate_llm_response
# Langchain implementation imports
from src.langchain_rag import create_langchain_vector_store, create_langchain_rag_chain

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
    if USE_LANGCHAIN:
        st.info("Using LangChain implementation.")
        if not os.path.exists(os.path.join(CHROMA_PERSIST_DIR)) or not os.listdir(CHROMA_PERSIST_DIR):
             st.info("Creating new vector store with LangChain...")
             vector_store = create_langchain_vector_store(docs_directory="docs")
        else:
             st.info("Loading existing vector store with LangChain.")
             # For LangChain, we just need to know the directory exists.
             # The retriever will be initialized on-the-fly when the chain is created.
             vector_store = "LangChain store initialized" # Placeholder to indicate success

        if vector_store is None:
            st.error("LangChain vector store initialization failed.")
            st.stop()

        rag_chain = create_langchain_rag_chain()
        return rag_chain

    else:
        st.info("Using original implementation.")
        # ... (rest of your original initialize_rag_system function) ...
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


def run_rag_pipeline_streamlit(query: str, rag_system_obj):
    if USE_LANGCHAIN:
        # rag_system_obj is the LangChain RAG chain
        rag_chain = rag_system_obj

        st.subheader("💬 Chatbot Response")
        with st.spinner("Thinking..."):
            response = rag_chain({"query": query})
            st.markdown(response["result"])

        with st.expander("📚 View Retrieved Source Documents"):
            for doc in response["source_documents"]:
                st.markdown(f"**Source: {doc.metadata.get('source', 'N/A')}**")
                st.markdown(f"> {doc.page_content.strip()}")
                st.markdown("---")
    else:
        # rag_system_obj is the original vector_store
        vector_store = rag_system_obj
        # ... (your original run_rag_pipeline_streamlit function) ...
        st.subheader("🧠 Retrieved Knowledge")
        retrieved_knowledge_results = vector_store.retrieve(query, top_n=TOP_N_RETRIEVAL)

        if not retrieved_knowledge_results:
            st.info("No relevant knowledge found in the vector store for your query.")
            knowledge_as_string = "No relevant context found in the knowledge base."
        else:
            formatted_retrieved_knowledge_display = []
            llm_context_chunks = []
            with st.expander("📚 View Retrieved Chunks", expanded=False):
                for i, (chunk_text, metadata, similarity_score) in enumerate(retrieved_knowledge_results):
                    display_text = f"**Chunk {i+1} (Source: {metadata.get('source', 'N/A')}, Page: {metadata.get('page', 'N/A')}, Score: {similarity_score:.2f})**\n\n> {chunk_text.strip()}\n\n---\n"
                    st.markdown(display_text)
                    llm_context_chunks.append(f"Source: {metadata.get('source', 'N/A')}, Page: {metadata.get('page', 'N/A')}\nContent: {chunk_text.strip()}")

            knowledge_as_string = "\n\n".join(llm_context_chunks)

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

# --- Main UI Layout ---
st.set_page_config(page_title=PROJECT_TITLE, layout="wide")

st.title(PROJECT_TITLE)
st.markdown(PROJECT_DESCRIPTION)
st.markdown(f"[View Project on GitHub]({PROJECT_GITHUB_LINK})")
st.markdown("---")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("🛠️ Controls")

    # New section for implementation toggle
    st.markdown("---")
    st.info(f"Current Mode: **{'LangChain' if USE_LANGCHAIN else 'Original'}**")
    st.caption("To change the implementation, edit `USE_LANGCHAIN` in `src/config.py` and restart the app.")
    st.markdown("---")

    if st.button("Clear Vector Store Cache & Rebuild", type="primary"):
        # This logic works for both implementations as it just clears the directory
        if os.path.exists(CHROMA_PERSIST_DIR):
            try:
                # No need to reset client manually, just clear cache and delete dir
                st.cache_resource.clear()
                shutil.rmtree(CHROMA_PERSIST_DIR)
                st.success(f"Successfully deleted persistent vector store directory: '{CHROMA_PERSIST_DIR}'.")
            except Exception as e:
                st.error(f"Error clearing vector store directory: {e}")
        else:
            st.info("No vector store cache directory to clear.")

        st.success("Cache cleared. Rerunning to re-initialize.")
        st.rerun()

    st.markdown("---")
    st.caption(f"Embedding Model: `{EMBEDDING_MODEL}`")
    st.caption(f"Language Model: `{LANGUAGE_MODEL}`")
    st.caption(f"Top N Retrieval: `{TOP_N_RETRIEVAL}`")
    st.caption(f"Using LangChain: `{USE_LANGCHAIN}`")


# Initialize RAG System (cached)
rag_system_obj = initialize_rag_system()

if rag_system_obj:
    st.header("❓ Ask a Question")
    query = st.text_input("Enter your question about cat facts:", placeholder="e.g., How long do cats sleep?")

    if query:
        run_rag_pipeline_streamlit(query, rag_system_obj)
else:
    st.error("RAG System could not be initialized. Please check logs and configurations.")

st.markdown("---")
st.caption("Built with Streamlit and Ollama.")