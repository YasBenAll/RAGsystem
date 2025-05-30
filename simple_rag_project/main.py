# main.py

import os
import shutil
import argparse
import subprocess # For launching Streamlit

from src.config import DATASET_PATH, CHROMA_PERSIST_DIR
from src.data_loader import load_documents, split_documents
from src.rag_chain import build_and_populate_vector_store # Removed run_rag_pipeline as app.py handles UI interaction
# If you still want a CLI RAG pipeline, you'd need a separate function for that in rag_chain.py or here.

# --- CLI RAG Interaction (Example, if you want to keep it) ---
def run_rag_pipeline_cli(query: str, vector_store):
    """
    Original RAG pipeline for CLI interaction.
    (This is a simplified version, adapt from your original run_rag_pipeline)
    """
    from src.rag_chain import format_retrieved_knowledge # Assuming this is still useful
    from src.llm_inference import generate_llm_response
    from src.config import TOP_N_RETRIEVAL

    print(f"\nProcessing query (CLI): '{query}'")

    retrieved_knowledge_results = vector_store.retrieve(query, top_n=TOP_N_RETRIEVAL)

    print('\nRetrieved knowledge (CLI):')
    if not retrieved_knowledge_results:
        print("No relevant knowledge found.")
        knowledge_as_string = "No relevant context found in the knowledge base."
    else:
        formatted_chunks = []
        for chunk_text, metadata, similarity_score in retrieved_knowledge_results:
            print(f' - (Source: {metadata.get("source", "N/A")}, Page: {metadata.get("page", "N/A")}, Score: {similarity_score:.2f}) {chunk_text.strip()}')
            formatted_chunks.append(f"Source: {metadata.get('source', 'N/A')}, Page: {metadata.get('page', 'N/A')}\nContent: {chunk_text.strip()}")
        knowledge_as_string = "\n\n".join(formatted_chunks)


    instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{knowledge_as_string}

Question: {query}
Answer:'''

    print('\nChatbot response (CLI):')
    stream = generate_llm_response(instruction_prompt, query)
    for chunk_response in stream:
        print(chunk_response['message']['content'], end='', flush=True)
    print("\n")

def main():
    parser = argparse.ArgumentParser(description="Run a RAG system.")
    parser.add_argument("--clear-db", action="store_true", help="Clear the persistent ChromaDB before starting.")
    parser.add_argument("--ui", choices=['cli', 'web'], default='cli', help="Choose the user interface: 'cli' or 'web'. Default is 'cli'.")
    args = parser.parse_args()

    if args.clear_db:
        if os.path.exists(CHROMA_PERSIST_DIR):
            print(f"Clearing persistent vector store at {CHROMA_PERSIST_DIR}...")
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print("Persistent store cleared.")
        else:
            print("No persistent store found to clear.")
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    if args.ui == 'web':
        print("Launching Streamlit Web UI...")
        # Ensure app.py is in the same directory or adjust path
        streamlit_app_path = os.path.join(os.path.dirname(__file__), "app.py")
        if not os.path.exists(streamlit_app_path):
            print(f"Error: Streamlit app file not found at {streamlit_app_path}")
            return
        try:
            subprocess.run(["streamlit", "run", streamlit_app_path], check=True)
        except FileNotFoundError:
            print("Error: Streamlit command not found. Please ensure Streamlit is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            print(f"Error running Streamlit app: {e}")
    else: # CLI mode
        print("Starting RAG system in CLI mode...")
        # --- 1. Load Documents ---
        docs_directory = "docs" # Assuming docs is in the same directory as main.py
        if not os.path.exists(docs_directory):
            print(f"Error: The 'docs' directory ('{docs_directory}') was not found in the project root. Please create it and add your source documents.")
            return

        documents = load_documents(docs_directory)
        if not documents:
            print("Exiting: No documents loaded from 'docs/' directory.")
            return

        # --- 2. Split Documents into Chunks ---
        from src.config import CHUNK_SIZE, CHUNK_OVERLAP
        chunks = split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        if not chunks:
            print("Exiting: No chunks generated from documents.")
            return

        # --- 3. Build and Populate Vector Store ---
        vector_store = build_and_populate_vector_store(chunks)
        if vector_store.count() == 0:
            print("Warning: Vector store is empty after initialization.")

        print(f"Vector store contains {vector_store.count()} entries.")
        print("\n--- RAG System Ready (CLI)! Ask me a question (type 'exit' to quit) ---")
        while True:
            input_query = input('Your question: ')
            if input_query.lower() == 'exit':
                break
            run_rag_pipeline_cli(input_query, vector_store) # Use the CLI pipeline
        print("Goodbye!")

if __name__ == "__main__":
    main()