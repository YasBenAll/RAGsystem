# main.py

import os
import shutil # For deleting the ChromaDB directory
import argparse # For command-line arguments to clear DB

from src.config import DATASET_PATH, CHROMA_PERSIST_DIR, EMBEDDING_MODEL # <-- Import EMBEDDING_MODEL for PersistentVectorStore
from src.data_loader import load_documents, split_documents # Changed to load_documents for flexibility
from src.rag_chain import build_and_populate_vector_store, run_rag_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run a RAG system with persistent vector store.")
    parser.add_argument("--clear-db", action="store_true", help="Clear the persistent ChromaDB before starting.")
    args = parser.parse_args()

    if args.clear_db:
        if os.path.exists(CHROMA_PERSIST_DIR):
            print(f"Clearing persistent vector store at {CHROMA_PERSIST_DIR}...")
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print("Persistent store cleared.")
        else:
            print("No persistent store found to clear.")
        # Ensure the directory is recreated for the new run
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)


    # --- 1. Load Documents ---
    # Now uses load_documents to handle potentially multiple file types from the 'docs' folder
    documents = load_documents("docs") # Assumes your cat-facts.txt is now in 'docs' folder
    if not documents:
        print("Exiting: No documents loaded from 'docs/' directory or an error occurred.")
        return

    # --- 2. Split Documents into Chunks ---
    # Use config for chunk size and overlap
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP # You need to add these to config.py if not already
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50) # Use fixed for now, add to config later
    if not chunks:
        print("Exiting: No chunks generated from documents.")
        return

    # --- 3. Build and Populate Vector Store ---
    # Pass the chunks to the function that initializes and populates the store
    vector_store = build_and_populate_vector_store(chunks)
    if vector_store.count() == 0: # Check the count of items in the collection
        print("Exiting: Vector store could not be populated or is empty.")
        return

    print(f"Vector store contains {vector_store.count()} entries.")


    # --- 4. Interactive RAG Query Loop ---
    print("\n--- RAG System Ready! Ask me a question (type 'exit' to quit) ---")
    while True:
        input_query = input('Your question: ')
        if input_query.lower() == 'exit':
            break

        run_rag_pipeline(input_query, vector_store)

    print("Goodbye!")

if __name__ == "__main__":
    # Ensure you have 'ollama' installed and the models specified in config.py
    # (e.g., 'bge-base-en' and 'llama3') are pulled locally using `ollama pull <model_name>`
    # before running this script.
    main()