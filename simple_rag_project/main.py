# main.py

import os
from src.config import DATASET_PATH
from src.data_loader import load_text_dataset
from src.rag_chain import build_and_populate_vector_store, run_rag_pipeline

def main():
    # --- 1. Load Dataset ---
    dataset_chunks = load_text_dataset(DATASET_PATH)
    if not dataset_chunks:
        print("Exiting: No dataset loaded or an error occurred.")
        return

    # --- 2. Build and Populate Vector Store ---
    # This step will generate embeddings for each chunk and store them.
    vector_store = build_and_populate_vector_store(dataset_chunks)
    if not vector_store.vector_db:
        print("Exiting: Vector store could not be populated.")
        return

    # --- 3. Interactive RAG Query Loop ---
    print("\n--- RAG System Ready! Ask me a question about cats (type 'exit' to quit) ---")
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