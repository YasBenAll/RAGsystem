# src/config.py

# Path to your dataset file
DATASET_PATH = 'cat-facts.txt'

# Directory for ChromaDB persistence
CHROMA_PERSIST_DIR = "chroma_db" # <--- ADD THIS LINE

# RAG parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest' # <--- IMPORTANT: Add ':latest'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest' # <--- IMPORTANT: Add ':latest'


# Retrieval parameters
TOP_N_RETRIEVAL = 3