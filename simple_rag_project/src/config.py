# src/config.py

# Path to your dataset file
DATASET_PATH = 'cat-facts.txt'

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf' # Example: `ollama pull bge-base-en`
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'       # Example: `ollama pull llama3`

# Retrieval parameters
TOP_N_RETRIEVAL = 3