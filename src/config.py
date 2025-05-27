import os

# Paths
DOCS_DIR = "docs"
CHROMA_DB_DIR = os.path.join("data", "embeddings_cache") # For persistent ChromaDB

# RAG parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Model names
OLLAMA_MODEL = "llama3"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"