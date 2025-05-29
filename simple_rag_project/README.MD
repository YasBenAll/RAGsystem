    simple_rag_project/
    ├── .venv/
    ├── cat-facts.txt                  # Your dataset file
    ├── src/
    │   ├── __init__.py                # Makes 'src' a Python package
    │   ├── config.py                  # Configuration variables
    │   ├── data_loader.py             # Handles loading your dataset
    │   ├── embeddings.py              # Handles Ollama embedding calls
    │   ├── vector_store.py            # Your custom in-memory vector database
    │   ├── llm_inference.py           # Handles Ollama LLM calls
    │   └── rag_chain.py               # Combines retrieval and generation
    ├── main.py                        # The main entry point
    ├── requirements.txt               # Python dependencies
    └── .gitignore                     # Files/directories to ignore in Git