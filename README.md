RAG system built specifically for chatbots

interesting papers:
https://arxiv.org/abs/2401.06800

Interesting repo's:
https://sudarshanpoudel.github.io/agenticrag/

Mastering RAG:
https://galileo.ai/blog/mastering-rag-how-to-architect-an-enterprise-rag-system

https://huggingface.co/blog/ngxson/make-your-own-rag

Components

Embedding model
hf.co/CompendiumLabs/bge-base-en-v1.5-gguf

Vector database
dataset: https://huggingface.co/ngxson/demo_simple_rag_py/blob/main/cat-facts.txt


Chatbot 
hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF

User interface


├── .venv/                         # Python virtual environment (hidden by default)
├── docs/                          # Your source documents for the RAG system
│   ├── ai_info.txt
│   └── rag_info.txt
├── data/                          # (Optional) For processed data, if separate from docs
│   └── embeddings_cache/          # (Optional) For persistent ChromaDB or other vector store files
├── src/                           # Main source code for your RAG system
│   ├── __init__.py                # Makes 'src' a Python package
│   ├── config.py                  # For all configurations (paths, model names, etc.)
│   ├── data_processing.py         # Functions for loading, splitting documents
│   ├── embeddings.py              # Functions for creating embeddings and vector store
│   ├── llm_integration.py         # Functions for interacting with the LLM
│   └── rag_pipeline.py            # The main logic that ties everything together (the RAG chain)
├── main.py                        # The entry point for running your application
├── requirements.txt               # List of Python dependencies
├── README.md                      # Project description, setup instructions, how to run
└── .gitignore                     # Files/directories to ignore in Git (e.g., .venv/, __pycache__/, data/)