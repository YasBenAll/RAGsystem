# main.py
import os
from src.config import DOCS_DIR, CHROMA_DB_DIR, OLLAMA_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
from src.data_processing import load_documents, split_documents
from src.embeddings import get_embedding_model, get_vector_store
from src.llm_integration import get_ollama_llm
from src.rag_pipeline import build_rag_chain

def main():
    # --- 1. Load Documents ---
    print("Loading documents...")
    documents = load_documents(DOCS_DIR)
    print(f"Loaded {len(documents)} documents.")

    # --- 2. Split Documents into Chunks ---
    print("Splitting documents into chunks...")
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Split into {len(chunks)} chunks.")

    # --- 3. Create Embeddings and Build Vector Store ---
    print("Creating embeddings and building ChromaDB vector store...")
    embeddings = get_embedding_model(EMBEDDING_MODEL_NAME)
    # For persistence, uncomment the line below and ensure data/embeddings_cache exists
    # vectorstore = get_vector_store(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
    vectorstore = get_vector_store(chunks, embeddings) # Using in-memory for simple start
    print("Vector store created.")

    # --- 4. Initialize LLM ---
    print(f"Initializing Ollama LLM with model: {OLLAMA_MODEL}...")
    llm = get_ollama_llm(OLLAMA_MODEL)
    print("LLM initialized.")

    # --- 5. Set up the Retrieval-Augmented Generation (RAG) Chain ---
    print("Setting up RAG chain...")
    qa_chain = build_rag_chain(llm, vectorstore.as_retriever())
    print("RAG chain ready!")

    # --- 6. Ask Questions ---
    print("\n--- Start Asking Questions (type 'exit' to quit) ---")
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break

        print("Thinking...")
        result = qa_chain.invoke({"query": query})

        print(f"\nAnswer: {result['result']}")
        if result.get('source_documents'):
            print("\n--- Sources Used ---")
            for i, doc in enumerate(result['source_documents']):
                print(f"Source {i+1} (Page Content): {doc.page_content[:200]}...")
                print(f"Source {i+1} (Metadata): {doc.metadata}")
                print("-" * 20)

    print("\nGoodbye!")

if __name__ == "__main__":
    main()