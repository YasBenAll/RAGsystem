# src/rag_chain.py

from typing import List, Tuple, Dict, Any # <-- Add Dict, Any
# Remove InMemoryVectorStore if you had it
from src.vector_store import PersistentVectorStore # <-- Import the new class
from src.embeddings import generate_embedding # Still needed for query embedding
from src.llm_inference import generate_llm_response
from src.config import TOP_N_RETRIEVAL, CHROMA_PERSIST_DIR, EMBEDDING_MODEL # <-- Import CHROMA_PERSIST_DIR and EMBEDDING_MODEL


def build_and_populate_vector_store(dataset_chunks: List[str]) -> PersistentVectorStore:
    """
    Initializes and populates the persistent vector store with dataset chunks.
    Only adds chunks if they are not already in the store.

    Args:
        dataset_chunks (List[Document]): A list of LangChain Document objects (chunks).

    Returns:
        PersistentVectorStore: An instance of the populated vector store.
    """
    print("Initializing persistent vector store...")
    vector_store = PersistentVectorStore(persist_directory=CHROMA_PERSIST_DIR)

    # Check if the store needs to be populated
    # This is a simple check; for production, you might need a more robust
    # versioning or hashing system to detect changes in source documents.
    if vector_store.count() < len(dataset_chunks):
        print(f"Populating vector store with {len(dataset_chunks)} new chunks (current count: {vector_store.count()})...")
        # Prepare chunks for ChromaDB: list of (content, metadata)
        chunks_for_chroma = [(chunk.page_content, chunk.metadata) for chunk in dataset_chunks]
        vector_store.add_chunks(chunks_for_chroma)
        print(f"Vector store populated. Total items: {vector_store.count()}")
    else:
        print(f"Vector store already contains {vector_store.count()} chunks. Skipping population.")

    return vector_store

def format_retrieved_knowledge(retrieved_chunks: List[Tuple[str, float]]) -> str:
    """
    Formats the retrieved knowledge into a string suitable for the LLM prompt.

    Args:
        retrieved_chunks (List[Tuple[str, float]]): List of (chunk_text, similarity_score) tuples.

    Returns:
        str: A single string containing the formatted knowledge.
    """
    lines_to_join = [f' - {chunk}' for chunk, similarity in retrieved_chunks]
    knowledge_as_string = '\n'.join(lines_to_join)
    return knowledge_as_string

def run_rag_pipeline(query: str, vector_store: PersistentVectorStore):
    """
    Executes the full RAG pipeline: retrieval, prompt construction, and LLM generation.

    Args:
        query (str): The user's question.
        vector_store (PersistentVectorStore): The populated vector store instance. # <-- Change type hint
    """
    print(f"\nProcessing query: '{query}'")

    # 1. Retrieve relevant knowledge
    query_embedding = generate_embedding(query)
    if not query_embedding:
        print("Failed to generate query embedding. Cannot proceed with retrieval.")
        return

    retrieved_knowledge = vector_store.retrieve(query_embedding, top_n=TOP_N_RETRIEVAL)

    print('\nRetrieved knowledge:')
    if not retrieved_knowledge:
        print("No relevant knowledge found.")
        knowledge_as_string = "No relevant context found in the knowledge base."
    else:
        for chunk, similarity in retrieved_knowledge:
            print(f' - (similarity: {similarity:.2f}) {chunk.strip()}') # .strip() to clean up newlines
        knowledge_as_string = format_retrieved_knowledge(retrieved_knowledge)

    # 2. Construct the instruction prompt for the LLM
    instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{knowledge_as_string}
'''
    # print(f"\n--- LLM System Prompt ---\n{instruction_prompt}\n------------------------") # For debugging

    # 3. Generate response using the LLM
    print('\nChatbot response:')
    stream = generate_llm_response(instruction_prompt, query)

    # Print the response from the chatbot in real-time
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print("\n") # Add a newline after the streamed response