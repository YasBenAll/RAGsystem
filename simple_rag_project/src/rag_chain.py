# src/rag_chain.py

from typing import List, Tuple
from src.embeddings import generate_embedding
from src.vector_store import InMemoryVectorStore
from src.llm_inference import generate_llm_response
from src.config import TOP_N_RETRIEVAL

def build_and_populate_vector_store(dataset_chunks: List[str]) -> InMemoryVectorStore:
    """
    Initializes and populates the in-memory vector store with embeddings of dataset chunks.

    Args:
        dataset_chunks (List[str]): A list of text chunks from the dataset.

    Returns:
        InMemoryVectorStore: An instance of the populated vector store.
    """
    print("Building and populating vector store...")
    vector_store = InMemoryVectorStore()
    for i, chunk in enumerate(dataset_chunks):
        embedding = generate_embedding(chunk)
        if embedding: # Only add if embedding was successfully generated
            vector_store.add_chunk(chunk, embedding)
        print(f'  - Processed chunk {i+1}/{len(dataset_chunks)}')
    print(f"Vector store populated with {len(vector_store.vector_db)} chunks.")
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

def run_rag_pipeline(query: str, vector_store: InMemoryVectorStore):
    """
    Executes the full RAG pipeline: retrieval, prompt construction, and LLM generation.

    Args:
        query (str): The user's question.
        vector_store (InMemoryVectorStore): The populated vector store instance.
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