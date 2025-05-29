# src/vector_store.py

import numpy as np
from typing import List, Tuple

class InMemoryVectorStore:
    """
    A simple in-memory vector database for storing text chunks and their embeddings.
    Provides functionality to add chunks and retrieve relevant ones based on cosine similarity.
    """
    def __init__(self):
        # Each element will be a tuple (chunk_text, embedding_vector)
        self.vector_db: List[Tuple[str, np.ndarray]] = []

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculates the cosine similarity between two vectors.
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0 # Handle zero vectors to avoid division by zero
        return dot_product / (norm_a * norm_b)

    def add_chunk(self, chunk_text: str, embedding: List[float]):
        """
        Adds a text chunk and its embedding to the vector database.

        Args:
            chunk_text (str): The text content of the chunk.
            embedding (List[float]): The embedding vector for the chunk.
        """
        # Convert embedding list to numpy array for efficient operations
        self.vector_db.append((chunk_text, np.array(embedding)))
        # print(f"Added chunk to database. Total chunks: {len(self.vector_db)}")

    def retrieve(self, query_embedding: List[float], top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieves the top_n most relevant chunks based on cosine similarity to the query embedding.

        Args:
            query_embedding (List[float]): The embedding of the query.
            top_n (int): The number of top relevant chunks to retrieve.

        Returns:
            List[Tuple[str, float]]: A list of (chunk_text, similarity_score) tuples,
                                      sorted by similarity in descending order.
        """
        query_vec = np.array(query_embedding)
        similarities = []
        for chunk, embedding_vec in self.vector_db:
            similarity = self._cosine_similarity(query_vec, embedding_vec)
            similarities.append((chunk, similarity))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_n]

# Example usage (for testing this module directly)
if __name__ == "__main__":
    print("Testing InMemoryVectorStore...")
    vec_store = InMemoryVectorStore()

    # Dummy embeddings (in a real scenario, these would come from your embedding model)
    # Ensure they are numpy arrays if you're adding them directly for testing
    vec_store.add_chunk("apple is a fruit", [0.1, 0.2, 0.3])
    vec_store.add_chunk("banana is yellow", [0.2, 0.3, 0.1])
    vec_store.add_chunk("red apple pie", [0.15, 0.25, 0.35])
    vec_store.add_chunk("dog barks loudly", [0.8, 0.7, 0.6])

    query_vec = [0.12, 0.22, 0.32] # Query for "apple" related content
    retrieved = vec_store.retrieve(query_vec, top_n=2)

    print("\nRetrieved chunks:")
    for chunk, sim in retrieved:
        print(f"  - (Similarity: {sim:.2f}) {chunk}")

    query_vec_dog = [0.7, 0.6, 0.5] # Query for "dog" related content
    retrieved_dog = vec_store.retrieve(query_vec_dog, top_n=1)
    print("\nRetrieved chunk for 'dog':")
    for chunk, sim in retrieved_dog:
        print(f"  - (Similarity: {sim:.2f}) {chunk}")