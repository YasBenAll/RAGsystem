# src/vector_store.py

import chromadb
import numpy as np
from typing import List, Tuple, Dict, Any

# Import the custom embedding function we just created
from src.embeddings import OllamaEmbeddingFunction # <-- Changed import name
from src.config import EMBEDDING_MODEL # To pass to ChromaDB's embedding function setup


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

class PersistentVectorStore:
    """
    A vector database using ChromaDB with disk persistence.
    """
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Instantiate your custom embedding function class
        self.chroma_embedding_function_instance = OllamaEmbeddingFunction() # <-- This creates the instance

        self.collection_name = "rag_knowledge_base"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            # Pass the *instance* of your custom embedding function
            # This ensures Chroma knows to use YOUR Ollama function for both add and query
            embedding_function=self.chroma_embedding_function_instance # <-- Ensure this line is exactly here
        )
        print(f"ChromaDB collection '{self.collection_name}' loaded/created at {persist_directory}.")


    def add_chunks(self, chunks_with_metadata: List[Tuple[str, Dict[str, Any]]]):
        """
        Adds text chunks and their metadata to the ChromaDB collection.
        ChromaDB automatically generates embeddings using the configured embedding_function.

        Args:
            chunks_with_metadata (List[Tuple[str, Dict[str, Any]]]):
                A list of tuples, where each tuple is (chunk_text, metadata_dict).
        """
        if not chunks_with_metadata:
            return

        current_count = self.collection.count()
        # A more robust ID generation might be needed for real applications
        ids = [f"doc_{current_count + i}_{hash(chunk_content) % 100000}"
               for i, (chunk_content, _) in enumerate(chunks_with_metadata)]

        documents = [c[0] for c in chunks_with_metadata]
        metadatas = [c[1] for c in chunks_with_metadata]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} chunks to ChromaDB. Total items: {self.collection.count()}")


    def retrieve(self, query_text: str, top_n: int = 3) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Retrieves top_n most relevant chunks from the ChromaDB collection.

        Args:
            query_text (str): The user's query text.
            top_n (int): The number of top relevant chunks to retrieve.

        Returns:
            List[Tuple[str, Dict[str, Any], float]]: A list of (chunk_content, metadata, distance) tuples,
                                                     sorted by distance (lower is better, meaning more similar).
        """
        # ChromaDB will use its configured embedding function to embed the query_text internally for search
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_n,
            include=['documents', 'distances', 'metadatas']
        )

        retrieved = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                doc_content = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                retrieved.append((doc_content, metadata, distance))
        return retrieved

    def count(self) -> int:
        """Returns the number of items in the collection."""
        return self.collection.count()

    def clear_collection(self):
        """Removes all items from the collection. Useful for development."""
        # To clear a collection by name using the client, you use client.delete_collection
        # Then, you need to get or create it again to have a collection object to work with.
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ollama_embedding_function # Pass the custom EF again
        )
        print(f"ChromaDB collection '{self.collection_name}' cleared.")
    def clear_collection(self):
        """Removes all items from the collection. Useful for development."""
        collection_name_to_clear = self.collection_name # Store it before client reset potentially clears it
        embedding_function_to_reuse = self.chroma_embedding_function_instance # Store it

        # Option 1: If client.reset() is too aggressive and deletes everything including other collections
        # self.client.delete_collection(name=collection_name_to_clear)
        # self.collection = self.client.get_or_create_collection(
        #     name=collection_name_to_clear,
        #     embedding_function=embedding_function_to_reuse # <-- Use the stored instance
        # )
        # print(f"ChromaDB collection '{collection_name_to_clear}' cleared and recreated.")

        # Option 2: A more thorough reset of the client if this method is intended to wipe the slate clean for this path
        # This is generally what you'd want if you are about to delete the directory.
        # However, the main "Clear Vector Store Cache & Rebuild" in app.py uses shutil.rmtree,
        # which is even more thorough. This method is for clearing *within* an existing client.
        # For now, let's assume it's about clearing the specific collection.
        
        print(f"Attempting to delete collection: {collection_name_to_clear}")
        self.client.delete_collection(name=collection_name_to_clear)
        print(f"Collection '{collection_name_to_clear}' deleted.")
        
        print(f"Re-creating collection: {collection_name_to_clear}")
        self.collection = self.client.get_or_create_collection(
            name=collection_name_to_clear,
            embedding_function=embedding_function_to_reuse # <-- CORRECTED
        )
        print(f"ChromaDB collection '{collection_name_to_clear}' cleared and re-created.")

    def _reset_client(self):
        """Resets the underlying ChromaDB client. Useful before directory deletion."""
        if self.client:
            print(f"Resetting ChromaDB client for path: {self.persist_directory}")
            self.client.reset() # This reinitializes the persistent client
            print("ChromaDB client has been reset.")
            # After reset, the self.collection object is likely invalid,
            # so it should be re-established if the vector_store is to be reused.
            # For directory deletion, this might be the last step for this client instance.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, # Use the original name
                embedding_function=self.chroma_embedding_function_instance
            )

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