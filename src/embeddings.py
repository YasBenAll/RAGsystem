# src/embeddings.py

import os # For getting config values directly if needed
import ollama
from typing import List
from chromadb import EmbeddingFunction, Documents, Embeddings # <-- NEW IMPORTS

def get_embedding_model_name() -> str:
    """
    Returns the configured embedding model name.
    """
    from src.config import EMBEDDING_MODEL
    return EMBEDDING_MODEL

def generate_embedding_single(text: str) -> List[float]:
    """
    Generates an embedding for a single text using Ollama.
    This is an internal helper function.
    """
    model_name = get_embedding_model_name()
    if not model_name:
        print("DEBUG: Embedding model name not configured. Check src/config.py.")
        return []

    try:
        response = ollama.embed(model=model_name, input=text) # Use 'input'
        if hasattr(response, 'embeddings') and isinstance(response.embeddings, list):
            return response.embeddings[0]
        else:
            # If the response object doesn't have the 'embeddings' attribute or it's not a list
            print(f"ERROR: Ollama.embed response object has no 'embeddings' attribute or it's not a list for text: '{text[:50]}...'")
            print(f"DEBUG: Full response object: {response}") # Still print full response for debugging
            return []
    except ollama.ResponseError as e:
        print(f"ERROR: Ollama server responded with an error for model '{model_name}': {e}")
        print(f"DEBUG: Ensure Ollama server is running and model '{model_name}' is pulled (`ollama pull {model_name}`).")
        return []
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during embedding for model '{model_name}': {e}")
        return []

# --- NEW: ChromaDB compatible EmbeddingFunction class ---
class OllamaEmbeddingFunction(EmbeddingFunction): # Inherit from EmbeddingFunction
    def __init__(self):
        # Optionally, you can pass the model name here if you want to make it configurable
        # self.model_name = get_embedding_model_name()
        pass # No explicit initialization needed if model_name is always pulled from config

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embeds a list of texts using Ollama.

        Args:
            input (Documents): A list of strings (Documents is a type alias for List[str]).

        Returns:
            Embeddings: A list of embedding vectors (Embeddings is a type alias for List[List[float]]).
        """
        print(f"DEBUG: OllamaEmbeddingFunction __call__ method invoked for {len(input)} texts.")
        embeddings: List[List[float]] = []
        for i, text in enumerate(input):
            embedding = generate_embedding_single(text) # Call our single embedding helper
            if embedding:
                embeddings.append(embedding)
            else:
                # If an embedding failed, we must still return an embedding for that document
                # to maintain list length. Returning zeros or raising an error are options.
                # Returning zeros is often safer for the pipeline to continue.
                # For robust error handling, consider raising an exception here.
                print(f"WARNING: Embedding failed for text {i+1}. Appending zeros to maintain shape.")
                # You'd need to know the embedding dimension. A robust way is to get it from a successful call.
                # For now, let's assume a common dimension like 768 for bge-base-en
                embeddings.append([0.0] * 768) # <-- Fallback: append a zero vector

        print(f"DEBUG: OllamaEmbeddingFunction __call__ finished. Generated {len(embeddings)} embeddings.")
        return embeddings

# Example usage (for testing this module directly)
if __name__ == "__main__":
    # Note: Ensure Ollama is running and 'bge-base-en' is pulled
    # You might need to adjust the model name if you pulled a different one.
    print("Testing embedding generation...")
    sample_text = "Hello, world!"
    embedding = generate_embedding(sample_text)
    if embedding:
        print(f"Embedding for '{sample_text}': {embedding[:5]}... (length: {len(embedding)})")
    else:
        print("Failed to generate embedding.")