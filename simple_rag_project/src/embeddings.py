# src/embeddings.py

import ollama
from typing import List

def get_embedding_model_name() -> str:
    """
    Returns the configured embedding model name.
    """
    from src.config import EMBEDDING_MODEL
    return EMBEDDING_MODEL

def generate_embedding(text: str) -> List[float]:
    """
    Generates an embedding for a given text using Ollama.

    Args:
        text (str): The text to embed.

    Returns:
        List[float]: The embedding vector as a list of floats.
    """
    model_name = get_embedding_model_name()
    try:
        # Ollama's embed function returns a dictionary, we extract the 'embeddings' list
        embedding = ollama.embed(model=model_name, input=text)['embeddings'][0]
        return embedding
    except Exception as e:
        print(f"Error generating embedding with {model_name}: {e}")
        # Return a dummy embedding or raise an error based on desired behavior
        return []

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