# src/llm_inference.py

import ollama
from typing import Iterator, Dict, Any

def get_language_model_name() -> str:
    """
    Returns the configured language model name.
    """
    from src.config import LANGUAGE_MODEL
    return LANGUAGE_MODEL

def generate_llm_response(system_prompt: str, user_query: str) -> Iterator[Dict[str, Any]]:
    """
    Generates a response from the LLM using Ollama, streaming the output.

    Args:
        system_prompt (str): The system-level instructions for the LLM.
        user_query (str): The user's question or prompt.

    Returns:
        Iterator[Dict[str, Any]]: An iterator yielding chunks of the LLM's response.
    """
    model_name = get_language_model_name()
    print(f"Generating response with LLM: {model_name}...")
    try:
        stream = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_query},
            ],
            stream=True,
        )
        return stream
    except Exception as e:
        print(f"Error generating LLM response with {model_name}: {e}")
        # Return an empty iterator or raise an error
        return iter([])

# Example usage (for testing this module directly)
if __name__ == "__main__":
    # Note: Ensure Ollama is running and 'llama3' (or your chosen LLM) is pulled
    print("Testing LLM response generation...")
    sys_prompt = "You are a helpful assistant."
    user_q = "What is the capital of France?"

    response_stream = generate_llm_response(sys_prompt, user_q)
    print("Chatbot response:")
    for chunk in response_stream:
        print(chunk['message']['content'], end='', flush=True)
    print("\n--- End of LLM test ---")