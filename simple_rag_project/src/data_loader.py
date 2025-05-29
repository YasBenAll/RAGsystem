# src/data_loader.py
import os
from typing import List

def load_text_dataset(file_path: str) -> List[str]:
    """
    Loads a text dataset where each line is considered a separate entry/chunk.

    Args:
        file_path (str): The path to the text file.

    Returns:
        List[str]: A list of strings, where each string is a line from the file.
    """
    print(f"Loading dataset from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            dataset = [line.strip() for line in file if line.strip()] # Read lines and remove empty ones
        print(f"Loaded {len(dataset)} entries.")
        return dataset
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return []


if __name__ == "__main__":
    # Create a dummy file for testing
    dummy_file_path = "test_data.txt"
    with open(dummy_file_path, "w", encoding="utf-8") as f:
        f.write("Line 1 of test data.\n")
        f.write("Line 2 of test data.\n")
        f.write("\n") # Empty line
        f.write("Line 3 of test data.\n")

    loaded_data = load_text_dataset(dummy_file_path)
    print("\nContent loaded:")
    for item in loaded_data:
        print(f"- {item}")

    os.remove(dummy_file_path) # Clean up