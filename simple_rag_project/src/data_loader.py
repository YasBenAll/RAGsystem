# src/data_loader.py
import os
from typing import List
from typing import List, Tuple, Dict, Any
import pypdf # Using pypdf for PDF processing

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

def load_documents(directory: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Loads text content and basic metadata from .txt and .pdf files
    in a specified directory. Each entry is a tuple of (content, metadata).
    """
    documents = []
    print(f"Loading documents from: {directory}")
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at {directory}")
        return []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        content = ""
        metadata = {"source": filename} # Basic metadata

        try:
            if filename.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"  - Loaded TXT: {filename}")
            elif filename.endswith(".pdf"):
                with pypdf.PdfReader(file_path) as reader:
                    num_pages = len(reader.pages)
                    page_contents = []
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text() or ""
                        page_contents.append(page_text)
                        # For PDF, we add a page number to the metadata
                        documents.append((page_text, {"source": filename, "page": i + 1}))
                    # For simplicity, if you want the whole PDF as one chunk,
                    # uncomment the next two lines and comment out the loop above:
                    # content = "\n".join(page_contents)
                    # documents.append((content, metadata))
                print(f"  - Loaded PDF: {filename} ({num_pages} pages)")
                continue # Already added pages individually for PDF

            else:
                print(f"  - Skipping unsupported file: {filename}")
                continue # Skip adding unsupported files

            # For .txt files and other single-chunk documents, add here
            if content:
                documents.append((content, metadata))

        except Exception as e:
            print(f"  - Error loading {filename}: {e}")

    return documents

def split_documents(
    documents: List[Tuple[str, Dict[str, Any]]], # Input type changed
    chunk_size: int,
    chunk_overlap: int
) -> List[Tuple[str, Dict[str, Any]]]: # Return type changed
    """
    Splits a list of (content, metadata) tuples into smaller, overlapping chunks.
    This is a basic character-based splitter.
    """
    print(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    chunks = []
    for doc_content, doc_metadata in documents:
        # Simple character-based splitting
        start = 0
        while start < len(doc_content):
            end = min(start + chunk_size, len(doc_content))
            chunk_text = doc_content[start:end]
            chunk_metadata = doc_metadata.copy() # Copy to avoid modifying original metadata
            chunk_metadata['start_index'] = start # Add start index

            chunks.append((chunk_text, chunk_metadata))
            if end == len(doc_content):
                break
            start += chunk_size - chunk_overlap
    print(f"Split into {len(chunks)} chunks.")
    return chunks


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