from src.data_loader import split_documents # Assuming src is in PYTHONPATH or tests are run from project root

def test_split_documents_simple():
    docs = [("This is a test document.", {"source": "doc1"})]
    # Using CHUNK_SIZE and CHUNK_OVERLAP from config or defining them for the test
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP # Or use fixed values for test predictability
    
    # For a predictable test, let's use fixed values here
    test_chunk_size = 10
    test_chunk_overlap = 5

    chunks = split_documents(docs, chunk_size=test_chunk_size, chunk_overlap=test_chunk_overlap)
    
    assert len(chunks) == 4 # Based on "This is a test document."
                           # "This is a "
                           # "is a test "
                           # "test docum"
                           # "document."
    assert chunks[0][0] == "This is a "
    assert chunks[0][1]['source'] == "doc1"
    assert chunks[1][0] == "is a test "
    # ... add more specific assertions for all generated chunks and their metadata
    assert chunks[2][0] == "test docum"
    assert chunks[3][0] == "document."
    assert chunks[3][1]['start_index'] is not None # Example metadata check