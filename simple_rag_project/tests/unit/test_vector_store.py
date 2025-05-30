# tests/unit/test_vector_store.py

import numpy as np
import pytest # For pytest.approx for float comparisons

# Assuming your InMemoryVectorStore is in src.vector_store
# and conftest.py is set up to allow 'from src import ...'
from src.vector_store import InMemoryVectorStore

# We need an instance of the class to test its method,
# or we can make the method static if it doesn't rely on instance state (which it doesn't).
# For this test, let's create an instance.
# Alternatively, if _cosine_similarity could be a static method or a standalone function,
# testing would be even more straightforward. Given it's an internal helper,
# testing it via an instance is fine.

# Create a fixture for the vector store instance if you plan to add more tests for this class
@pytest.fixture
def vector_store_instance():
    """Provides an instance of InMemoryVectorStore for testing."""
    return InMemoryVectorStore()

def test_cosine_similarity_identical_vectors(vector_store_instance):
    """Test cosine similarity with identical vectors (should be 1.0)."""
    vec_a = np.array([1.0, 2.0, 3.0])
    vec_b = np.array([1.0, 2.0, 3.0])
    similarity = vector_store_instance._cosine_similarity(vec_a, vec_b)
    assert similarity == pytest.approx(1.0)

def test_cosine_similarity_orthogonal_vectors(vector_store_instance):
    """Test cosine similarity with orthogonal vectors (should be 0.0)."""
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([0.0, 1.0, 0.0])
    similarity = vector_store_instance._cosine_similarity(vec_a, vec_b)
    assert similarity == pytest.approx(0.0)

def test_cosine_similarity_opposite_vectors(vector_store_instance):
    """Test cosine similarity with opposite vectors (should be -1.0)."""
    vec_a = np.array([1.0, 2.0, 3.0])
    vec_b = np.array([-1.0, -2.0, -3.0])
    similarity = vector_store_instance._cosine_similarity(vec_a, vec_b)
    assert similarity == pytest.approx(-1.0)

def test_cosine_similarity_one_zero_vector(vector_store_instance):
    """Test cosine similarity when one vector is a zero vector (should be 0.0)."""
    vec_a = np.array([1.0, 2.0, 3.0])
    vec_b = np.array([0.0, 0.0, 0.0])
    similarity = vector_store_instance._cosine_similarity(vec_a, vec_b)
    assert similarity == pytest.approx(0.0)

def test_cosine_similarity_both_zero_vectors(vector_store_instance):
    """Test cosine similarity when both vectors are zero vectors (should be 0.0)."""
    vec_a = np.array([0.0, 0.0, 0.0])
    vec_b = np.array([0.0, 0.0, 0.0])
    similarity = vector_store_instance._cosine_similarity(vec_a, vec_b)
    assert similarity == pytest.approx(0.0)

def test_cosine_similarity_different_magnitudes_same_direction(vector_store_instance):
    """Test with vectors of different magnitudes but same direction (should be 1.0)."""
    vec_a = np.array([1.0, 1.0, 1.0])
    vec_b = np.array([5.0, 5.0, 5.0])
    similarity = vector_store_instance._cosine_similarity(vec_a, vec_b)
    assert similarity == pytest.approx(1.0)

def test_cosine_similarity_general_case_1(vector_store_instance):
    """Test with a general case of non-trivial vectors."""
    vec_a = np.array([1.0, 2.0, 3.0])
    vec_b = np.array([4.0, 5.0, 6.0])
    # Manual calculation for this case:
    # dot_product = (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
    # norm_a = sqrt(1^2 + 2^2 + 3^2) = sqrt(1 + 4 + 9) = sqrt(14)
    # norm_b = sqrt(4^2 + 5^2 + 6^2) = sqrt(16 + 25 + 36) = sqrt(77)
    # similarity = 32 / (sqrt(14) * sqrt(77)) = 32 / sqrt(1078)
    # sqrt(1078) approx 32.8329
    # similarity approx 32 / 32.8329 approx 0.9746
    expected_similarity = 32 / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    similarity = vector_store_instance._cosine_similarity(vec_a, vec_b)
    assert similarity == pytest.approx(expected_similarity)

def test_cosine_similarity_general_case_2_less_similar(vector_store_instance):
    """Test with another general case, expecting lower similarity."""
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([1.0, 1.0, 1.0])
    # dot_product = 1
    # norm_a = 1
    # norm_b = sqrt(3)
    # similarity = 1 / sqrt(3) approx 0.57735
    expected_similarity = 1 / np.sqrt(3)
    similarity = vector_store_instance._cosine_similarity(vec_a, vec_b)
    assert similarity == pytest.approx(expected_similarity)

