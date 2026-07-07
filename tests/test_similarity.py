import numpy as np
from mini_vectordb.similarity import cosine_similarity

def test_identical_vectors():
    a = np.array([1.0, 2.0, 3.0])
    assert np.isclose(cosine_similarity(a, a), 1.0)

def test_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert np.isclose(cosine_similarity(a, b), -1.0)


def test_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.isclose(cosine_similarity(a, b), 0.0)


def test_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert cosine_similarity(a, b) == 0.0