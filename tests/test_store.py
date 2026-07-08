import numpy as np
import pytest
from mini_vectordb.store import VectorStore

def test_add_and_get():
    store = VectorStore(dim=3)
    vec = np.array([1.0, 2.0, 3.0])
    store.add("doc1", vec, text="hello world", metadata={"source": "test.pdf"})

    record = store.get("doc1")
    assert record.id == "doc1"
    assert np.array_equal(record.vector, vec)
    assert record.text == "hello world"
    assert record.metadata == {"source": "test.pdf"}

def test_add_wrong_dimension_raises():
    store = VectorStore(dim=3)
    bad_vec = np.array([1.0, 2.0])  
    with pytest.raises(ValueError):
        store.add("doc1", bad_vec)

def test_add_overwrites_existing_id():
    store = VectorStore(dim=2)
    store.add("doc1", np.array([1.0, 1.0]))
    store.add("doc1", np.array([2.0, 2.0]))

    record = store.get("doc1")
    assert np.array_equal(record.vector, np.array([2.0, 2.0]))
    assert len(store) == 1 


def test_get_nonexistent_raises():
    store = VectorStore(dim=2)
    with pytest.raises(KeyError):
        store.get("missing")


def test_delete_removes_record():
    store = VectorStore(dim=2)
    store.add("doc1", np.array([1.0, 1.0]))
    store.delete("doc1")
    assert len(store) == 0

def test_delete_nonexistent_raises():
    store = VectorStore(dim=2)
    with pytest.raises(KeyError):
        store.delete("missing")