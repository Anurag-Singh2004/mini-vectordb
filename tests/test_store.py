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

def test_search_returns_most_similar_first():
    store = VectorStore(dim=2)
    store.add("same", np.array([1.0, 0.0]))
    store.add("orthogonal", np.array([0.0, 1.0]))
    store.add("opposite", np.array([-1.0, 0.0]))

    query = np.array([1.0, 0.0])
    results = store.search(query, k=3)

    ids_in_order = [record.id for record, score in results]
    assert ids_in_order == ["same", "orthogonal", "opposite"]

def test_search_returns_full_record_with_score():
    store = VectorStore(dim=2)
    store.add("doc1", np.array([1.0,0.0]), text="hello", metadata={"source":"a"})

    results = store.search(np.array([1.0,0.0]), k=1)
    record, score = results[0]

    assert record.text =="hello"
    assert record.metadata == {"source": "a"}
    assert np.isclose(score,1.0)

def test_search_k_larger_than_store_size():
    store = VectorStore(dim=2)
    store.add("a", np.array([1.0, 0.0]))
    store.add("b", np.array([0.0, 1.0]))

    results = store.search(np.array([1.0, 0.0]), k=10)
    assert len(results) == 2


def test_search_empty_store_returns_empty_list():
    store = VectorStore(dim=2)
    results = store.search(np.array([1.0, 0.0]), k=5)
    assert results == []