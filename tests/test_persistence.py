import os
import numpy as np
from mini_vectordb.store import VectorStore

def test_save_and_load_roundtrip(tmp_path):
    store = VectorStore(dim=2)
    store.add("a", np.array([1.0, 0.0]), text="hello", metadata={"category": "fruit"})
    store.add("b", np.array([0.0, 1.0]), text="world", metadata={"category": "animal"})

    file_path = os.path.join(tmp_path, "test_store.json")
    store.save(file_path)

    loaded_store = VectorStore.load(file_path)

    assert len(loaded_store) == 2
    record = loaded_store.get("a")
    assert record.text == "hello"
    assert record.metadata == {"category": "fruit"}
    assert np.array_equal(record.vector, np.array([1.0, 0.0]))


def test_load_preserves_search_behavior(tmp_path):
    store = VectorStore(dim=2)
    store.add("same", np.array([1.0, 0.0]))
    store.add("opposite", np.array([-1.0, 0.0]))

    file_path = os.path.join(tmp_path, "test_store.json")
    store.save(file_path)
    loaded_store = VectorStore.load(file_path)

    results = loaded_store.search(np.array([1.0, 0.0]), k=1)
    assert results[0][0].id == "same"