# examples/basic_usage.py
import numpy as np
from mini_vectordb.store import VectorStore

def main():
    store = VectorStore(dim=3)

    store.add("fruit_apple", np.array([1.0, 0.0, 0.0]), text="Apple is a fruit")
    store.add("fruit_banana", np.array([0.9, 0.1, 0.0]), text="Banana is also a fruit")
    store.add("animal_dog", np.array([0.0, 1.0, 0.0]), text="Dog is an animal")
    store.add("vehicle_car", np.array([0.0, 0.0, 1.0]), text="Car is a vehicle")

    query = np.array([1.0, 0.0, 0.0])  # should match "fruit" vectors most closely

    print(f"Query vector: {query}\n")
    results = store.search(query, k=3)

    for record, score in results:
        print(f"id={record.id:15s} score={score:.4f}  text={record.text}")

if __name__ == "__main__":
    main()