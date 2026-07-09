import numpy as np
from mini_vectordb.store import VectorStore

def main():
    store = VectorStore(dim=3)

    store.add("fruit_apple", np.array([1.0, 0.0, 0.0]), text="Apple is a fruit", metadata={"category": "fruit"})
    store.add("fruit_banana", np.array([0.9, 0.1, 0.0]), text="Banana is also a fruit", metadata={"category": "fruit"})
    store.add("animal_dog", np.array([0.0, 1.0, 0.0]), text="Dog is an animal", metadata={"category": "animal"})
    store.add("vehicle_car", np.array([0.0, 0.0, 1.0]), text="Car is a vehicle", metadata={"category": "vehicle"})

    query = np.array([1.0, 0.0, 0.0])  # should match "fruit" vectors most closely

    print("=== Search without filter ===")
    results = store.search(query, k=4)
    for record, score in results:
        print(f"id={record.id:15s} score={score:.4f}  text={record.text}")
    
    print("\n=== Search with metadata_filter={'category': 'fruit'} ===")
    filtered_results = store.search(query, k=4, metadata_filter={"category": "fruit"})
    for record, score in filtered_results:
        print(f"id={record.id:15s} score={score:.4f}  text={record.text}")

if __name__ == "__main__":
    main()