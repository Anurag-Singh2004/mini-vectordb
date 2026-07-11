import json
import numpy as np
from mini_vectordb.store import Record, VectorStore

def save_to_file(store: VectorStore, path:str)-> None:
    data = {
        "dim": store.dim,
        "records": [
            {
                "id": record.id,
                "vector": record.vector.tolist(),
                "text": record.text,
                "metadata": record.metadata
            }
            for record in store._records.values()
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f)
    

def load_from_file(path:str)-> VectorStore:
    with open(path, "r") as f:
        data = json.load(f)
    
    store = VectorStore(dim=data["dim"])
    for record_data in data["records"]:
        store.add(
            id=record_data["id"],
            vector=np.array(record_data["vector"]),
            text=record_data["text"],
            metadata=record_data["metadata"],
        )
    return store       
