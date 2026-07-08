from dataclasses import dataclass, field
from typing import Any
import numpy as np
import heapq
from mini_vectordb.similarity import cosine_similarity

@dataclass
class Record:
    id: str
    vector: np.ndarray
    text: str=""
    metadata: dict[str, Any] = field(default_factory=dict)

class VectorStore:
    def __init__(self,dim: int):
        self.dim = dim
        self._records: dict[str, Record] = {}
    
    def add(self, id:str, vector: np.ndarray,text:str= "", metadata: dict | None= None) -> None:
        if vector.shape != (self.dim,):
            raise ValueError(f"Expected vector of shape ({self.dim},), got {vector.shape}")
        
        self._records[id] = Record(id=id, vector=vector, text=text, metadata=metadata or {})
    
    def get(self, id:str)->Record:
        if id not in self._records:
            raise KeyError(f"No record found with id '{id}'")
        return self._records[id]
    
    def delete(self,id:str)->None:
        if id not in self._records:
            raise KeyError(f"No record found with id '{id}'")
        del self._records[id]
    
    def __len__(self)->int:
        return len(self._records)
    
    def search(self,query: np.ndarray, k:int=5)-> list[tuple[Record,float]]:
        heap: list[tuple[float, str]] = []

        for record_id, record in self._records.items():
            score = cosine_similarity(query, record.vector)
            heapq.heappush(heap, (score, record_id))
            if len(heap)>k:
                heapq.heappop(heap)
            
        results = sorted(heap, key=lambda x: x[0], reverse=True)
        return [(self._records[record_id], score) for score, record_id in results]
