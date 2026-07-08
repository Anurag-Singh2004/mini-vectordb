from dataclasses import dataclass, field
from typing import Any
import numpy as np

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