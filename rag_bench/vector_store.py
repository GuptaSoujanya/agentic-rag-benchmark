import math
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Document:
    doc_id: str
    text: str

class InMemoryVectorStore:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.documents: List[Document] = []
        self.vectors: List[List[float]] = []

    def _embed(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for tok in text.lower().split():
            v[abs(hash(tok)) % self.dim] += 1.0
        norm = math.sqrt(sum(x*x for x in v))
        return [x/norm for x in v] if norm > 0 else v

    def add_documents(self, docs: List[Document]):
        for d in docs:
            self.documents.append(d)
            self.vectors.append(self._embed(d.text))

    def similarity_search(self, query: str, top_k: int = 3):
        q_v = self._embed(query)
        scored = []
        for d, v in zip(self.documents, self.vectors):
            dot = sum(a*b for a, b in zip(q_v, v))
            scored.append((d, dot))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
