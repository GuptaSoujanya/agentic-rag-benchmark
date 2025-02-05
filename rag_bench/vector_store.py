import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .embeddings import HashEmbeddingModel

@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = None

@dataclass
class SearchResult:
    document: Document
    score: float

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(0.0, min(1.0, dot))

class InMemoryVectorStore:
    """In-memory vector store supporting dense cosine similarity retrieval."""

    def __init__(self, embedding_model: Optional[HashEmbeddingModel] = None):
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []

    def add_documents(self, docs: List[Document]):
        for d in docs:
            emb = self.embedding_model.embed_text(d.text)
            self.documents.append(d)
            self.embeddings.append(emb)

    def similarity_search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if not self.documents:
            return []
        q_emb = self.embedding_model.embed_text(query)
        scored = []
        for doc, emb in zip(self.documents, self.embeddings):
            sim = cosine_similarity(q_emb, emb)
            scored.append(SearchResult(document=doc, score=sim))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
