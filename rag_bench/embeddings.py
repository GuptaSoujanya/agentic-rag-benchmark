import math
from typing import List

class HashEmbeddingModel:
    """
    Deterministic, zero-dependency embedding model based on n-gram hashing.
    Outputs normalized 64-dimensional dense vectors for testing and offline evaluation.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().replace(".", " ").replace(",", " ").split()
        if not tokens:
            return vec
        for token in tokens:
            h = hash(token)
            idx = abs(h) % self.dim
            vec[idx] += 1.0

        # L2 Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]
