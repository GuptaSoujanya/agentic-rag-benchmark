import math
from typing import List, Set

def compute_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Fraction of relevant documents retrieved in top-k."""
    if not ground_truth_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & ground_truth_ids)
    return hits / len(ground_truth_ids)

def compute_precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & ground_truth_ids)
    return hits / k

def compute_mrr(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """Mean Reciprocal Rank (MRR) of the first relevant document."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0

def compute_ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    top_k = retrieved_ids[:k]
    for i, doc_id in enumerate(top_k, start=1):
        rel = 1.0 if doc_id in ground_truth_ids else 0.0
        dcg += rel / math.log2(i + 1)

    # Ideal DCG (all relevant docs ranked first)
    ideal_hits = min(len(ground_truth_ids), k)
    if ideal_hits == 0:
        return 1.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

    if not ground_truth_ids:
        return 1.0
