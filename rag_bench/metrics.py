import math
import re

def compute_recall_at_k(retrieved, ground_truth, k=2):
    if not ground_truth: return 1.0
    hits = len(set(retrieved[:k]) & set(ground_truth))
    return hits / len(ground_truth)

def compute_ndcg_at_k(retrieved, ground_truth, k=2):
    if not ground_truth: return 1.0
    dcg = sum((1.0 if d in ground_truth else 0.0) / math.log2(i + 2) for i, d in enumerate(retrieved[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0

def compute_faithfulness(response, context):
    statements = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if len(s.strip()) > 5]
    if not statements: return 1.0
    ctx_words = set(re.findall(r'\b\w+\b', " ".join(context).lower()))
    supported = 0
    for stmt in statements:
        s_words = [w for w in re.findall(r'\b\w+\b', stmt.lower()) if len(w) > 3]
        if not s_words or sum(1 for w in s_words if w in ctx_words) / len(s_words) >= 0.5:
            supported += 1
    return round(supported / len(statements), 4)
