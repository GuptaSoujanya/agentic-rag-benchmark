import json
from dataclasses import dataclass
from typing import List, Dict, Any
from .vector_store import InMemoryVectorStore, Document
from .agent import MultiHopAgent
from .metrics.retrieval import compute_recall_at_k, compute_precision_at_k, compute_mrr, compute_ndcg_at_k
from .metrics.faithfulness import compute_faithfulness_score
from .metrics.context_relevance import compute_context_relevance

@dataclass
class RAGBenchmarkReport:
    total_queries: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_mrr: float
    mean_ndcg: float
    mean_faithfulness: float
    mean_context_relevance: float

def run_agentic_rag_benchmark(data_path: str, top_k: int = 2) -> RAGBenchmarkReport:
    with open(data_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    recalls, precisions, mrrs, ndcgs = [], [], [], []
    faithfulness_scores, relevance_scores = [], []

    for item in bench_data:
        store = InMemoryVectorStore()
        docs = [Document(doc_id=c["doc_id"], text=c["text"]) for c in item["corpus"]]
        store.add_documents(docs)

        agent = MultiHopAgent(store, max_hops=2)
        trajectory = agent.execute(item["query"])

        all_retrieved = []
        for h in trajectory.hops:
            all_retrieved.extend(h.retrieved_doc_ids)
        # Deduplicate preserving order
        unique_retrieved = list(dict.fromkeys(all_retrieved))
        gt_ids = set(item["ground_truth_doc_ids"])

        recalls.append(compute_recall_at_k(unique_retrieved, gt_ids, k=top_k))
        precisions.append(compute_precision_at_k(unique_retrieved, gt_ids, k=top_k))
        mrrs.append(compute_mrr(unique_retrieved, gt_ids))
        ndcgs.append(compute_ndcg_at_k(unique_retrieved, gt_ids, k=top_k))

        f_score, _ = compute_faithfulness_score(trajectory.final_response, trajectory.accumulated_context)
        faithfulness_scores.append(f_score)

        c_rel = compute_context_relevance(item["query"], trajectory.accumulated_context)
        relevance_scores.append(c_rel)

    n = len(bench_data)
    return RAGBenchmarkReport(
        total_queries=n,
        mean_recall_at_k=round(sum(recalls) / n, 4),
        mean_precision_at_k=round(sum(precisions) / n, 4),
        mean_mrr=round(sum(mrrs) / n, 4),
        mean_ndcg=round(sum(ndcgs) / n, 4),
        mean_faithfulness=round(sum(faithfulness_scores) / n, 4),
        mean_context_relevance=round(sum(relevance_scores) / n, 4)
    )
