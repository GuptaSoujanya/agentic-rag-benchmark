import argparse
import os
from .benchmark_runner import run_agentic_rag_benchmark

def main():
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="agentic-rag-benchmark: Diagnose Retrieval, Faithfulness & Agentic Trajectories."
    )
    parser.add_argument("--data", type=str, default="data/multihop_bench.json", help="Path to benchmark JSON dataset")
    parser.add_argument("--top-k", type=int, default=2, help="Rank threshold K for retrieval metrics")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Dataset '{args.data}' not found. Please provide valid JSON.")
        return

    report = run_agentic_rag_benchmark(args.data, top_k=args.top_k)

    print("\n" + "="*56)
    print("        AGENTIC RAG EVALUATION REPORT")
    print("="*56)
    print(f" Evaluated Queries       : {report.total_queries}")
    print(f" Retrieval Recall@{args.top_k}    : {report.mean_recall_at_k * 100:.2f}%")
    print(f" Retrieval Precision@{args.top_k} : {report.mean_precision_at_k * 100:.2f}%")
    print(f" Mean Reciprocal Rank    : {report.mean_mrr:.4f}")
    print(f" NDCG@{args.top_k}                : {report.mean_ndcg:.4f}")
    print(f" Faithfulness Score      : {report.mean_faithfulness * 100:.2f}%")
    print(f" Context Relevance Score : {report.mean_context_relevance * 100:.2f}%")
    print("="*56 + "\n")

if __name__ == "__main__":
    main()
