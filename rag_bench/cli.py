import argparse
from .vector_store import InMemoryVectorStore, Document
from .metrics import compute_recall_at_k, compute_ndcg_at_k, compute_faithfulness

def main():
    parser = argparse.ArgumentParser(prog="rag-eval")
    parser.add_argument("--query", type=str, default="Who authored Attention Is All You Need?")
    args = parser.parse_args()

    store = InMemoryVectorStore()
    store.add_documents([
        Document("doc1", "Vaswani et al. published Attention Is All You Need introducing the Transformer in 2017."),
        Document("doc2", "ResNet was introduced by Kaiming He in 2015.")
    ])

    results = store.similarity_search(args.query, top_k=2)
    retrieved_ids = [r[0].doc_id for r in results]
    recall = compute_recall_at_k(retrieved_ids, ["doc1"], k=2)
    ndcg = compute_ndcg_at_k(retrieved_ids, ["doc1"], k=2)
    faithfulness = compute_faithfulness("Vaswani authored the Transformer architecture in 2017.", [r[0].text for r in results])

    print("\n" + "="*52)
    print("         AGENTIC RAG EVALUATION REPORT")
    print("="*52)
    print(f" Query                  : {args.query}")
    print(f" Retrieval Recall@2    : {recall*100:.1f}%")
    print(f" Retrieval NDCG@2      : {ndcg:.4f}")
    print(f" Grounded Faithfulness : {faithfulness*100:.1f}%")
    print("="*52 + "\n")

if __name__ == "__main__":
    main()
