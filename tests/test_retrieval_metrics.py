import unittest
from rag_bench.metrics.retrieval import compute_recall_at_k, compute_precision_at_k, compute_mrr, compute_ndcg_at_k

class TestRetrievalMetrics(unittest.TestCase):
    def test_recall_and_precision(self):
        retrieved = ["doc_1", "doc_2", "doc_3"]
        ground_truth = {"doc_1", "doc_4"}
        self.assertAlmostEqual(compute_recall_at_k(retrieved, ground_truth, k=2), 0.5)
        self.assertAlmostEqual(compute_precision_at_k(retrieved, ground_truth, k=2), 0.5)

    def test_mrr(self):
        retrieved = ["doc_wrong", "doc_target"]
        ground_truth = {"doc_target"}
        self.assertAlmostEqual(compute_mrr(retrieved, ground_truth), 0.5)

    def test_ndcg(self):
        retrieved = ["doc_target", "doc_wrong"]
        ground_truth = {"doc_target"}
        self.assertAlmostEqual(compute_ndcg_at_k(retrieved, ground_truth, k=2), 1.0)

if __name__ == "__main__":
    unittest.main()
