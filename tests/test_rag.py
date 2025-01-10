import unittest
from rag_bench.metrics import compute_recall_at_k, compute_faithfulness

class TestRAG(unittest.TestCase):
    def test_recall(self):
        self.assertEqual(compute_recall_at_k(["d1", "d2"], ["d1"], k=1), 1.0)
    def test_faithfulness(self):
        score = compute_faithfulness("Neural networks use layers.", ["Neural networks use layers of neurons."])
        self.assertEqual(score, 1.0)

if __name__ == "__main__":
    unittest.main()
