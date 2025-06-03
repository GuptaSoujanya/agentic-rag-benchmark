import unittest
from rag_bench.metrics.faithfulness import compute_faithfulness_score, extract_statements

class TestFaithfulness(unittest.TestCase):
    def test_statement_extraction(self):
        text = "Transformers are neural networks. They use attention."
        stmts = extract_statements(text)
        self.assertEqual(len(stmts), 2)

    def test_grounded_response(self):
        context = ["Ashish Vaswani co-authored the Transformer paper in 2017."]
        response = "Ashish Vaswani authored the Transformer architecture paper in 2017."
        score, _ = compute_faithfulness_score(response, context)
        self.assertGreaterEqual(score, 0.8)

if __name__ == "__main__":
    unittest.main()
