import unittest
from rag_bench.vector_store import InMemoryVectorStore, Document

class TestVectorStore(unittest.TestCase):
    def test_store_and_search(self):
        store = InMemoryVectorStore()
        docs = [
            Document("d1", "Quantum computing uses superposition and entanglement."),
            Document("d2", "Machine learning harnesses gradient descent optimization.")
        ]
        store.add_documents(docs)
        results = store.similarity_search("quantum entanglement", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document.doc_id, "d1")

if __name__ == "__main__":
    unittest.main()
