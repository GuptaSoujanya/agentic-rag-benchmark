from dataclasses import dataclass, field
from typing import List, Set, Dict, Any
from .vector_store import InMemoryVectorStore, Document, SearchResult

@dataclass
class HopTrace:
    hop_index: int
    sub_query: str
    retrieved_doc_ids: List[str]

@dataclass
class AgentTrajectory:
    initial_query: str
    hops: List[HopTrace] = field(default_factory=list)
    final_response: str = ""
    accumulated_context: List[str] = field(default_factory=list)

class MultiHopAgent:
    """Simulates an autonomous multi-step reasoning agent with dynamic retrieval."""

    def __init__(self, vector_store: InMemoryVectorStore, max_hops: int = 3):
        self.vector_store = vector_store
        self.max_hops = max_hops

    def execute(self, query: str) -> AgentTrajectory:
        trajectory = AgentTrajectory(initial_query=query)
        current_query = query

        for hop_idx in range(self.max_hops):
            results = self.vector_store.similarity_search(current_query, top_k=2)
            doc_ids = [r.document.doc_id for r in results]
            docs_text = [r.document.text for r in results]

            trajectory.hops.append(HopTrace(
                hop_index=hop_idx + 1,
                sub_query=current_query,
                retrieved_doc_ids=doc_ids
            ))
            trajectory.accumulated_context.extend(docs_text)

            # Heuristic multi-hop refinement (simulating entity hop)
            if hop_idx == 0 and "author" in query.lower():
                current_query = "books written by author awards"
            elif hop_idx == 1:
                break

        # Synthesize final answer from accumulated context
        summary = " ".join(trajectory.accumulated_context[:2])
        trajectory.final_response = f"Based on retrieved records: {summary}"
        return trajectory
