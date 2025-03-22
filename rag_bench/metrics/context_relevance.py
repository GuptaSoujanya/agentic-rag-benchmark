import re
from typing import List

def compute_context_relevance(query: str, retrieved_contexts: List[str]) -> float:
    """Measure proportion of retrieved context sentences relevant to the query."""
    if not retrieved_contexts:
        return 0.0
    query_words = set(w.lower() for w in re.findall(r'\w+', query) if len(w) > 2)
    if not query_words:
        return 1.0

    total_sentences = 0
    relevant_sentences = 0

    for ctx in retrieved_contexts:
        sentences = re.split(r'(?<=[.!?])\s+', ctx.strip())
        for s in sentences:
            if not s.strip():
                continue
            total_sentences += 1
            s_words = set(w.lower() for w in re.findall(r'\w+', s))
            if len(query_words & s_words) > 0:
                relevant_sentences += 1

    return round(relevant_sentences / total_sentences, 4) if total_sentences > 0 else 0.0
