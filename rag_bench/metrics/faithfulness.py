import re
from typing import List, Tuple

def extract_statements(response_text: str) -> List[str]:
    """Split response text into discrete atomic claims/sentences."""
    raw_sentences = re.split(r'(?<=[.!?])\s+', response_text.strip())
    statements = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
    return statements or [response_text.strip()]

def compute_faithfulness_score(response_text: str, context_documents: List[str]) -> Tuple[float, List[dict]]:
    """
    Evaluate whether statements in the model response are grounded in the retrieved context.
    Returns faithfulness score (0.0 - 1.0) and statement-level breakdown.
    """
    statements = extract_statements(response_text)
    if not statements:
        return 1.0, []

    combined_context = " ".join(context_documents).lower()
    supported_count = 0
    breakdown = []

    for stmt in statements:
        # Check token overlap / grounding heuristic
        words = [w.lower() for w in re.findall(r'\w+', stmt) if len(w) > 3]
        if not words:
            breakdown.append({"statement": stmt, "grounded": True, "overlap_ratio": 1.0})
            supported_count += 1
            continue

        matched_words = sum(1 for w in words if w in combined_context)
        overlap_ratio = matched_words / len(words)
        is_grounded = overlap_ratio >= 0.55

        if is_grounded:
            supported_count += 1

        breakdown.append({
            "statement": stmt,
            "grounded": is_grounded,
            "overlap_ratio": round(overlap_ratio, 3)
        })

    faithfulness = supported_count / len(statements)
    return round(faithfulness, 4), breakdown
