from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class RAGBenchConfig:
    top_k: int = 3
    relevance_threshold: float = 0.65
    faithfulness_threshold: float = 0.70
    max_hops: int = 3
