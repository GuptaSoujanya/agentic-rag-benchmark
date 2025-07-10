# agentic-rag-benchmark

[![CI](https://github.com/GuptaSoujanya/agentic-rag-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/GuptaSoujanya/agentic-rag-benchmark/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An evaluation and diagnostic toolkit measuring retrieval precision, context recall, hallucination rates, and multi-step reasoning accuracy in Agentic Retrieval-Augmented Generation (RAG) pipelines.

---

## Architectural Topology

```mermaid
flowchart TD
    subgraph Query_Planning["1. Query Decomposition & Agent Trajectory"]
        Q["User Query"]
        AG["Multi-Hop Agent Controller"]
        Q --> AG
        AG -->|Hop 1 Sub-query| VS["Vector Store Index"]
        AG -->|Hop 2 Entity Expansion| VS
    end

    subgraph Metrics_Pipeline["2. Quantitative Evaluation Engine"]
        VS --> RET_M["Retrieval Metrics:<br/>Recall@K, Precision@K, MRR, NDCG@K"]
        AG --> FAITH_M["Faithfulness & Groundedness:<br/>Claim Extraction & Context Verification"]
        VS --> REL_M["Context Relevance:<br/>Sentence-level Noise Filtering"]
    end

    subgraph Output_Dashboard["3. Diagnostic Reporting"]
        RET_M & FAITH_M & REL_M --> REP["CLI Diagnostic Matrix & JSON"]
    end
```

---

## Core Capabilities

- **Retrieval Metrics Suite:** Implements unbiased NDCG@K, MRR (Mean Reciprocal Rank), Precision@K, and Recall@K.
- **Atomic Claim Faithfulness Evaluator:** Parses model outputs into constituent claims and verifies ground truth entailment to catch subtle hallucinations.
- **Agentic Trajectory Tracker:** Evaluates multi-hop query refinement across successive retrieval steps.
- **Zero-Dependency Lightweight Core:** Runs standalone without heavy external vector database servers or API keys.

---

## Installation

```bash
git clone https://github.com/GuptaSoujanya/agentic-rag-benchmark.git
cd agentic-rag-benchmark
pip install -e .
```

---

## Quickstart

Run evaluation on the multi-hop evaluation set:

```bash
python -m rag_bench.cli --data data/multihop_bench.json --top-k 2
```

Sample output:
```text
========================================================
        AGENTIC RAG EVALUATION REPORT
========================================================
 Evaluated Queries       : 2
 Retrieval Recall@2    : 100.00%
 Retrieval Precision@2 : 75.00%
 Mean Reciprocal Rank    : 1.0000
 NDCG@2                : 0.9197
 Faithfulness Score      : 100.00%
 Context Relevance Score : 66.67%
========================================================
```

---

## Running Test Suite

```bash
python -m unittest discover tests -v
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
