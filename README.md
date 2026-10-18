# AgentEval RAG Retrieval Study

Retrieval-augmented question answering over a corpus of IT incident reports, architecture notes and SLA policies (AgentEval Kaggle competition, July 2026). The main question: which retriever works best, BM25, embedding cosine similarity, or a hybrid of the two?

## Finding

On 2,200 training questions (Recall@5, alpha = weight on embeddings):

| BM25 only (0.0) | 0.2 | 0.4 | Hybrid (0.5) | 0.8 | Embeddings only (1.0) |
|---|---|---|---|---|---|
| **0.297** | 0.237 | 0.132 | 0.124 | 0.113 | 0.099 |

BM25 won and recall fell as semantic weight rose. The questions quote exact document titles and IDs, which favours keyword matching, so this may not hold for paraphrased questions. The final pipeline uses BM25 plus a title-match boost and type-aware extractive answers, which raised the local overall score from 0.478 to 0.514.

## Limits

- Answers are extractive, so the grounding score of 1.0 holds by construction and says nothing about correctness.
- Citation precision and recall are about 0.10: the pipeline often finds the right document but cites the wrong chunk.
- Scores are on the training split with a local re-implementation of the metric. No leaderboard rank is claimed.

## Contents

| File | Purpose |
|---|---|
| `rag_pipeline.ipynb` | Exploration, retriever comparison, final pipeline |
| `_run_pipeline.py` | HybridRetriever class and alpha tuning |
| `_run_optimized.py` | Vectorised alpha sweep, type-aware answers |
| `_run_pipeline_v3.py` | Variant with document-level restricted search |
| `RESULTS_AND_CONCLUSIONS.md` | Full results tables |

## Data

The competition data (corpus, train and test CSVs, embeddings) is not included. Download it from the AgentEval Kaggle competition and set the paths at the top of the scripts. Embeddings use `sentence-transformers/all-MiniLM-L6-v2`; BM25 uses `rank_bm25`.
