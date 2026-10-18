# Agent Eval Project — Results & Conclusions

> **Project**: RAG (Retrieval-Augmented Generation) Pipeline for Agent Evaluation
> **Dataset**: AgentEval Kaggle Competition
> **Date**: July 2026
> **Status**: COMPLETE — Submission Generated

---

## Project Overview

We built an end-to-end **RAG (Retrieval-Augmented Generation) pipeline** to answer questions grounded in a corpus of enterprise IT incident reports, architecture notes, and SLA policies. The system retrieves relevant document chunks and generates concise, cited answers.

---

## Dataset Summary

| Dataset | Rows | Description |
|---|---|---|
| corpus.csv | 3,600 chunks | 1,200 documents x ~3 chunks each (300-400 words/chunk) |
| train.csv | 2,200 questions | Questions with gold answers + gold chunk citations |
| test.csv | 800 questions | Questions to generate predictions for |
| sample_submission.csv | 800 rows | Required output format |

**Document Types in Corpus:** architecture_note, deployment_log, regional_outage_summary, incident_report, sla_policy, advisory, cdn_config_note

**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 | Dim: 384 | Chunks: 3,600

---

## Question Type Distribution (Train Set)

| Question Type | Count | Description |
|---|---|---|
| summarize | 1,003 | "Based on the knowledge base, summarize..." |
| multi_doc | 554 | "Combine evidence from ..." |
| sla_policy | 324 | "According to the ... policy, ..." |
| key_detail | 319 | "Answer using evidence. What is the correct..." |

---

## What We Built

### 1. Baselines (Methods Folder)
- **Baseline 1** (baseline-1-bm25-tf-idf-retrieval-answer-citation.ipynb): BM25 + TF-IDF lexical retrieval
- **Baseline 2** (baseline-2-embedding-retrieval-rerank.ipynb): Semantic embedding retrieval

### 2. Hybrid retriever, evaluated (rag_pipeline.ipynb, _run_pipeline.py)
A **HybridRetriever** class combining (the alpha sweep selected pure BM25, alpha 0.0, for the final run):
- BM25Okapi — lexical keyword matching (handles incident IDs, dates, specific terms)
- Cosine Similarity on pre-computed embeddings — semantic meaning-level matching
- Score fusion: score = alpha x semantic + (1-alpha) x BM25 (min-max normalized)
- Title matching boost — boosted chunks whose document title matches named references in the question

### 3. Optimized Pipeline (_run_optimized.py, _run_pipeline_v3.py)
- Vectorized matrix operations for fast alpha sweeping
- Type-aware answer generation (different strategies per question type)
- Enhanced citation selection (multi-doc gets 2 citations; others get 1)

---

## Retrieval Performance (BM25 Baseline — Train, n=2200)

| Metric | Value |
|---|---|
| Recall@1 | 0.096 |
| Recall@3 | 0.242 |
| Recall@5 | 0.299 |
| Recall@10 | 0.338 |

Note: Pure BM25 recall is modest in absolute terms, but it was the best retriever tested (see below).

### Retrieval comparison (Train, n=2200, Recall@5, alpha = weight on embeddings)

| alpha | 0.0 (BM25 only) | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 (embeddings only) |
|---|---|---|---|---|---|---|---|
| Recall@5 | **0.2966** | 0.2368 | 0.1320 | 0.1243 | 0.1214 | 0.1132 | 0.0986 |

Recall fell as semantic weight rose. The questions quote exact document titles and IDs, which favours keyword matching. The final pipeline therefore uses alpha = 0.0 (BM25) plus the title-match boost, not a hybrid blend.

---

## Final Pipeline Evaluation Scores (Train Set, n=2200)

### v1 Baseline Pipeline vs v2 Improved Pipeline

| Component | v1 (Baseline) | v2 (Improved) | Delta |
|---|---|---|---|
| Answer Quality (Token F1) | 0.2371 | 0.3740 | +0.1369 |
| Citation Precision | 0.0945 | 0.0980 | +0.0034 |
| Citation Recall | 0.0945 | 0.0980 | +0.0034 |
| Grounding Consistency | 0.9644 | 1.0000 | +0.0356 |
| Over-Citation Penalty | 1.0000 | 1.0000 | 0.0000 |
| **OVERALL** | **0.4781** | **0.5140** | **+0.0359** |

The overall pipeline improved from 0.4781 to 0.5140 across all 5 evaluation components.

---

## Scores by Question Type (v2 Pipeline)

| Question Type | Answer Quality | Citation Precision | Citation Recall | Grounding | Count |
|---|---|---|---|---|---|
| sla_policy | 1.0000 | 0.0031 | 0.0031 | 1.0000 | 324 |
| key_detail | 0.3072 | 0.1567 | 0.1567 | 1.0000 | 319 |
| summarize | 0.3059 | 0.0808 | 0.0808 | 1.0000 | 1,003 |
| multi_doc | 0.1698 | 0.1507 | 0.1507 | 1.0000 | 554 |

**Key Observations:**
- sla_policy questions achieve perfect answer quality (1.0) — answers closely match templated SLA text
- multi_doc questions are hardest — require evidence fusion across 2 separate documents
- Grounding consistency is 1.0 across ALL types — every answer is grounded in a retrieved chunk
- Citation precision/recall are the biggest gap — often retrieves right doc area but wrong chunk

---

## Submission Validation — Test Set (n=800)

| Check | Status |
|---|---|
| Row count: 800 (matches sample) | PASSED |
| Columns: question_id, answer, cited_chunk_ids | PASSED |
| Question IDs match | PASSED |
| Empty answers: 0 | PASSED |
| Invalid citations: 0 | PASSED |
| Citation count distribution | {1: 590, 2: 210} |

Submission file: submission.csv (800 predictions, ready for Kaggle upload)

---

## Is The Agent Working?

### What Is Working Well

| Component | Status | Notes |
|---|---|---|
| Data loading & parsing | WORKING | All CSVs and .npy embeddings load correctly |
| BM25 index | WORKING | Built over 3,600 chunks, ~63 tokens/chunk avg |
| Semantic retrieval | WORKING | Cosine sim via normalized dot-product, fast |
| Hybrid fusion | WORKING | Alpha-blended BM25 + semantic scores |
| Title boosting | WORKING | Named doc references extracted and scored |
| Type-aware answer generation | WORKING | Separate logic per question type |
| Citation selection | WORKING | 1-2 citations per answer, no invalid chunk IDs |
| Grounding Consistency | PERFECT 1.000 | Answers always grounded in retrieved chunks |
| Over-citation Penalty | PERFECT 1.000 | Never over-cites relative to gold |
| Submission format validation | PASSED | All 800 rows valid |

### What Needs Improvement

| Component | Status | Notes |
|---|---|---|
| Citation precision/recall | LOW (~0.098) | Right doc, wrong chunk |
| Answer quality for multi_doc | LOW (0.1698) | Needs generative LLM for true fusion |
| Recall@1 | LOW (0.096) | Top-1 chunk frequently misses gold |
| Overall score ceiling | ~0.51 | Extractive methods limited without LLM |

---

## Key Takeaways

1. **BM25 beat both hybrid fusion and embedding-only retrieval on this dataset** (Recall@5 0.30 vs 0.12 at alpha 0.5 vs 0.10) — the questions quote exact titles and IDs, so keyword matching wins. This may not hold for paraphrased questions.
2. **Title-matching boost was crucial** — questions often explicitly name documents.
3. **Grounding of 1.0 is by construction** — answers are extracted from the chunk that is cited, so this metric cannot fail for an extractive pipeline. It says nothing about answer correctness.
4. **Citation recall is the hardest metric** — requires better retrieval or a reranker.
5. **SLA policy questions are near-perfect** — templated, keyword-predictable answers.
6. **Without a generative LLM, multi-doc synthesis is the weakest link** — extractive methods can't truly fuse evidence.

---

## File Reference

| File | Purpose |
|---|---|
| rag_pipeline.ipynb | Main exploratory notebook — analysis, iteration, final pipeline |
| _run_pipeline.py | Full pipeline script with HybridRetriever class |
| _run_optimized.py | Optimized vectorized pipeline with alpha sweep |
| _run_pipeline_v3.py | V3 with restricted document-level search |
| Methods/baseline-1-*.ipynb | BM25/TF-IDF baseline |
| Methods/baseline-2-*.ipynb | Embedding-only baseline |
| submission.csv | FINAL test predictions (800 rows) |
| corpus_embeddings.npy | Pre-computed corpus embeddings (3600 x 384) |
| test_question_embeddings.npy | Pre-computed test embeddings (800 x 384) |
| embedding_model_info.txt | Embedding model metadata |

---

## Final Score Summary

Overall Pipeline Score (Train Validation): 0.5140
  - Answer Quality (Token F1):             0.3740
  - Citation Precision:                    0.0980
  - Citation Recall:                       0.0980
  - Grounding Consistency:                 1.0000  (PERFECT)
  - Over-Citation Penalty:                 1.0000  (PERFECT)

Submission: submission.csv — 800 test predictions, all valid

---

Generated: July 17, 2026 | Model: sentence-transformers/all-MiniLM-L6-v2 | Retrieval: BM25 + title matching (hybrid retriever evaluated, alpha 0.0 selected)
