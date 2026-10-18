# Install rank_bm25 if not available (lightweight, pure-Python BM25 implementation)
# pip install already done

import pandas as pd
import numpy as np
import ast
import re
import warnings
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 120)

print("[OK] All imports successful")

# -- Load CSVs ------------------------------------------------------------------
corpus = pd.read_csv('corpus.csv')
train  = pd.read_csv('train.csv')
test   = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

# -- Load pre-computed embeddings -----------------------------------------------
corpus_embeddings     = np.load('corpus_embeddings.npy')           # (3600, 384)
corpus_chunk_ids      = np.load('corpus_chunk_ids.npy', allow_pickle=True)  # (3600,)
test_question_embs    = np.load('test_question_embeddings.npy')    # (800, 384)
test_question_ids     = np.load('test_question_ids.npy', allow_pickle=True) # (800,)

print("=" * 70)
print("DATA SHAPES")
print("=" * 70)
for name, df in [('corpus', corpus), ('train', train), ('test', test), ('sample_submission', sample_sub)]:
    print(f"  {name:20s} -> {str(df.shape):12s}  columns: {list(df.columns)}")

print(f"\n  {'corpus_embeddings':20s} -> {str(corpus_embeddings.shape):12s}  dtype: {corpus_embeddings.dtype}")
print(f"  {'corpus_chunk_ids':20s} -> {str(corpus_chunk_ids.shape):12s}  sample: {corpus_chunk_ids[:3]}")
print(f"  {'test_question_embs':20s} -> {str(test_question_embs.shape):12s}  dtype: {test_question_embs.dtype}")
print(f"  {'test_question_ids':20s} -> {str(test_question_ids.shape):12s}  sample: {test_question_ids[:3]}")

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
for name, df in [('corpus', corpus), ('train', train), ('test', test)]:
    nulls = df.isnull().sum()
    if nulls.any():
        print(f"  [WARN]  {name}: {nulls[nulls > 0].to_dict()}")
    else:
        print(f"  [OK]  {name}: No missing values")

# -- Inspect corpus ------------------------------------------------------------
print("CORPUS — Sample Rows")
print("-" * 70)
print(corpus.head(3))

print(f"\nDocument types: {corpus['doc_type'].nunique()} unique")
print(corpus['doc_type'].value_counts().to_string())
print(f"\nAvg chunk_text length: {corpus['chunk_text'].str.len().mean():.0f} chars")
print(f"Unique doc_ids: {corpus['doc_id'].nunique()} | Chunks per doc: {corpus.groupby('doc_id').size().mean():.1f}")

# -- Inspect train -------------------------------------------------------------
print("TRAIN — Sample Rows")
print("-" * 70)
print(train.head(5))

# Parse gold_chunk_ids from string to list
train['parsed_gold'] = train['gold_chunk_ids'].apply(ast.literal_eval)
train['num_gold'] = train['parsed_gold'].apply(len)

print(f"\nGold chunks per question:")
print(train['num_gold'].value_counts().sort_index().to_string())
print(f"\nAvg answer length: {train['answer'].str.len().mean():.0f} chars")

# Classify question types
def classify_question(q):
    if q.startswith('Based on'):
        return 'summarize'
    elif q.startswith('Answer using'):
        return 'key_detail'
    elif q.startswith('Combine evidence'):
        return 'multi_doc'
    elif q.startswith('According to'):
        return 'sla_policy'
    else:
        return 'other'

train['q_type'] = train['question'].apply(classify_question)
print(f"\nQuestion type distribution:")
print(train['q_type'].value_counts().to_string())

# -- Inspect test & submission format ------------------------------------------
print("TEST — Sample Rows")
print("-" * 70)
print(test.head(3))

print("\nSAMPLE SUBMISSION — Format")
print("-" * 70)
print(sample_sub.head(3))
print(f"\nSubmission columns: {list(sample_sub.columns)}")
print(f"Submission rows: {len(sample_sub)} (should match test: {len(test)})")

# ==============================================================================
# 3a. Build BM25 Index
# ==============================================================================

# Build a lookup from chunk_id to all chunk metadata
chunk_lookup = corpus.set_index('chunk_id').to_dict('index')

# Tokenize corpus texts for BM25
# We include title + doc_type + chunk_text for richer matching, since questions
# often reference document titles (e.g., "Architecture note: CDN / Data Pipeline")
corpus_texts_for_bm25 = []
for _, row in corpus.iterrows():
    combined = f"{row['title']} {row['doc_type']} {row['chunk_text']}"
    corpus_texts_for_bm25.append(combined.lower().split())

bm25_index = BM25Okapi(corpus_texts_for_bm25)

# Map from chunk_id to index for fast lookups
chunk_id_to_idx = {cid: i for i, cid in enumerate(corpus['chunk_id'])}
idx_to_chunk_id = {i: cid for cid, i in chunk_id_to_idx.items()}

print(f"[OK] BM25 index built over {len(corpus_texts_for_bm25)} chunks")
print(f"   Avg tokens per chunk: {np.mean([len(t) for t in corpus_texts_for_bm25]):.0f}")

# ==============================================================================
# 3b. Semantic Embedding Index
# ==============================================================================

# The competition provides pre-computed embeddings using sentence-transformers/all-MiniLM-L6-v2
# These are L2-normalized, so cosine similarity = dot product (very fast!)

# Verify normalization
norms = np.linalg.norm(corpus_embeddings, axis=1)
print(f"Corpus embedding norms — min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}")
print(f"[OK] Embeddings are {'normalized' if np.allclose(norms, 1.0, atol=1e-3) else 'NOT normalized'}")

# Verify alignment between chunk_ids in corpus.csv and corpus_chunk_ids.npy
assert list(corpus_chunk_ids) == list(corpus['chunk_id']), "Chunk ID mismatch between CSV and NPY!"
print(f"[OK] Chunk IDs aligned between corpus.csv and corpus_chunk_ids.npy")

# ==============================================================================
# 3c. Hybrid Retriever Class
# ==============================================================================

class HybridRetriever:
    """
    Combines BM25 (lexical) and semantic embedding retrieval.
    
    Design Rationale:
    - BM25 handles exact keyword matches (incident IDs, dates, specific terms)
    - Semantic handles meaning-level similarity (paraphrases, conceptual matches)
    - Hybrid fusion gives the best of both worlds
    
    The alpha parameter controls the blend:
    - alpha=1.0 -> pure semantic
    - alpha=0.0 -> pure BM25
    - alpha=0.5 -> equal weight (our starting point)
    """
    
    def __init__(self, bm25_index, corpus_embeddings, corpus_chunk_ids, corpus_df, alpha=0.5):
        self.bm25 = bm25_index
        self.embeddings = corpus_embeddings
        self.chunk_ids = list(corpus_chunk_ids)
        self.corpus_df = corpus_df
        self.alpha = alpha
        self.n_chunks = len(corpus_chunk_ids)
    
    def retrieve_bm25(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Retrieve using BM25. Returns [(chunk_id, score), ...]"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], scores[i]) for i in top_indices]
    
    def retrieve_semantic(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        """Retrieve using cosine similarity with pre-computed embeddings."""
        # Since embeddings are normalized, dot product = cosine similarity
        similarities = query_embedding @ self.embeddings.T
        if similarities.ndim > 1:
            similarities = similarities.flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.chunk_ids[i], similarities[i]) for i in top_indices]
    
    def retrieve_hybrid(
        self, 
        query: str, 
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        retrieval_pool: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Hybrid retrieval combining BM25 and semantic scores.
        
        Process:
        1. Get top-N candidates from BOTH BM25 and semantic
        2. Normalize scores to [0, 1] range for fair combination
        3. Combine with weighted average: alpha * semantic + (1-alpha) * bm25
        4. Return top-K from the fused ranking
        
        Args:
            query: The question text
            query_embedding: Pre-computed embedding for the query (optional)
            top_k: Number of results to return
            retrieval_pool: How many candidates to pull from each retriever before fusion
        """
        # Get candidates from both retrievers
        bm25_results = self.retrieve_bm25(query, top_k=retrieval_pool)
        
        if query_embedding is not None:
            semantic_results = self.retrieve_semantic(query_embedding, top_k=retrieval_pool)
        else:
            # Fallback: use BM25 only if no embedding available
            return bm25_results[:top_k]
        
        # Collect all candidate chunk_ids
        all_candidates = set()
        bm25_scores = {}
        semantic_scores = {}
        
        for cid, score in bm25_results:
            all_candidates.add(cid)
            bm25_scores[cid] = score
        
        for cid, score in semantic_results:
            all_candidates.add(cid)
            semantic_scores[cid] = score
        
        # Normalize scores to [0, 1]
        def normalize(scores_dict):
            if not scores_dict:
                return scores_dict
            vals = list(scores_dict.values())
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return {k: 1.0 for k in scores_dict}
            return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}
        
        bm25_norm = normalize(bm25_scores)
        semantic_norm = normalize(semantic_scores)
        
        # Fuse scores
        fused = {}
        for cid in all_candidates:
            s_bm25 = bm25_norm.get(cid, 0.0)
            s_sem = semantic_norm.get(cid, 0.0)
            fused[cid] = self.alpha * s_sem + (1 - self.alpha) * s_bm25
        
        # Sort and return top-K
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked
    
    def get_chunk_text(self, chunk_id: str) -> str:
        """Get the text of a chunk by its ID."""
        idx = self.corpus_df[self.corpus_df['chunk_id'] == chunk_id].index
        if len(idx) > 0:
            return self.corpus_df.loc[idx[0], 'chunk_text']
        return ""
    
    def get_chunk_info(self, chunk_id: str) -> dict:
        """Get all metadata for a chunk."""
        row = self.corpus_df[self.corpus_df['chunk_id'] == chunk_id]
        if len(row) > 0:
            return row.iloc[0].to_dict()
        return {}


# Instantiate the retriever
retriever = HybridRetriever(
    bm25_index=bm25_index,
    corpus_embeddings=corpus_embeddings,
    corpus_chunk_ids=corpus_chunk_ids,
    corpus_df=corpus,
    alpha=0.5
)

print("[OK] Hybrid retriever initialized (α=0.5)")

# ==============================================================================
# 3d. Retrieval Quality Check on Train Set
# ==============================================================================

# For train questions, we don't have pre-computed embeddings,
# so we test with BM25-only retrieval first.
# Later we'll generate embeddings for train questions if needed.

def evaluate_retrieval(retriever, questions_df, top_k_values=[1, 3, 5, 10], 
                       query_embeddings=None, max_eval=None):
    """
    Evaluate retrieval recall@K on questions with gold chunk IDs.
    
    Recall@K = fraction of gold chunks found in top-K retrieved chunks.
    We report both per-question recall (macro) and overall chunk-level recall (micro).
    """
    results = {k: [] for k in top_k_values}
    
    eval_df = questions_df.head(max_eval) if max_eval else questions_df
    
    for idx, row in eval_df.iterrows():
        gold = set(row['parsed_gold'])
        q_emb = query_embeddings[idx] if query_embeddings is not None else None
        
        retrieved = retriever.retrieve_hybrid(
            row['question'], 
            query_embedding=q_emb,
            top_k=max(top_k_values)
        )
        
        retrieved_ids = [cid for cid, _ in retrieved]
        
        for k in top_k_values:
            top_k_ids = set(retrieved_ids[:k])
            recall = len(gold & top_k_ids) / len(gold) if gold else 0
            results[k].append(recall)
    
    print(f"\n{'Recall@K':>12s} | {'Mean':>8s} | {'Perfect (=1.0)':>14s} | {'Zero (=0.0)':>12s}")
    print("-" * 55)
    for k in top_k_values:
        vals = results[k]
        mean_r = np.mean(vals)
        perfect = sum(1 for v in vals if v == 1.0)
        zero = sum(1 for v in vals if v == 0.0)
        print(f"  Recall@{k:<3d} | {mean_r:>7.3f} | {perfect:>6d}/{len(vals):<6d} | {zero:>5d}/{len(vals):<5d}")
    
    return results

print("Evaluating BM25-only retrieval on train set...")
print("(No semantic embeddings for train questions — using BM25 fallback)")
retriever_bm25_only = HybridRetriever(
    bm25_index=bm25_index,
    corpus_embeddings=corpus_embeddings,
    corpus_chunk_ids=corpus_chunk_ids,
    corpus_df=corpus,
    alpha=0.0  # Pure BM25
)
bm25_results = evaluate_retrieval(retriever_bm25_only, train, top_k_values=[1, 3, 5, 10])

# Try to generate train embeddings with sentence-transformers
# If not available, we'll use a TF-IDF-based approximation

train_question_embeddings = None

try:
    from sentence_transformers import SentenceTransformer
    
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("Encoding train questions...")
    train_question_embeddings = st_model.encode(
        train['question'].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    print(f"[OK] Generated train embeddings: {train_question_embeddings.shape}")
    
except ImportError:
    print("[WARN]  sentence-transformers not available.")
    print("   Falling back to TF-IDF cosine similarity for semantic component.")
    print("   To get better results, install: pip install sentence-transformers")
    
    # TF-IDF fallback: build a TF-IDF matrix over corpus + questions
    # This gives us an approximate semantic signal
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    corpus_tfidf = tfidf.fit_transform(
        corpus['title'] + ' ' + corpus['chunk_text']
    )
    train_tfidf = tfidf.transform(train['question'])
    
    # Convert to dense numpy arrays for compatibility
    # We'll use these as a pseudo-embedding
    print(f"   TF-IDF corpus matrix: {corpus_tfidf.shape}")
    print(f"   TF-IDF train matrix: {train_tfidf.shape}")

# ==============================================================================
# 3f. Evaluate Hybrid Retrieval (if embeddings available)
# ==============================================================================

if train_question_embeddings is not None:
    print("Evaluating HYBRID retrieval (BM25 + Semantic) on train set...")
    hybrid_results = evaluate_retrieval(
        retriever, train, 
        top_k_values=[1, 3, 5, 10],
        query_embeddings=train_question_embeddings
    )
    
    # Also try different alpha values to find the sweet spot
    print("\n" + "=" * 60)
    print("ALPHA TUNING — Recall@5 for different BM25/Semantic weights")
    print("=" * 60)
    
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        retriever.alpha = alpha
        recalls = []
        for idx, row in train.iterrows():
            gold = set(row['parsed_gold'])
            results = retriever.retrieve_hybrid(
                row['question'],
                query_embedding=train_question_embeddings[idx],
                top_k=5
            )
            retrieved_ids = set(cid for cid, _ in results)
            recalls.append(len(gold & retrieved_ids) / len(gold))
        print(f"  α={alpha:.1f}  ->  Recall@5 = {np.mean(recalls):.4f}")
    
    # Reset to best alpha (will be updated based on results)
    retriever.alpha = 0.5  # Default; update after seeing results above
else:
    print("Using BM25-only retrieval (no train embeddings available)")
    print("The pipeline will still work, but hybrid retrieval is preferred.")
    retriever.alpha = 0.0  # Pure BM25 fallback

# ==============================================================================
# 4a. Answer Generation Functions
# ==============================================================================

def extract_sentences(text: str) -> List[str]:
    """Split text into sentences. Handles common abbreviations."""
    # Simple sentence splitter that respects common patterns
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def score_sentence_relevance(sentence: str, question: str) -> float:
    """
    Score how relevant a sentence is to the question using word overlap.
    
    This is a simple but effective heuristic: sentences that share more
    content words with the question are more likely to contain the answer.
    We use this instead of an LLM to stay fully offline.
    """
    # Extract content words (lowercase, remove stopwords)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
                 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                 'before', 'after', 'above', 'below', 'between', 'and', 'but', 'or',
                 'not', 'no', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'each',
                 'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such',
                 'than', 'too', 'very', 'just', 'about', 'up', 'out', 'if', 'then',
                 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                 'it', 'its', 'he', 'she', 'they', 'them', 'we', 'you', 'i', 'me',
                 'my', 'your', 'his', 'her', 'our', 'their', 'how', 'when', 'where',
                 'why', 'based', 'using', 'evidence', 'answer', 'according', 'knowledge',
                 'base', 'combine', 'summarize', 'key', 'point', 'detail', 'correct',
                 'ignore', 'unrelated', 'notes'}
    
    q_words = set(re.findall(r'\w+', question.lower())) - stopwords
    s_words = set(re.findall(r'\w+', sentence.lower())) - stopwords
    
    if not q_words:
        return 0.0
    
    overlap = len(q_words & s_words)
    return overlap / len(q_words)


def generate_extractive_answer(
    question: str, 
    retrieved_chunks: List[Tuple[str, float]], 
    corpus_df: pd.DataFrame,
    max_sentences: int = 3
) -> str:
    """
    Generate an answer by extracting the most relevant sentences from retrieved chunks.
    
    Strategy:
    1. Get text from top retrieved chunks
    2. Split each chunk into sentences
    3. Score each sentence by relevance to the question
    4. Return the top sentence(s)
    
    This approach guarantees grounding (answer comes directly from source)
    and is sufficient for this competition where gold answers are often
    direct extractions from chunks.
    """
    all_sentences = []
    
    for chunk_id, score in retrieved_chunks[:5]:  # Consider top-5 chunks
        row = corpus_df[corpus_df['chunk_id'] == chunk_id]
        if len(row) == 0:
            continue
        chunk_text = row.iloc[0]['chunk_text']
        sentences = extract_sentences(chunk_text)
        
        for sent in sentences:
            relevance = score_sentence_relevance(sent, question)
            # Boost sentences from higher-ranked chunks
            combined_score = relevance * 0.7 + score * 0.3
            all_sentences.append((sent, combined_score, chunk_id))
    
    if not all_sentences:
        # Fallback: return the full text of the top chunk
        if retrieved_chunks:
            cid = retrieved_chunks[0][0]
            row = corpus_df[corpus_df['chunk_id'] == cid]
            if len(row) > 0:
                return row.iloc[0]['chunk_text']
        return ""
    
    # Sort by combined score and take the best sentences
    all_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # For single-chunk questions, usually 1-2 sentences is enough
    # For multi-chunk questions, we may need sentences from multiple chunks
    selected = []
    seen_chunks = set()
    
    for sent, score, cid in all_sentences:
        if len(selected) >= max_sentences:
            break
        selected.append(sent)
        seen_chunks.add(cid)
    
    return ' '.join(selected)


def generate_answer_for_question(
    question: str,
    retrieved_chunks: List[Tuple[str, float]],
    corpus_df: pd.DataFrame,
    q_type: str = 'unknown'
) -> str:
    """
    Main answer generation function. Routes to the appropriate strategy
    based on question type.
    
    For multi-doc questions: extracts and combines from multiple chunks.
    For single-doc questions: extracts from the top chunk.
    """
    if q_type == 'multi_doc':
        # For "Combine evidence" questions, try to get sentences from
        # at least 2 different chunks to cover both gold sources
        sentences_by_chunk = defaultdict(list)
        
        for chunk_id, score in retrieved_chunks[:5]:
            row = corpus_df[corpus_df['chunk_id'] == chunk_id]
            if len(row) == 0:
                continue
            chunk_text = row.iloc[0]['chunk_text']
            sents = extract_sentences(chunk_text)
            for sent in sents:
                rel = score_sentence_relevance(sent, question)
                sentences_by_chunk[chunk_id].append((sent, rel))
        
        # Get the best sentence from each of the top-2 chunks
        combined_parts = []
        for chunk_id, score in retrieved_chunks[:2]:
            if chunk_id in sentences_by_chunk:
                best_sent = max(sentences_by_chunk[chunk_id], key=lambda x: x[1])
                combined_parts.append(best_sent[0])
        
        if combined_parts:
            return ' '.join(combined_parts)
    
    # Default: extractive answer from top chunks
    return generate_extractive_answer(
        question, retrieved_chunks, corpus_df,
        max_sentences=2 if q_type != 'multi_doc' else 3
    )


print("[OK] Answer generation functions defined")

# Test on a few examples from each question type
print("ANSWER GENERATION — Sample Outputs")
print("=" * 70)

for q_type in ['summarize', 'key_detail', 'multi_doc', 'sla_policy']:
    sample = train[train['q_type'] == q_type].head(2)
    print(f"\n{'-' * 70}")
    print(f"Question Type: {q_type}")
    print(f"{'-' * 70}")
    
    for _, row in sample.iterrows():
        q_emb = train_question_embeddings[row.name] if train_question_embeddings is not None else None
        retrieved = retriever.retrieve_hybrid(
            row['question'], query_embedding=q_emb, top_k=5
        )
        
        answer = generate_answer_for_question(
            row['question'], retrieved, corpus, q_type=q_type
        )
        
        print(f"\n  Q: {row['question'][:100]}...")
        print(f"  Gold answer:      {row['answer'][:100]}")
        print(f"  Generated answer: {answer[:100]}")
        print(f"  Gold chunks: {row['gold_chunk_ids']}")
        print(f"  Retrieved top-3: {[cid for cid, _ in retrieved[:3]]}")

# ==============================================================================
# 5. Citation Selection
# ==============================================================================

def select_citations(
    question: str,
    retrieved_chunks: List[Tuple[str, float]],
    q_type: str = 'unknown',
    score_threshold: float = 0.1,
    max_single: int = 1,
    max_multi: int = 2
) -> List[str]:
    """
    Select which chunks to cite based on question type and retrieval scores.
    
    Policy:
    - Multi-doc questions -> cite top-2 (these questions explicitly reference 2 sources)
    - Single-doc questions -> cite top-1 (over-citation is penalized)
    - Apply score threshold to avoid citing very low-confidence chunks
    
    This policy is tuned for the precision/recall/over-citation tradeoff:
    - Citing too few -> hurts recall
    - Citing too many -> hurts precision AND triggers over-citation penalty
    - Citing exactly right -> maximizes all three
    """
    if not retrieved_chunks:
        return []
    
    # Determine how many citations based on question type
    if q_type == 'multi_doc':
        max_citations = max_multi
    else:
        max_citations = max_single
    
    # Select top chunks above threshold
    selected = []
    for chunk_id, score in retrieved_chunks:
        if len(selected) >= max_citations:
            break
        if score >= score_threshold or len(selected) == 0:  # Always cite at least 1
            selected.append(chunk_id)
    
    return selected


print("[OK] Citation selection policy defined")
print(f"   Single-doc questions: cite top-1")
print(f"   Multi-doc questions:  cite top-2")

# ==============================================================================
# 6. Local Evaluation Functions
# ==============================================================================

def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for scoring."""
    return re.findall(r'\w+', text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 score between prediction and reference.
    
    This approximates the competition's Answer Quality metric:
    - Precision: what fraction of predicted tokens are in the reference?
    - Recall: what fraction of reference tokens are in the prediction?
    - F1: harmonic mean of precision and recall
    
    Robust to minor phrasing differences since it operates on token bags.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    
    # Count shared tokens (taking minimum count for each token)
    shared = sum((pred_counts & ref_counts).values())
    
    precision = shared / sum(pred_counts.values()) if pred_counts else 0
    recall = shared / sum(ref_counts.values()) if ref_counts else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def citation_precision(predicted_chunks: List[str], gold_chunks: List[str]) -> float:
    """Of the chunks we cited, how many are actually gold?"""
    if not predicted_chunks:
        return 0.0
    pred_set = set(predicted_chunks)
    gold_set = set(gold_chunks)
    return len(pred_set & gold_set) / len(pred_set)


def citation_recall(predicted_chunks: List[str], gold_chunks: List[str]) -> float:
    """Of the gold chunks, how many did we cite?"""
    if not gold_chunks:
        return 1.0  # No gold chunks to miss
    pred_set = set(predicted_chunks)
    gold_set = set(gold_chunks)
    return len(pred_set & gold_set) / len(gold_set)


def grounding_consistency(answer: str, cited_chunk_ids: List[str], corpus_df: pd.DataFrame) -> float:
    """
    Measure how well the answer is grounded in the cited chunks.
    
    Computed as: fraction of answer tokens that appear in the cited chunk texts.
    Higher = answer is more faithful to cited evidence.
    Lower = answer contains unsupported claims (hallucination).
    
    This is our proxy for the competition's Grounding Consistency metric.
    """
    if not answer or not cited_chunk_ids:
        return 0.0
    
    # Gather all text from cited chunks
    cited_text = ""
    for cid in cited_chunk_ids:
        row = corpus_df[corpus_df['chunk_id'] == cid]
        if len(row) > 0:
            cited_text += " " + row.iloc[0]['chunk_text']
    
    answer_tokens = set(tokenize(answer))
    cited_tokens = set(tokenize(cited_text))
    
    # Remove stopwords for a more meaningful grounding check
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                 'and', 'but', 'or', 'not', 'no', 'so', 'if', 'then',
                 'it', 'its', 'this', 'that', 'these', 'those'}
    
    answer_content = answer_tokens - stopwords
    
    if not answer_content:
        return 1.0  # All stopwords -> vacuously grounded
    
    grounded = len(answer_content & cited_tokens) / len(answer_content)
    return grounded


def over_citation_penalty(predicted_chunks: List[str], gold_chunks: List[str]) -> float:
    """
    Penalty for citing more chunks than necessary.
    
    Returns a value in [0, 1] where:
    - 1.0 = no over-citation (|predicted| <= |gold|)
    - 0.0 = extreme over-citation
    
    Penalty formula: max(0, 1 - (excess / max(1, |gold|)))
    where excess = max(0, |predicted| - |gold|)
    """
    n_pred = len(set(predicted_chunks))
    n_gold = len(set(gold_chunks))
    
    excess = max(0, n_pred - n_gold)
    penalty = max(0.0, 1.0 - excess / max(1, n_gold))
    return penalty


def evaluate_pipeline(
    predictions: pd.DataFrame,
    gold_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Full 5-component evaluation of the RAG pipeline.
    
    Args:
        predictions: DataFrame with columns [question_id, answer, cited_chunk_ids]
        gold_df: DataFrame with columns [question_id, answer, gold_chunk_ids (parsed)]
        corpus_df: The corpus DataFrame
    
    Returns:
        Dict with component scores and overall score
    """
    scores = {
        'answer_quality': [],
        'citation_precision': [],
        'citation_recall': [],
        'grounding_consistency': [],
        'over_citation_penalty': []
    }
    
    # Per-question-type tracking
    scores_by_type = defaultdict(lambda: defaultdict(list))
    
    for _, pred_row in predictions.iterrows():
        qid = pred_row['question_id']
        gold_row = gold_df[gold_df['question_id'] == qid]
        
        if len(gold_row) == 0:
            continue
        
        gold_row = gold_row.iloc[0]
        
        # Parse predicted citations
        pred_citations = pred_row['cited_chunk_ids']
        if isinstance(pred_citations, str):
            try:
                pred_citations = ast.literal_eval(pred_citations)
            except:
                pred_citations = []
        
        gold_citations = gold_row['parsed_gold']
        q_type = gold_row.get('q_type', 'unknown')
        
        # Compute each component
        aq = token_f1(str(pred_row['answer']), str(gold_row['answer']))
        cp = citation_precision(pred_citations, gold_citations)
        cr = citation_recall(pred_citations, gold_citations)
        gc = grounding_consistency(str(pred_row['answer']), pred_citations, corpus_df)
        ocp = over_citation_penalty(pred_citations, gold_citations)
        
        scores['answer_quality'].append(aq)
        scores['citation_precision'].append(cp)
        scores['citation_recall'].append(cr)
        scores['grounding_consistency'].append(gc)
        scores['over_citation_penalty'].append(ocp)
        
        # Track by question type
        for key in scores:
            scores_by_type[q_type][key].append(scores[key][-1])
    
    # Compute means
    means = {k: np.mean(v) for k, v in scores.items()}
    
    # Overall score (equal weighting — adjust if competition weights differ)
    means['overall'] = np.mean(list(means.values()))
    
    if verbose:
        print("\n" + "=" * 60)
        print("PIPELINE EVALUATION — 5 Component Scores")
        print("=" * 60)
        print(f"\n  {'Component':<25s} {'Score':>8s}  {'Status'}")
        print("  " + "-" * 50)
        
        thresholds = {'answer_quality': 0.5, 'citation_precision': 0.7, 
                      'citation_recall': 0.7, 'grounding_consistency': 0.7,
                      'over_citation_penalty': 0.8}
        
        for key in ['answer_quality', 'citation_precision', 'citation_recall',
                     'grounding_consistency', 'over_citation_penalty']:
            score = means[key]
            status = '[OK] Good' if score >= thresholds[key] else '[WARN]  Needs work' if score >= 0.4 else '[FAIL] Critical'
            print(f"  {key:<25s} {score:>7.4f}  {status}")
        
        print(f"  {'-' * 50}")
        print(f"  {'OVERALL':<25s} {means['overall']:>7.4f}")
        
        # Breakdown by question type
        print(f"\n\n  SCORES BY QUESTION TYPE")
        print(f"  {'Type':<15s}", end="")
        for key in ['answer_quality', 'citation_precision', 'citation_recall', 'grounding_consistency']:
            short = key.replace('citation_', 'cit_').replace('grounding_', 'gnd_').replace('over_citation_', 'oc_')
            print(f" {short:>12s}", end="")
        print(f" {'count':>8s}")
        print(f"  {'-' * 70}")
        
        for q_type in sorted(scores_by_type.keys()):
            print(f"  {q_type:<15s}", end="")
            for key in ['answer_quality', 'citation_precision', 'citation_recall', 'grounding_consistency']:
                vals = scores_by_type[q_type][key]
                print(f" {np.mean(vals):>11.4f}", end="")
            print(f" {len(scores_by_type[q_type]['answer_quality']):>8d}")
    
    return means


print("[OK] Evaluation functions defined (5 components + breakdown by question type)")

# ==============================================================================
# 7a. Run Pipeline on Full Train Set
# ==============================================================================

def run_pipeline(
    questions_df: pd.DataFrame,
    retriever: HybridRetriever,
    corpus_df: pd.DataFrame,
    query_embeddings: Optional[np.ndarray] = None,
    top_k_retrieve: int = 10,
    progress_interval: int = 200
) -> pd.DataFrame:
    """
    Run the full RAG pipeline on a set of questions.
    
    Steps per question:
    1. Classify question type
    2. Retrieve top-K chunks (hybrid: BM25 + semantic)
    3. Generate extractive answer from top chunks
    4. Select citations based on question type and retrieval scores
    
    Returns a DataFrame matching the submission format.
    """
    results = []
    
    for i, (idx, row) in enumerate(questions_df.iterrows()):
        # Step 1: Classify question type
        q_type = classify_question(row['question'])
        
        # Step 2: Retrieve
        q_emb = query_embeddings[idx] if query_embeddings is not None else None
        retrieved = retriever.retrieve_hybrid(
            row['question'], query_embedding=q_emb, top_k=top_k_retrieve
        )
        
        # Step 3: Generate answer
        answer = generate_answer_for_question(
            row['question'], retrieved, corpus_df, q_type=q_type
        )
        
        # Step 4: Select citations
        citations = select_citations(
            row['question'], retrieved, q_type=q_type
        )
        
        results.append({
            'question_id': row['question_id'],
            'answer': answer,
            'cited_chunk_ids': str(citations)
        })
        
        if (i + 1) % progress_interval == 0:
            print(f"  Processed {i + 1}/{len(questions_df)} questions...")
    
    print(f"  [OK] Done: {len(results)} questions processed")
    return pd.DataFrame(results)


print("Running pipeline on full train set (2,200 questions)...")
print("This may take a minute...\n")

train_predictions = run_pipeline(
    train, retriever, corpus,
    query_embeddings=train_question_embeddings,
    top_k_retrieve=10
)

# ==============================================================================
# 7b. Evaluate Train Predictions
# ==============================================================================

train_scores = evaluate_pipeline(train_predictions, train, corpus, verbose=True)

# ==============================================================================
# 7c. Error Analysis
# ==============================================================================

# Find examples where each component scored worst
print("ERROR ANALYSIS — Worst Examples by Component")
print("=" * 70)

# Compute per-question scores
per_question_scores = []

for _, pred_row in train_predictions.iterrows():
    qid = pred_row['question_id']
    gold_row = train[train['question_id'] == qid].iloc[0]
    
    pred_citations = ast.literal_eval(pred_row['cited_chunk_ids'])
    gold_citations = gold_row['parsed_gold']
    
    per_question_scores.append({
        'question_id': qid,
        'q_type': gold_row['q_type'],
        'answer_quality': token_f1(str(pred_row['answer']), str(gold_row['answer'])),
        'citation_precision': citation_precision(pred_citations, gold_citations),
        'citation_recall': citation_recall(pred_citations, gold_citations),
        'pred_answer_len': len(str(pred_row['answer'])),
        'gold_answer_len': len(str(gold_row['answer'])),
        'n_pred_citations': len(pred_citations),
        'n_gold_citations': len(gold_citations),
    })

pq_df = pd.DataFrame(per_question_scores)

# Show distribution of scores
print("\nScore Distribution Summary:")
print(pq_df[['answer_quality', 'citation_precision', 'citation_recall']].describe().round(3))

# Citation accuracy analysis
print(f"\n\nCitation Accuracy:")
print(f"  Perfect citation precision: {(pq_df['citation_precision'] == 1.0).sum()}/{len(pq_df)} "
      f"({(pq_df['citation_precision'] == 1.0).mean():.1%})")
print(f"  Perfect citation recall:    {(pq_df['citation_recall'] == 1.0).sum()}/{len(pq_df)} "
      f"({(pq_df['citation_recall'] == 1.0).mean():.1%})")
print(f"  Zero citation precision:    {(pq_df['citation_precision'] == 0.0).sum()}/{len(pq_df)} "
      f"({(pq_df['citation_precision'] == 0.0).mean():.1%})")

# Show a few worst answer quality examples
print("\n\nWorst Answer Quality Examples (bottom 5):")
print("-" * 70)
worst_aq = pq_df.nsmallest(5, 'answer_quality')
for _, row in worst_aq.iterrows():
    qid = row['question_id']
    gold = train[train['question_id'] == qid].iloc[0]
    pred = train_predictions[train_predictions['question_id'] == qid].iloc[0]
    print(f"  [{qid}] type={row['q_type']} | AQ={row['answer_quality']:.3f}")
    print(f"    Gold: {str(gold['answer'])[:80]}...")
    print(f"    Pred: {str(pred['answer'])[:80]}...")
    print()

# ==============================================================================
# 8a. Improvement: Question-Aware Retrieval (Document Title Matching)
# ==============================================================================

def extract_doc_references(question: str) -> List[str]:
    """
    Extract document title references from the question text.
    
    Many questions explicitly name their source documents, e.g.:
    - "Based on the knowledge base, summarize the key point from: Architecture note: CDN / Data Pipeline"
    - "Combine evidence from 'EU cdn_issue incident report (2025-07-13)' and 'Cdn Config Note: operations'"
    
    By extracting these references, we can boost chunks from matching documents.
    """
    refs = []
    
    # Pattern 1: "from: <title>" (summarize questions)
    match = re.search(r'from:\s*(.+?)(?:\.|$)', question)
    if match:
        refs.append(match.group(1).strip())
    
    # Pattern 2: 'title' in quotes (key_detail and multi_doc questions)
    quoted = re.findall(r"'([^']+)'", question)
    refs.extend(quoted)
    
    # Pattern 3: According to the <type> (sla_policy questions)
    match = re.search(r'According to the\s+(.+?),', question)
    if match:
        refs.append(match.group(1).strip())
    
    return refs


def title_match_score(title: str, references: List[str]) -> float:
    """
    Score how well a document title matches the question's references.
    Returns a value in [0, 1].
    """
    if not references:
        return 0.0
    
    title_lower = title.lower()
    best_score = 0.0
    
    for ref in references:
        ref_lower = ref.lower()
        ref_words = set(re.findall(r'\w+', ref_lower))
        title_words = set(re.findall(r'\w+', title_lower))
        
        if not ref_words:
            continue
        
        overlap = len(ref_words & title_words) / len(ref_words)
        best_score = max(best_score, overlap)
    
    return best_score


class EnhancedRetriever(HybridRetriever):
    """
    Enhanced retriever that also uses document title matching.
    
    This adds a third signal: when the question explicitly names a document,
    we boost chunks from documents with matching titles.
    """
    
    def retrieve_enhanced(
        self, 
        question: str, 
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
        title_boost: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Enhanced retrieval with title matching boost.
        
        Process:
        1. Get hybrid (BM25 + semantic) scores
        2. Extract document references from the question
        3. Boost scores for chunks whose document title matches the references
        """
        # Get base hybrid results
        hybrid_results = self.retrieve_hybrid(
            question, query_embedding=query_embedding, 
            top_k=top_k * 3,  # Get more candidates for reranking
            retrieval_pool=100
        )
        
        # Extract document references from question
        doc_refs = extract_doc_references(question)
        
        if not doc_refs:
            return hybrid_results[:top_k]
        
        # Boost scores for title matches
        boosted = []
        for chunk_id, score in hybrid_results:
            info = self.get_chunk_info(chunk_id)
            if info:
                t_score = title_match_score(info.get('title', ''), doc_refs)
                boosted_score = score + title_boost * t_score
            else:
                boosted_score = score
            boosted.append((chunk_id, boosted_score))
        
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted[:top_k]


# Create enhanced retriever
enhanced_retriever = EnhancedRetriever(
    bm25_index=bm25_index,
    corpus_embeddings=corpus_embeddings,
    corpus_chunk_ids=corpus_chunk_ids,
    corpus_df=corpus,
    alpha=retriever.alpha  # Use the same alpha as the tuned retriever
)

# Test on a few examples
print("Enhanced Retriever — Title Matching Examples")
print("=" * 60)
for _, row in train.head(3).iterrows():
    refs = extract_doc_references(row['question'])
    print(f"  Q: {row['question'][:80]}...")
    print(f"  References found: {refs}")
    print(f"  Gold: {row['gold_chunk_ids']}")
    
    q_emb = train_question_embeddings[row.name] if train_question_embeddings is not None else None
    enhanced = enhanced_retriever.retrieve_enhanced(
        row['question'], query_embedding=q_emb, top_k=5
    )
    print(f"  Enhanced top-5: {[cid for cid, _ in enhanced]}")
    print()

# ==============================================================================
# 8b. Improvement: Better Answer Extraction
# ==============================================================================

def generate_improved_answer(
    question: str,
    retrieved_chunks: List[Tuple[str, float]],
    corpus_df: pd.DataFrame,
    q_type: str = 'unknown'
) -> str:
    """
    Improved answer generation with better strategies per question type.
    
    Key improvements over v1:
    1. For short chunks (< 200 chars), use the full chunk text instead of 
       extracting individual sentences. Many gold answers ARE the chunk text.
    2. For multi-doc, extract key info from each of the top-2 chunks.
    3. For SLA questions, look for specific severity/target patterns.
    """
    if not retrieved_chunks:
        return ""
    
    top_chunk_id = retrieved_chunks[0][0]
    top_chunk_row = corpus_df[corpus_df['chunk_id'] == top_chunk_id]
    
    if len(top_chunk_row) == 0:
        return ""
    
    top_text = top_chunk_row.iloc[0]['chunk_text']
    
    if q_type == 'multi_doc':
        # For multi-doc questions, combine key sentences from top-2 chunks
        parts = []
        for chunk_id, score in retrieved_chunks[:2]:
            row = corpus_df[corpus_df['chunk_id'] == chunk_id]
            if len(row) > 0:
                chunk_text = row.iloc[0]['chunk_text']
                # Use the most relevant sentence from each chunk
                sentences = extract_sentences(chunk_text)
                if sentences:
                    scored = [(s, score_sentence_relevance(s, question)) for s in sentences]
                    best = max(scored, key=lambda x: x[1])
                    parts.append(best[0])
                else:
                    parts.append(chunk_text)
        return ' '.join(parts)
    
    elif q_type == 'sla_policy':
        # SLA questions ask about specific severity targets
        # Look for sentences mentioning severity, acknowledge, mitigation
        sentences = extract_sentences(top_text)
        sla_keywords = ['severity', 'acknowledge', 'mitigation', 'target', 'sla', 'response']
        
        best_sent = top_text  # Fallback to full text
        best_score = 0
        
        for sent in sentences:
            sent_lower = sent.lower()
            kw_count = sum(1 for kw in sla_keywords if kw in sent_lower)
            relevance = score_sentence_relevance(sent, question)
            combined = kw_count * 0.3 + relevance * 0.7
            if combined > best_score:
                best_score = combined
                best_sent = sent
        
        return best_sent
    
    else:
        # For summarize and key_detail questions:
        # If the chunk is short enough, return it directly
        # (many gold answers ARE the full chunk text)
        if len(top_text) <= 250:
            return top_text
        
        # Otherwise, extract the best 1-2 sentences
        sentences = extract_sentences(top_text)
        if not sentences:
            return top_text
        
        scored = [(s, score_sentence_relevance(s, question)) for s in sentences]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 1-2 sentences
        selected = [s for s, _ in scored[:2]]
        return ' '.join(selected)


print("[OK] Improved answer generation function defined")

# ==============================================================================
# 8c. Run Improved Pipeline on Train Set
# ==============================================================================

def run_improved_pipeline(
    questions_df: pd.DataFrame,
    retriever: EnhancedRetriever,
    corpus_df: pd.DataFrame,
    query_embeddings: Optional[np.ndarray] = None,
    top_k_retrieve: int = 10,
    progress_interval: int = 500
) -> pd.DataFrame:
    """
    Improved pipeline with enhanced retrieval and better answer generation.
    """
    results = []
    
    for i, (idx, row) in enumerate(questions_df.iterrows()):
        q_type = classify_question(row['question'])
        q_emb = query_embeddings[idx] if query_embeddings is not None else None
        
        # Enhanced retrieval with title matching
        retrieved = retriever.retrieve_enhanced(
            row['question'], query_embedding=q_emb, top_k=top_k_retrieve
        )
        
        # Improved answer generation
        answer = generate_improved_answer(
            row['question'], retrieved, corpus_df, q_type=q_type
        )
        
        # Citation selection (same policy — it's already good)
        citations = select_citations(
            row['question'], retrieved, q_type=q_type
        )
        
        results.append({
            'question_id': row['question_id'],
            'answer': answer,
            'cited_chunk_ids': str(citations)
        })
        
        if (i + 1) % progress_interval == 0:
            print(f"  Processed {i + 1}/{len(questions_df)} questions...")
    
    print(f"  [OK] Done: {len(results)} questions processed")
    return pd.DataFrame(results)


print("Running IMPROVED pipeline on train set...\n")

train_predictions_v2 = run_improved_pipeline(
    train, enhanced_retriever, corpus,
    query_embeddings=train_question_embeddings,
    top_k_retrieve=10
)

# ==============================================================================
# 8d. Evaluate Improved Pipeline
# ==============================================================================

print("IMPROVED PIPELINE (v2) — Evaluation")
train_scores_v2 = evaluate_pipeline(train_predictions_v2, train, corpus, verbose=True)

# Compare v1 vs v2
print("\n\n" + "=" * 60)
print("COMPARISON: Baseline (v1) vs Improved (v2)")
print("=" * 60)
print(f"  {'Component':<25s} {'v1':>8s} {'v2':>8s} {'Δ':>8s}")
print("  " + "-" * 50)
for key in ['answer_quality', 'citation_precision', 'citation_recall', 
            'grounding_consistency', 'over_citation_penalty', 'overall']:
    v1 = train_scores.get(key, 0)
    v2 = train_scores_v2.get(key, 0)
    delta = v2 - v1
    arrow = '^' if delta > 0 else 'v' if delta < 0 else '->'
    print(f"  {key:<25s} {v1:>7.4f} {v2:>7.4f} {arrow}{abs(delta):>6.4f}")

# ==============================================================================
# 9a. Prepare Test Data with Embeddings
# ==============================================================================

# Verify test question IDs match
assert list(test_question_ids) == list(test['question_id']), \
    "Test question IDs don't match between CSV and NPY!"
print(f"[OK] Test question IDs verified: {len(test)} questions")
print(f"   ID range: {test['question_id'].iloc[0]} to {test['question_id'].iloc[-1]}")
print(f"   Embeddings shape: {test_question_embs.shape}")

# ==============================================================================
# 9b. Run Pipeline on Test Set
# ==============================================================================

print("Running IMPROVED pipeline on test set (800 questions)...")
print("Using pre-computed test question embeddings for hybrid retrieval.\n")

test_predictions = run_improved_pipeline(
    test, enhanced_retriever, corpus,
    query_embeddings=test_question_embs,
    top_k_retrieve=10,
    progress_interval=200
)

# ==============================================================================
# 9c. Validate & Write Submission
# ==============================================================================

# -- Validation checks ---------------------------------------------------------
print("SUBMISSION VALIDATION")
print("=" * 60)

# Check row count
assert len(test_predictions) == len(sample_sub), \
    f"Row count mismatch: {len(test_predictions)} vs {len(sample_sub)}"
print(f"  [OK] Row count: {len(test_predictions)} (matches sample submission)")

# Check columns
assert set(test_predictions.columns) == set(sample_sub.columns), \
    f"Column mismatch: {set(test_predictions.columns)} vs {set(sample_sub.columns)}"
print(f"  [OK] Columns: {list(test_predictions.columns)}")

# Check question IDs
assert list(test_predictions['question_id']) == list(sample_sub['question_id']), \
    "Question ID order mismatch!"
print(f"  [OK] Question IDs match sample submission")

# Check no empty answers
empty_answers = test_predictions['answer'].isna() | (test_predictions['answer'] == '')
print(f"  {'[OK]' if not empty_answers.any() else '[WARN] '} Empty answers: {empty_answers.sum()}")

# Check cited_chunk_ids are valid
valid_chunk_ids = set(corpus['chunk_id'])
invalid_citations = 0
for _, row in test_predictions.iterrows():
    try:
        cited = ast.literal_eval(row['cited_chunk_ids'])
        for cid in cited:
            if cid not in valid_chunk_ids:
                invalid_citations += 1
    except:
        invalid_citations += 1
print(f"  {'[OK]' if invalid_citations == 0 else '[WARN] '} Invalid citations: {invalid_citations}")

# Citation count distribution
citation_counts = test_predictions['cited_chunk_ids'].apply(
    lambda x: len(ast.literal_eval(x))
)
print(f"\n  Citation count distribution:")
print(f"  {citation_counts.value_counts().sort_index().to_dict()}")

# -- Sample predictions --------------------------------------------------------
print("\n\nSAMPLE PREDICTIONS")
print("-" * 60)
print(test_predictions.head(5))

# -- Write submission file -----------------------------------------------------
submission_path = 'submission.csv'
test_predictions.to_csv(submission_path, index=False)

# Verify the written file
verify = pd.read_csv(submission_path)
print(f"\n[OK] Submission written to: {submission_path}")
print(f"   Shape: {verify.shape}")
print(f"   Columns: {list(verify.columns)}")
print(f"   File size: {pd.io.common.file_exists(submission_path)}")

# Final comparison with sample submission format
print("\n   First row comparison:")
print(f"   Sample: {sample_sub.iloc[0].to_dict()}")
print(f"   Ours:   {verify.iloc[0].to_dict()}")

print("="*60)
print("* PIPELINE COMPLETE")
print("="*60)
print(f"\n  > Submission file: submission.csv")
print(f"  > Test predictions: {len(test_predictions)} questions")
print(f"  > Train validation scores:")
for key, val in train_scores_v2.items():
    print(f"     {key}: {val:.4f}")
print(f"\n  Upload 'submission.csv' to Kaggle to see your leaderboard score!")