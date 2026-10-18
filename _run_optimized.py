# -*- coding: utf-8 -*-
"""
AgentEval RAG Pipeline - Optimized Execution Script
Precomputes scores as matrices for fast alpha tuning.
"""
import pandas as pd
import numpy as np
import ast
import re
import warnings
import time
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 120)

print("[OK] All imports successful")

# ======================================================================
# 1. LOAD DATA
# ======================================================================
corpus = pd.read_csv('corpus.csv')
train  = pd.read_csv('train.csv')
test   = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

corpus_embeddings  = np.load('corpus_embeddings.npy')
corpus_chunk_ids   = np.load('corpus_chunk_ids.npy', allow_pickle=True)
test_question_embs = np.load('test_question_embeddings.npy')
test_question_ids  = np.load('test_question_ids.npy', allow_pickle=True)

print(f"Corpus: {corpus.shape}, Train: {train.shape}, Test: {test.shape}")
print(f"Corpus embeddings: {corpus_embeddings.shape}, Test Q embeddings: {test_question_embs.shape}")

# Parse gold chunk ids
train['parsed_gold'] = train['gold_chunk_ids'].apply(ast.literal_eval)
train['num_gold'] = train['parsed_gold'].apply(len)

# Classify question types
def classify_question(q):
    if q.startswith('Based on'):     return 'summarize'
    elif q.startswith('Answer using'): return 'key_detail'
    elif q.startswith('Combine evidence'): return 'multi_doc'
    elif q.startswith('According to'):  return 'sla_policy'
    else: return 'other'

train['q_type'] = train['question'].apply(classify_question)
print(f"Question types: {train['q_type'].value_counts().to_dict()}")

# ======================================================================
# 2. BUILD INDICES
# ======================================================================
print("\nBuilding BM25 index...")
t0 = time.time()

# Build chunk text lookup (indexed by position for fast access)
chunk_texts = corpus['chunk_text'].tolist()
chunk_ids_list = corpus['chunk_id'].tolist()
chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids_list)}

# BM25 with title + type + text
corpus_texts_for_bm25 = []
for _, row in corpus.iterrows():
    combined = f"{row['title']} {row['doc_type']} {row['chunk_text']}"
    corpus_texts_for_bm25.append(combined.lower().split())

bm25_index = BM25Okapi(corpus_texts_for_bm25)
print(f"  BM25 index built in {time.time()-t0:.1f}s")

# ======================================================================
# 3. GENERATE TRAIN EMBEDDINGS
# ======================================================================
train_question_embeddings = None
try:
    from sentence_transformers import SentenceTransformer
    print("\nEncoding train questions with all-MiniLM-L6-v2...")
    t0 = time.time()
    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    train_question_embeddings = st_model.encode(
        train['question'].tolist(),
        batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    print(f"  Train embeddings: {train_question_embeddings.shape} in {time.time()-t0:.1f}s")
except ImportError:
    print("[WARN] sentence-transformers not available, using BM25 only")

# ======================================================================
# 4. VECTORIZED RETRIEVAL EVALUATION
# ======================================================================

def evaluate_recall_vectorized(questions_df, query_embeddings, bm25, corpus_embs, 
                                chunk_ids, alpha=0.5, top_k=5, 
                                retrieval_pool=50):
    """
    Efficient retrieval evaluation using vectorized operations.
    For each question, computes hybrid scores and checks recall.
    """
    n_questions = len(questions_df)
    recalls = []
    top_retrieved = []
    
    for idx, (_, row) in enumerate(questions_df.iterrows()):
        gold = set(row['parsed_gold'])
        
        # BM25 scores (for top retrieval_pool)
        bm25_scores = bm25.get_scores(row['question'].lower().split())
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:retrieval_pool]
        
        if query_embeddings is not None:
            # Semantic scores
            sim = query_embeddings[idx] @ corpus_embs.T
            sem_top_idx = np.argsort(sim)[::-1][:retrieval_pool]
            
            # Combine candidates
            candidate_idx = list(set(bm25_top_idx.tolist() + sem_top_idx.tolist()))
            
            # Get scores for candidates
            bm25_cand = bm25_scores[candidate_idx]
            sem_cand = sim[candidate_idx]
            
            # Min-max normalize
            def norm(arr):
                mn, mx = arr.min(), arr.max()
                if mx == mn: return np.ones_like(arr)
                return (arr - mn) / (mx - mn)
            
            bm25_n = norm(bm25_cand)
            sem_n = norm(sem_cand)
            
            fused = alpha * sem_n + (1 - alpha) * bm25_n
            
            # Top-K from fused
            top_k_local = np.argsort(fused)[::-1][:top_k]
            retrieved_ids = [chunk_ids[candidate_idx[i]] for i in top_k_local]
        else:
            top_k_idx = bm25_top_idx[:top_k]
            retrieved_ids = [chunk_ids[i] for i in top_k_idx]
        
        recall = len(gold & set(retrieved_ids)) / len(gold) if gold else 0
        recalls.append(recall)
        top_retrieved.append(retrieved_ids)
    
    return np.mean(recalls), recalls, top_retrieved


# Evaluate BM25-only
print("\n--- BM25-only Recall ---")
for k in [1, 3, 5, 10]:
    mean_r, _, _ = evaluate_recall_vectorized(
        train, None, bm25_index, corpus_embeddings, 
        chunk_ids_list, alpha=0.0, top_k=k
    )
    print(f"  Recall@{k}: {mean_r:.4f}")

# Evaluate hybrid with different alphas
if train_question_embeddings is not None:
    print("\n--- Alpha Tuning (Recall@5) ---")
    best_alpha = 0.5
    best_recall = 0
    
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        t0 = time.time()
        mean_r, _, _ = evaluate_recall_vectorized(
            train, train_question_embeddings, bm25_index,
            corpus_embeddings, chunk_ids_list,
            alpha=alpha, top_k=5
        )
        elapsed = time.time() - t0
        marker = " <-- BEST" if mean_r > best_recall else ""
        print(f"  alpha={alpha:.1f}  Recall@5={mean_r:.4f}  ({elapsed:.1f}s){marker}")
        if mean_r > best_recall:
            best_recall = mean_r
            best_alpha = alpha
    
    print(f"\n  Best alpha: {best_alpha} with Recall@5: {best_recall:.4f}")
    
    # Evaluate at different K with best alpha
    print(f"\n--- Hybrid Retrieval (alpha={best_alpha}) ---")
    for k in [1, 3, 5, 10, 20]:
        mean_r, _, _ = evaluate_recall_vectorized(
            train, train_question_embeddings, bm25_index,
            corpus_embeddings, chunk_ids_list,
            alpha=best_alpha, top_k=k
        )
        print(f"  Recall@{k}: {mean_r:.4f}")
else:
    best_alpha = 0.0
    print("Using BM25 only (no embeddings)")

# ======================================================================
# 5. ANSWER GENERATION & CITATION SELECTION
# ======================================================================

def extract_doc_references(question):
    """Extract document references from question text."""
    refs = []
    match = re.search(r'from:\s*(.+?)(?:\.|$)', question)
    if match: refs.append(match.group(1).strip())
    quoted = re.findall(r"'([^']+)'", question)
    refs.extend(quoted)
    match = re.search(r'According to the\s+(.+?),', question)
    if match: refs.append(match.group(1).strip())
    return refs

def title_match_score(title, references):
    """Score how well a doc title matches question references."""
    if not references: return 0.0
    title_lower = title.lower()
    best = 0.0
    for ref in references:
        ref_words = set(re.findall(r'\w+', ref.lower()))
        title_words = set(re.findall(r'\w+', title_lower))
        if not ref_words: continue
        overlap = len(ref_words & title_words) / len(ref_words)
        best = max(best, overlap)
    return best

def retrieve_enhanced(question, query_embedding, bm25, corpus_embs, chunk_ids, 
                       corpus_df, alpha, top_k=10, retrieval_pool=100, title_boost=0.5):
    """
    Enhanced hybrid retrieval with title matching boost.
    """
    bm25_scores = bm25.get_scores(question.lower().split())
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:retrieval_pool]
    
    if query_embedding is not None:
        sim = query_embedding @ corpus_embs.T
        sem_top_idx = np.argsort(sim)[::-1][:retrieval_pool]
        candidate_idx = list(set(bm25_top_idx.tolist() + sem_top_idx.tolist()))
        
        bm25_cand = bm25_scores[candidate_idx]
        sem_cand = sim[candidate_idx]
        
        def norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx == mn: return np.ones_like(arr)
            return (arr - mn) / (mx - mn)
        
        bm25_n = norm(bm25_cand)
        sem_n = norm(sem_cand)
        fused = alpha * sem_n + (1 - alpha) * bm25_n
    else:
        candidate_idx = bm25_top_idx.tolist()
        fused_vals = bm25_scores[candidate_idx]
        mn, mx = fused_vals.min(), fused_vals.max()
        if mx == mn: fused = np.ones(len(candidate_idx))
        else: fused = (fused_vals - mn) / (mx - mn)
    
    # Title matching boost
    doc_refs = extract_doc_references(question)
    if doc_refs:
        for j, ci in enumerate(candidate_idx):
            title = corpus_df.iloc[ci]['title']
            t_score = title_match_score(title, doc_refs)
            fused[j] += title_boost * t_score
    
    top_k_local = np.argsort(fused)[::-1][:top_k]
    return [(chunk_ids[candidate_idx[i]], float(fused[i])) for i in top_k_local]


def extract_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def score_sentence_relevance(sentence, question):
    stopwords = {'the','a','an','is','are','was','were','be','been','being',
                 'have','has','had','do','does','did','will','would','could',
                 'should','may','might','shall','can','to','of','in','for',
                 'on','with','at','by','from','as','into','through','during',
                 'before','after','above','below','between','and','but','or',
                 'not','no','nor','so','yet','both','either','neither','each',
                 'every','all','any','few','more','most','other','some','such',
                 'than','too','very','just','about','up','out','if','then',
                 'what','which','who','whom','this','that','these','those',
                 'it','its','he','she','they','them','we','you','i','me',
                 'my','your','his','her','our','their','how','when','where',
                 'why','based','using','evidence','answer','according','knowledge',
                 'base','combine','summarize','key','point','detail','correct',
                 'ignore','unrelated','notes'}
    q_words = set(re.findall(r'\w+', question.lower())) - stopwords
    s_words = set(re.findall(r'\w+', sentence.lower())) - stopwords
    if not q_words: return 0.0
    return len(q_words & s_words) / len(q_words)

def generate_answer(question, retrieved_chunks, corpus_df, q_type='unknown'):
    """Generate extractive answer from retrieved chunks."""
    if not retrieved_chunks:
        return ""
    
    top_cid = retrieved_chunks[0][0]
    top_row = corpus_df[corpus_df['chunk_id'] == top_cid]
    if len(top_row) == 0:
        return ""
    top_text = top_row.iloc[0]['chunk_text']
    
    if q_type == 'multi_doc':
        # Combine from top-2 chunks
        parts = []
        for cid, score in retrieved_chunks[:2]:
            row = corpus_df[corpus_df['chunk_id'] == cid]
            if len(row) > 0:
                chunk_text = row.iloc[0]['chunk_text']
                sentences = extract_sentences(chunk_text)
                if sentences:
                    scored = [(s, score_sentence_relevance(s, question)) for s in sentences]
                    best = max(scored, key=lambda x: x[1])
                    parts.append(best[0])
                else:
                    parts.append(chunk_text)
        return ' '.join(parts) if parts else top_text
    
    elif q_type == 'sla_policy':
        sentences = extract_sentences(top_text)
        sla_kw = ['severity','acknowledge','mitigation','target','sla','response']
        best_sent, best_score = top_text, 0
        for sent in sentences:
            sent_lower = sent.lower()
            kw_count = sum(1 for kw in sla_kw if kw in sent_lower)
            rel = score_sentence_relevance(sent, question)
            combined = kw_count * 0.3 + rel * 0.7
            if combined > best_score:
                best_score = combined
                best_sent = sent
        return best_sent
    
    else:
        # For short chunks, return full text
        if len(top_text) <= 250:
            return top_text
        sentences = extract_sentences(top_text)
        if not sentences:
            return top_text
        scored = [(s, score_sentence_relevance(s, question)) for s in sentences]
        scored.sort(key=lambda x: x[1], reverse=True)
        return ' '.join(s for s, _ in scored[:2])

def select_citations(retrieved_chunks, q_type='unknown'):
    """Select chunks to cite based on question type."""
    if not retrieved_chunks: return []
    max_cite = 2 if q_type == 'multi_doc' else 1
    return [cid for cid, _ in retrieved_chunks[:max_cite]]


# ======================================================================
# 6. LOCAL EVALUATION FUNCTIONS
# ======================================================================

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def token_f1(prediction, reference):
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens: return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    shared = sum((pred_counts & ref_counts).values())
    prec = shared / sum(pred_counts.values()) if pred_counts else 0
    rec = shared / sum(ref_counts.values()) if ref_counts else 0
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)

def cit_precision(pred, gold):
    if not pred: return 0.0
    return len(set(pred) & set(gold)) / len(set(pred))

def cit_recall(pred, gold):
    if not gold: return 1.0
    return len(set(pred) & set(gold)) / len(set(gold))

def grounding(answer, cited_ids, corpus_df):
    if not answer or not cited_ids: return 0.0
    cited_text = ""
    for cid in cited_ids:
        row = corpus_df[corpus_df['chunk_id'] == cid]
        if len(row) > 0:
            cited_text += " " + row.iloc[0]['chunk_text']
    
    stopwords = {'the','a','an','is','are','was','were','be','been','have','has','had',
                 'do','does','did','will','would','to','of','in','for','on','with','at',
                 'by','from','and','but','or','not','no','so','if','then','it','its',
                 'this','that','these','those'}
    ans_tokens = set(tokenize(answer)) - stopwords
    cited_tokens = set(tokenize(cited_text)) - stopwords
    if not ans_tokens: return 1.0
    return len(ans_tokens & cited_tokens) / len(ans_tokens)

def over_cit_penalty(pred, gold):
    n_pred, n_gold = len(set(pred)), len(set(gold))
    excess = max(0, n_pred - n_gold)
    return max(0.0, 1.0 - excess / max(1, n_gold))


# ======================================================================
# 7. RUN FULL PIPELINE ON TRAIN
# ======================================================================

print("\n" + "="*60)
print("RUNNING FULL PIPELINE ON TRAIN SET")
print("="*60)

t0 = time.time()
train_results = []

for i, (idx, row) in enumerate(train.iterrows()):
    q_type = row['q_type']
    q_emb = train_question_embeddings[idx] if train_question_embeddings is not None else None
    
    retrieved = retrieve_enhanced(
        row['question'], q_emb, bm25_index, corpus_embeddings,
        chunk_ids_list, corpus, best_alpha, top_k=10
    )
    
    answer = generate_answer(row['question'], retrieved, corpus, q_type)
    citations = select_citations(retrieved, q_type)
    
    train_results.append({
        'question_id': row['question_id'],
        'answer': answer,
        'cited_chunk_ids': str(citations),
        'q_type': q_type
    })
    
    if (i+1) % 500 == 0:
        print(f"  Processed {i+1}/{len(train)} train questions...")

elapsed = time.time() - t0
print(f"  Done: {len(train_results)} questions in {elapsed:.1f}s ({len(train_results)/elapsed:.0f} q/s)")

train_pred_df = pd.DataFrame(train_results)

# ======================================================================
# 8. EVALUATE
# ======================================================================

print("\n" + "="*60)
print("TRAIN SET EVALUATION")
print("="*60)

component_scores = defaultdict(list)
scores_by_type = defaultdict(lambda: defaultdict(list))

for _, pred_row in train_pred_df.iterrows():
    qid = pred_row['question_id']
    gold_row = train[train['question_id'] == qid].iloc[0]
    
    pred_cit = ast.literal_eval(pred_row['cited_chunk_ids'])
    gold_cit = gold_row['parsed_gold']
    qt = pred_row['q_type']
    
    aq = token_f1(str(pred_row['answer']), str(gold_row['answer']))
    cp = cit_precision(pred_cit, gold_cit)
    cr = cit_recall(pred_cit, gold_cit)
    gc = grounding(str(pred_row['answer']), pred_cit, corpus)
    ocp = over_cit_penalty(pred_cit, gold_cit)
    
    component_scores['answer_quality'].append(aq)
    component_scores['citation_precision'].append(cp)
    component_scores['citation_recall'].append(cr)
    component_scores['grounding_consistency'].append(gc)
    component_scores['over_citation_penalty'].append(ocp)
    
    for key in component_scores:
        scores_by_type[qt][key].append(component_scores[key][-1])

print(f"\n  {'Component':<25s} {'Score':>8s}")
print(f"  {'-'*40}")
overall_scores = {}
for key in ['answer_quality','citation_precision','citation_recall',
            'grounding_consistency','over_citation_penalty']:
    mean_val = np.mean(component_scores[key])
    overall_scores[key] = mean_val
    status = '[OK]' if mean_val >= 0.5 else '[WARN]'
    print(f"  {key:<25s} {mean_val:>7.4f}  {status}")

overall = np.mean(list(overall_scores.values()))
print(f"  {'-'*40}")
print(f"  {'OVERALL':<25s} {overall:>7.4f}")

# Breakdown by question type
print(f"\n\n  BY QUESTION TYPE:")
print(f"  {'Type':<12s} {'Ans_Q':>8s} {'Cit_P':>8s} {'Cit_R':>8s} {'Ground':>8s} {'OC_Pen':>8s} {'Count':>7s}")
print(f"  {'-'*60}")
for qt in sorted(scores_by_type.keys()):
    vals = scores_by_type[qt]
    print(f"  {qt:<12s}", end="")
    for key in ['answer_quality','citation_precision','citation_recall',
                'grounding_consistency','over_citation_penalty']:
        print(f" {np.mean(vals[key]):>7.4f}", end="")
    print(f" {len(vals['answer_quality']):>7d}")

# Perfect citation stats
pred_cits_all = train_pred_df['cited_chunk_ids'].apply(ast.literal_eval)
gold_cits_all = train['parsed_gold']

exact_match = sum(1 for p, g in zip(pred_cits_all, gold_cits_all) if set(p) == set(g))
print(f"\n  Exact citation match: {exact_match}/{len(train)} ({exact_match/len(train):.1%})")


# ======================================================================
# 9. GENERATE TEST PREDICTIONS
# ======================================================================

print("\n" + "="*60)
print("GENERATING TEST PREDICTIONS")
print("="*60)

t0 = time.time()
test_results = []
test['q_type'] = test['question'].apply(classify_question)

for i, (idx, row) in enumerate(test.iterrows()):
    q_type = row['q_type']
    q_emb = test_question_embs[idx]
    
    retrieved = retrieve_enhanced(
        row['question'], q_emb, bm25_index, corpus_embeddings,
        chunk_ids_list, corpus, best_alpha, top_k=10
    )
    
    answer = generate_answer(row['question'], retrieved, corpus, q_type)
    citations = select_citations(retrieved, q_type)
    
    test_results.append({
        'question_id': row['question_id'],
        'answer': answer,
        'cited_chunk_ids': str(citations)
    })
    
    if (i+1) % 200 == 0:
        print(f"  Processed {i+1}/{len(test)} test questions...")

elapsed = time.time() - t0
print(f"  Done: {len(test_results)} questions in {elapsed:.1f}s")

test_pred_df = pd.DataFrame(test_results)

# ======================================================================
# 10. VALIDATE & WRITE SUBMISSION
# ======================================================================

print("\n" + "="*60)
print("SUBMISSION VALIDATION")
print("="*60)

# Check format matches sample
assert len(test_pred_df) == len(sample_sub), f"Row mismatch: {len(test_pred_df)} vs {len(sample_sub)}"
assert set(test_pred_df.columns) == set(sample_sub.columns), "Column mismatch"
assert list(test_pred_df['question_id']) == list(sample_sub['question_id']), "Question ID mismatch"

empty = test_pred_df['answer'].isna() | (test_pred_df['answer'] == '')
print(f"  [OK] Row count: {len(test_pred_df)}")
print(f"  [OK] Columns: {list(test_pred_df.columns)}")
print(f"  [OK] Question IDs match")
print(f"  {'[OK]' if not empty.any() else '[WARN]'} Empty answers: {empty.sum()}")

# Validate chunk IDs
valid_cids = set(corpus['chunk_id'])
invalid = 0
for _, row in test_pred_df.iterrows():
    for cid in ast.literal_eval(row['cited_chunk_ids']):
        if cid not in valid_cids:
            invalid += 1
print(f"  {'[OK]' if invalid==0 else '[WARN]'} Invalid citations: {invalid}")

# Citation count dist
cit_counts = test_pred_df['cited_chunk_ids'].apply(lambda x: len(ast.literal_eval(x)))
print(f"  Citation count dist: {cit_counts.value_counts().sort_index().to_dict()}")

# Write
test_pred_df.to_csv('submission.csv', index=False)
print(f"\n  >>> Submission written to: submission.csv")

# Show sample
print("\n  Sample predictions:")
for _, row in test_pred_df.head(3).iterrows():
    print(f"    {row['question_id']}: ans='{str(row['answer'])[:60]}...' cit={row['cited_chunk_ids']}")

print("\n" + "="*60)
print("PIPELINE COMPLETE!")
print("="*60)
print(f"  Train scores: {overall_scores}")
print(f"  Overall: {overall:.4f}")
print(f"  Best alpha: {best_alpha}")
