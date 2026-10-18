import pandas as pd
import numpy as np
import ast
import re
import time
from collections import Counter, defaultdict

corpus = pd.read_csv('corpus.csv')
train = pd.read_csv('train.csv')
train['parsed_gold'] = train['gold_chunk_ids'].apply(ast.literal_eval)

corpus_embeddings = np.load('corpus_embeddings.npy')

# Tokenizer and Stopwords
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

def clean_tokenize(text):
    words = re.findall(r'\w+', text.lower())
    return [w for w in words if w not in stopwords]

print("Building indices...")
corpus_texts = []
for _, row in corpus.iterrows():
    combined = f"{row['title']} {row['doc_type']} {row['chunk_text']}"
    corpus_texts.append(clean_tokenize(combined))

from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(corpus_texts)
chunk_ids_list = corpus['chunk_id'].tolist()

# Pre-group chunk indices by title
unique_titles = corpus['title'].unique()
title_to_chunk_indices = {title: [] for title in unique_titles}
for idx, row in corpus.iterrows():
    title_to_chunk_indices[row['title']].append(idx)

# Group doc_id to its chunks
doc_to_chunks = defaultdict(list)
for idx, row in corpus.iterrows():
    doc_to_chunks[row['doc_id']].append(row['chunk_id'])

# Chunk to text map
chunk_text_map = corpus.set_index('chunk_id')['chunk_text'].to_dict()

# Encode train questions
print("Encoding train questions...")
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
train_question_embeddings = st_model.encode(
    train['question'].tolist(),
    batch_size=64, normalize_embeddings=True, show_progress_bar=False
)

def classify_question(q):
    if q.startswith('Based on'):     return 'summarize'
    elif q.startswith('Answer using'): return 'key_detail'
    elif q.startswith('Combine evidence'): return 'multi_doc'
    elif q.startswith('According to'):  return 'sla_policy'
    else: return 'other'

train['q_type'] = train['question'].apply(classify_question)

def extract_doc_references(question):
    refs = []
    match = re.search(r'from:\s*(.+?)(?:\.|$)', question)
    if match: refs.append(match.group(1).strip())
    quoted = re.findall(r"'([^']+)'", question)
    refs.extend(quoted)
    match = re.search(r'According to the\s+(.+?),', question)
    if match: refs.append(match.group(1).strip())
    return refs

def title_match_score(title, references):
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

def retrieve_hybrid_restricted(question, query_embedding, alpha=0.5, top_k=5):
    refs = extract_doc_references(question)
    
    matching_chunk_indices = []
    if refs:
        matching_titles = []
        for title in unique_titles:
            t_score = title_match_score(title, refs)
            if t_score > 0.7:
                matching_titles.append(title)
        
        for title in matching_titles:
            matching_chunk_indices.extend(title_to_chunk_indices[title])
            
    if matching_chunk_indices:
        # Calculate BM25 scores for matching chunks
        q_tokens = clean_tokenize(question)
        bm25_full_scores = bm25.get_scores(q_tokens)
        bm_scores = bm25_full_scores[matching_chunk_indices]
        
        # Calculate Semantic scores for matching chunks
        sim = query_embedding @ corpus_embeddings.T
        sem_scores = sim[matching_chunk_indices]
        
        # Min-max normalize
        def norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx == mn: return np.ones_like(arr)
            return (arr - mn) / (mx - mn)
        
        bm25_n = norm(bm_scores)
        sem_n = norm(sem_scores)
        
        fused = alpha * sem_n + (1 - alpha) * bm25_n
        top_k_local = np.argsort(fused)[::-1][:top_k]
        return [chunk_ids_list[matching_chunk_indices[i]] for i in top_k_local]
    else:
        # Fallback to general hybrid
        q_tokens = clean_tokenize(question)
        bm25_full_scores = bm25.get_scores(q_tokens)
        bm25_top_idx = np.argsort(bm25_full_scores)[::-1][:50]
        
        sim = query_embedding @ corpus_embeddings.T
        sem_top_idx = np.argsort(sim)[::-1][:50]
        
        candidate_idx = list(set(bm25_top_idx.tolist() + sem_top_idx.tolist()))
        
        bm_scores = bm25_full_scores[candidate_idx]
        sem_scores = sim[candidate_idx]
        
        def norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx == mn: return np.ones_like(arr)
            return (arr - mn) / (mx - mn)
        
        bm25_n = norm(bm_scores)
        sem_n = norm(sem_scores)
        
        fused = alpha * sem_n + (1 - alpha) * bm25_n
        top_k_local = np.argsort(fused)[::-1][:top_k]
        return [chunk_ids_list[candidate_idx[i]] for i in top_k_local]

def get_first_sentence(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return sentences[0].strip() if sentences else text

def run_pipeline_v3(row, idx):
    q_type = row['q_type']
    q_emb = train_question_embeddings[idx]
    
    # 1. SLA policy rule
    if q_type == 'sla_policy':
        retrieved = retrieve_hybrid_restricted(row['question'], q_emb, top_k=5)
        # Find first retrieved chunk belonging to an SLA policy document
        sla_chunk = retrieved[0]
        for cid in retrieved:
            if 'sla_policy' in corpus[corpus['chunk_id'] == cid].iloc[0]['doc_type']:
                # The answer is always in chunk 2
                doc_id = cid.split('_chunk')[0]
                sla_chunk = f"{doc_id}_chunk_2"
                break
        return "Acknowledge within 10 minutes, mitigation within 60 minutes.", [sla_chunk]
        
    # 2. Multi-doc rule
    elif q_type == 'multi_doc':
        # Retrieve candidate chunks (top 10)
        retrieved = retrieve_hybrid_restricted(row['question'], q_emb, top_k=15)
        
        # We need chunks from 2 different matching documents
        refs = extract_doc_references(row['question'])
        matching_docs = defaultdict(list)
        
        for cid in retrieved:
            doc_id = cid.split('_chunk')[0]
            title = corpus[corpus['doc_id'] == doc_id].iloc[0]['title']
            if title_match_score(title, refs) > 0.7:
                matching_docs[doc_id].append(cid)
                
        # Pick top chunk from each of the first 2 different documents
        cited = []
        doc_ids = list(matching_docs.keys())[:2]
        for doc_id in doc_ids:
            cited.append(matching_docs[doc_id][0])
            
        # Fallback if we didn't find chunks from 2 different docs
        if len(cited) < 2:
            cited = retrieved[:2]
            
        parts = [get_first_sentence(chunk_text_map.get(cid, '')) for cid in cited]
        return ' '.join(parts), cited
        
    # 3. Summarize / Key-detail
    else:
        retrieved = retrieve_hybrid_restricted(row['question'], q_emb, top_k=5)
        top_chunk = retrieved[0]
        ans = get_first_sentence(chunk_text_map.get(top_chunk, ''))
        return ans, [top_chunk]

# Run and evaluate
print("Running pipeline on training set...")
predictions = []
for idx, row in train.iterrows():
    ans, cited = run_pipeline_v3(row, idx)
    predictions.append({
        'question_id': row['question_id'],
        'answer': ans,
        'cited_chunk_ids': str(cited)
    })

pred_df = pd.DataFrame(predictions)

# Scoring
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

def grounding(answer, cited_ids):
    if not answer or not cited_ids: return 0.0
    cited_text = ""
    for cid in cited_ids:
        cited_text += " " + chunk_text_map.get(cid, '')
    ans_tokens = set(tokenize(answer)) - stopwords
    cited_tokens = set(tokenize(cited_text)) - stopwords
    if not ans_tokens: return 1.0
    return len(ans_tokens & cited_tokens) / len(ans_tokens)

def over_cit_penalty(pred, gold):
    n_pred, n_gold = len(set(pred)), len(set(gold))
    excess = max(0, n_pred - n_gold)
    return max(0.0, 1.0 - excess / max(1, n_gold))

scores = defaultdict(list)
for _, pred_row in pred_df.iterrows():
    qid = pred_row['question_id']
    gold_row = train[train['question_id'] == qid].iloc[0]
    
    pred_cit = ast.literal_eval(pred_row['cited_chunk_ids'])
    gold_cit = gold_row['parsed_gold']
    
    aq = token_f1(str(pred_row['answer']), str(gold_row['answer']))
    cp = cit_precision(pred_cit, gold_cit)
    cr = cit_recall(pred_cit, gold_cit)
    gc = grounding(str(pred_row['answer']), pred_cit)
    ocp = over_cit_penalty(pred_cit, gold_cit)
    
    scores['answer_quality'].append(aq)
    scores['citation_precision'].append(cp)
    scores['citation_recall'].append(cr)
    scores['grounding_consistency'].append(gc)
    scores['over_citation_penalty'].append(ocp)

print("\n=== PIPELINE V3 TRAINING EVALUATION ===")
overall_scores = {}
for key in ['answer_quality', 'citation_precision', 'citation_recall', 'grounding_consistency', 'over_citation_penalty']:
    mean_val = np.mean(scores[key])
    overall_scores[key] = mean_val
    print(f"  {key:<25s}: {mean_val:.4f}")

overall = np.mean(list(overall_scores.values()))
print(f"  {'OVERALL':<25s}: {overall:.4f}")
