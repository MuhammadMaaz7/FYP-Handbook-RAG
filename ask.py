import json
import os
import faiss
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import math
import torch
from datetime import datetime

OUT_DIR = "handbook_index"
CHUNKS_FILE = os.path.join(OUT_DIR, "chunks.jsonl")
FAISS_FILE = os.path.join(OUT_DIR, "index.faiss")
PROMPT_LOG_FILE = os.path.join(OUT_DIR, "prompt_log.txt")

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
SIM_THRESHOLD = 0.25

def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    return model

def embed_query(model, q):
    vec = model.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    return vec

def lexical_overlap_score(query, text):
    q_tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
    if not q_tokens:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for t in q_tokens if t in text_lower)
    return hits / len(q_tokens)

def boost_and_rerank(raw_scores, raw_idxs, chunks, query):
    results = []
    for score, idx in zip(raw_scores, raw_idxs):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        hint = chunk.get("section_hint", "") or ""
        prefix = chunk.get("text", "")[:400]
        s_hint = lexical_overlap_score(query, hint)
        s_prefix = lexical_overlap_score(query, prefix)
        boost = 1.0 + 0.6 * s_hint + 0.9 * s_prefix
        boosted_score = float(score) * boost
        results.append({"score": boosted_score, "orig_score": float(score), "chunk": chunk})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:TOP_K]

def extract_relevant_sentences_from_results(results, question, embed_model, max_sentences=4, prefer_top_chunk=True):
    all_sentences = []

    if prefer_top_chunk and results:
        top_chunk = results[0]["chunk"]
        sentences = [s.strip() for s in top_chunk["text"].split(".") if len(s.split()) > 5]
        for s in sentences:
            all_sentences.append((s, top_chunk["page"]))

    if not all_sentences:
        for r in results:
            text = r["chunk"]["text"]
            sentences = [s.strip() for s in text.split(".") if len(s.split()) > 5]
            for s in sentences:
                all_sentences.append((s, r["chunk"]["page"]))

    if not all_sentences:
        return None

    sent_texts = [s[0] for s in all_sentences]
    sent_vecs = embed_model.encode(sent_texts, convert_to_numpy=True, normalize_embeddings=True)
    q_vec = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]

    scores = np.dot(sent_vecs, q_vec)
    top_idx = np.argsort(scores)[::-1][:max_sentences]

    selected = []
    for idx in top_idx:
        s, page = all_sentences[idx]
        selected.append(f"{s.strip()}. (p. {page})")

    return " ".join(selected)

def retrieve(query, model, index, chunks, top_k=TOP_K):
    q_vec = embed_query(model, query)
    D, I = index.search(q_vec, top_k)
    raw_scores = D[0].tolist()
    raw_idxs = I[0].tolist()
    results = boost_and_rerank(raw_scores, raw_idxs, chunks, query)
    return results

PROMPT_TEMPLATE = """You are a handbook assistant. Answer ONLY from the context.
Cite page numbers like "(p. X)". If unsure, say you don't know.

Question: {user_question}

Context:
{context_text}
"""

def format_context(results):
    parts = []
    for r in results:
        chunk = r["chunk"]
        parts.append(f"(p. {chunk['page']}) {chunk['text']}")
    return "\n\n---\n\n".join(parts)

def log_prompt(query, prompt):
    # append to prompt_log.txt with timestamp
    with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] QUERY: {query}\n")
        f.write(f"[{datetime.now()}] PROMPT:\n{prompt}\n")
        f.write("-" * 80 + "\n")

def main():
    model = load_model()
    index = faiss.read_index(FAISS_FILE)
    chunks = load_chunks()

    print("Model and index loaded. Ready.\n")

    while True:
        q = input("Ask a question (or 'quit'): ").strip()
        if q.lower() in ["quit", "exit"]:
            break
        if not q:
            continue

        results = retrieve(q, model, index, chunks)

        if not results or results[0]["orig_score"] < SIM_THRESHOLD:
            print("I don't have that in the handbook.")
            continue

        print("\n--- Retrieved (after re-rank) ---")
        for r in results:
            c = r["chunk"]
            print(f"score={r['score']:.4f} (orig={r['orig_score']:.4f}) page={c['page']} id={c['chunk_id']} hint={c.get('section_hint','')[:40]}")

        answer = extract_relevant_sentences_from_results(results, q, model, prefer_top_chunk=True)
        if not answer:
            top_chunk = results[0]["chunk"]
            snippet = top_chunk["text"][:600].strip().replace("\n"," ")
            answer = f"{snippet} (p. {top_chunk['page']})"

        print("\nAnswer (paraphrased from handbook):\n")
        print(answer)
        print("\nSources (for verification):")
        print(format_context(results))

        # build and log prompt
        prompt = PROMPT_TEMPLATE.format(user_question=q, context_text=format_context(results))
        log_prompt(q, prompt)

        print("\n--- LLM Prompt logged to prompt_log.txt ---\n")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    main()