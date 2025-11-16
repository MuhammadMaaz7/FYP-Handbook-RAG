# ask.py
import json
import os
import faiss
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import math
import torch

OUT_DIR = "handbook_index"
CHUNKS_FILE = os.path.join(OUT_DIR, "chunks.jsonl")
FAISS_FILE = os.path.join(OUT_DIR, "index.faiss")

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
SIM_THRESHOLD = 0.25  # tune if needed

# -------------------------------
# Helpers
# -------------------------------
def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def load_model():
    # let sentence-transformers pick GPU if available; ensure torch is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    return model

def embed_query(model, q):
    vec = model.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    return vec

def splitter_sentences(text):
    # lightweight sentence splitter
    sentences = re.split(r'(?<=[\.\?\!])\s+', text.replace('\n', ' '))
    return [s.strip() for s in sentences if s.strip()]

def lexical_overlap_score(query, text):
    # simple token overlap: count query tokens present in text / query length
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
        # lexical overlap with section_hint and beginning of chunk text
        hint = chunk.get("section_hint", "") or ""
        prefix = chunk.get("text", "")[:400]  # early text
        s_hint = lexical_overlap_score(query, hint)
        s_prefix = lexical_overlap_score(query, prefix)
        # combined boost factor (no hardcoded keywords)
        boost = 1.0 + 0.6 * s_hint + 0.9 * s_prefix
        boosted_score = float(score) * boost
        results.append({"score": boosted_score, "orig_score": float(score), "chunk": chunk})
    # sort descending by boosted score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:TOP_K]

def extract_relevant_sentences_from_results(results, question, embed_model, max_sentences=4):
    """
    Extracts the most semantically relevant sentences from retrieved chunks.
    No hardcoding. Pure semantic relevance.
    """
    all_sentences = []
    for r in results:
        text = r["chunk"]["text"]
        sentences = [s.strip() for s in text.split(".") if len(s.split()) > 5]
        for s in sentences:
            all_sentences.append((s, r["chunk"]["page"]))

    if not all_sentences:
        return None

    # embed question & candidate sentences
    sent_texts = [s[0] for s in all_sentences]
    sent_vecs = embed_model.encode(sent_texts, convert_to_numpy=True, normalize_embeddings=True)
    q_vec = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]

    # cosine scores
    scores = np.dot(sent_vecs, q_vec)

    # pick best N sentences
    top_idx = np.argsort(scores)[::-1][:max_sentences]

    selected = []
    for idx in top_idx:
        s, page = all_sentences[idx]
        selected.append(f"{s.strip()}. (p. {page})")

    return " ".join(selected)


# -------------------------------
# Retrieval
# -------------------------------
def retrieve(query, model, index, chunks, top_k=TOP_K):
    q_vec = embed_query(model, query)
    D, I = index.search(q_vec, top_k)
    raw_scores = D[0].tolist()
    raw_idxs = I[0].tolist()
    # boost & rerank using lexical overlap w/ section_hint and early text
    results = boost_and_rerank(raw_scores, raw_idxs, chunks, query)
    return results

# -------------------------------
# Prompt template (for validation / LLM)
# -------------------------------
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

# -------------------------------
# CLI app
# -------------------------------
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

        # show debug retrieval
        print("\n--- Retrieved (after re-rank) ---")
        for r in results:
            c = r["chunk"]
            print(f"score={r['score']:.4f} (orig={r['orig_score']:.4f}) page={c['page']} id={c['chunk_id']} hint={c.get('section_hint','')[:40]}")

        # create concise paraphrased answer using top results
        answer = extract_relevant_sentences_from_results(results, q, model)
        if not answer:
            # fallback: show top chunk text excerpt
            top_chunk = results[0]["chunk"]
            snippet = top_chunk["text"][:600].strip().replace("\n"," ")
            answer = f"{snippet} (p. {top_chunk['page']})"

        print("\nAnswer (paraphrased from handbook):\n")
        print(answer)
        print("\nSources (for verification):")
        print(format_context(results))
        # Also print LLM prompt for validation if needed
        prompt = PROMPT_TEMPLATE.format(user_question=q, context_text=format_context(results))
        print("\n--- LLM Prompt (copy/paste) ---\n")
        print(prompt)

if __name__ == "__main__":
    main()
