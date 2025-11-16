# app.py
import streamlit as st
import json
import os
import faiss
from sentence_transformers import SentenceTransformer
from ask import retrieve, format_context, PROMPT_TEMPLATE, extract_relevant_sentences_from_results, load_model, load_chunks
import torch

OUT_DIR = "handbook_index"
CHUNKS_FILE = os.path.join(OUT_DIR, "chunks.jsonl")
FAISS_FILE = os.path.join(OUT_DIR, "index.faiss")

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
SIM_THRESHOLD = 0.25

# -------------------------------
# Load resources once
# -------------------------------
@st.cache_resource
def load_resources():
    # load model (sentence-transformers uses device option internally)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    index = faiss.read_index(FAISS_FILE)
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return model, index, chunks

model, index, chunks = load_resources()

# -------------------------------
# UI
# -------------------------------
st.title("📘 FAST-NUCES FYP Handbook Assistant (RAG System)")

with st.form("query_form"):
    q = st.text_input("Ask a question about the FYP Handbook")
    submitted = st.form_submit_button("Ask")

if submitted:
    if not q or not q.strip():
        st.warning("Please enter a question.")
    else:
        results = retrieve(q, model, index, chunks)

        if not results or results[0]["orig_score"] < SIM_THRESHOLD:
            st.error("I don't have that in the handbook.")
        else:
            # Build context and concise answer
            context_text = format_context(results)
            answer = extract_relevant_sentences_from_results(results, q, model, prefer_top_chunk=True)
            if not answer:
                # fallback
                top_chunk = results[0]["chunk"]
                answer = f"{top_chunk['text'][:600].strip()} (p. {top_chunk['page']})"

            st.subheader("Answer (paraphrased from handbook)")
            st.write(answer)

            with st.expander("Sources (page refs) — debug"):
                for r in results:
                    c = r["chunk"]
                    st.write(f"📄 Page {c['page']} — score={r['score']:.3f})")
                    st.write(c["section_hint"])
                    st.write(c["text"][:800] + "...\n")
