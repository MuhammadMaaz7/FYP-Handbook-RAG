# ingest.py
import pdfplumber
import json
import os
import uuid
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

# -------------------------------
# Paths & settings
# -------------------------------
PDF_PATH = "FYP-Handbook-2023.pdf"   # change if needed
OUT_DIR = "handbook_index"
os.makedirs(OUT_DIR, exist_ok=True)

CHUNKS_FILE = os.path.join(OUT_DIR, "chunks.jsonl")
FAISS_FILE = os.path.join(OUT_DIR, "index.faiss")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking settings
WORDS_PER_CHUNK = 300
OVERLAP_PCT = 0.4  # increase overlap for better context
OVERLAP_WORDS = int(WORDS_PER_CHUNK * OVERLAP_PCT)
MIN_WORDS_PER_PAGE = 50  # skip empty/trivial pages

# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_pages(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if len(text.split()) >= MIN_WORDS_PER_PAGE:  # skip empty pages
                pages.append({"page_num": i, "text": text})
    return pages

# -------------------------------
# Extract page section heading (first meaningful line)
# -------------------------------
def get_section_hint(text):
    for line in text.splitlines():
        s = line.strip()
        if 3 <= len(s.split()) <= 12 and len(s) < 120:
            return s
    return "No Heading"

# -------------------------------
# Chunking page text
# -------------------------------
def chunk_page(text, page_num):
    words = text.split()
    chunks = []
    if not words:
        return chunks

    i = 0
    idx = 0
    while i < len(words):
        chunk = " ".join(words[i:i + WORDS_PER_CHUNK])
        chunk_id = f"{page_num}_{idx}_{uuid.uuid4().hex[:6]}"
        chunks.append({
            "chunk_id": chunk_id,
            "page": page_num,
            "text": chunk
        })
        idx += 1
        if i + WORDS_PER_CHUNK >= len(words):
            break
        i = i + WORDS_PER_CHUNK - OVERLAP_WORDS
    return chunks

# -------------------------------
# Build all chunks with section hints
# -------------------------------
def build_chunks(pages):
    all_chunks = []
    for p in pages:
        section_hint = get_section_hint(p["text"])
        page_chunks = chunk_page(p["text"], p["page_num"])
        for c in page_chunks:
            c["section_hint"] = section_hint
            all_chunks.append(c)
    return all_chunks

# -------------------------------
# Embed chunks
# -------------------------------
def embed_chunks(model, chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)  # normalize for cosine similarity
    return embeddings

# -------------------------------
# Save chunks to JSONL
# -------------------------------
def save_chunks(chunks):
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

# -------------------------------
# Main pipeline
# -------------------------------
def main():
    print("Extracting PDF pages...")
    pages = extract_pages(PDF_PATH)
    print(f"Total meaningful pages: {len(pages)}")

    print("Building chunks...")
    chunks = build_chunks(pages)
    print(f"Total chunks created: {len(chunks)}")
    save_chunks(chunks)

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Embedding chunks (GPU if available)...")
    embeddings = embed_chunks(model, chunks)

    dim = embeddings.shape[1]

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_FILE)
    print("Done! FAISS index and chunks saved.")

if __name__ == "__main__":
    main()
