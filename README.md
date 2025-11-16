📘 FAST-NUCES FYP Handbook Assistant (RAG System)

This project is an end-to-end Retrieval-Augmented Generation (RAG) assistant for the FAST-NUCES Final Year Project Handbook.
It allows students to ask natural-language questions and retrieves answers directly from the handbook.

Features

Sentence-transformers embedding model

FAISS semantic search index

Lexical re-ranking for relevance

Extractive answer generation (no hallucinations)

GPU support (if available)

Streamlit UI

Run the app
pip install -r requirements.txt
streamlit run app.py

Build the index (optional)

If you need to regenerate the index:

python ingest.py