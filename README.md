# 📘 FAST-NUCES FYP Handbook Assistant (RAG System)

This repository contains an end-to-end **Retrieval-Augmented Generation (RAG)** assistant built specifically for the **FAST-NUCES Final Year Project Handbook**.

It allows students to ask **natural-language questions** and retrieves **accurate, citation-ready answers** directly from the handbook using semantic search, lexical ranking, and context-grounded generation.

---

## 🚀 Features

- 🔍 **Semantic Search** using Sentence-Transformers  
- ⚡ **FAISS Index** for fast vector retrieval  
- 📑 **Lexical Re-ranking** for improved relevance  
- 🧠 **Extractive, Non-Hallucinating Answer Generation** (answers strictly grounded in handbook text)  
- 🎛️ **GPU Acceleration** (automatically uses CUDA if available)  
- 🖥️ **Streamlit Frontend** for clean and interactive UI  
- 📘 **Complete RAG Pipeline** (ingestion → embedding → retrieval → answer generation)

---

## ▶️ How to Run the App

Install dependencies:

```bash
pip install -r requirements.txt

```bash
streamlit run app.py