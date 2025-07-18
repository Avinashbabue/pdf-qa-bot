# 🧠 PDF Question Answering Bot using RAG and LLaMA

This is a simple chatbot application that allows users to ask questions based on the content of PDF files. It uses **Retrieval-Augmented Generation (RAG)** with a **local LLaMA model** and **FAISS** for semantic search.

---

## ✅ Features

- Upload and process PDFs
- Chunk documents and embed them using `sentence-transformers` (`all-MiniLM-L6-v2`)
- Semantic similarity search with FAISS
- Local inference using LLaMA 2 (via `CTransformers`)
- Simple Streamlit web interface for querying
- Shows answer *and* source document snippets

---

## 📁 Project Structure

```text
.
├── app.py                 # Streamlit app (loads index, answers questions)
├── store_index.py         # Build and save FAISS index from PDFs
├── requirements.txt       # Python dependencies
├── Data/                  # Place your PDF files here
├── Model/                 # Place LLaMA model .bin file here
└── src/
    ├── helper.py          # PDF loading, splitting, embeddings
    └── prompt.py          # Prompt template
