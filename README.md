# 🧠 PDF Question Answering Bot using RAG and LLaMA

This is a simple chatbot application that allows users to ask questions based on the content of uploaded PDF files. It uses **Retrieval-Augmented Generation (RAG)** with a **local LLaMA model** and **FAISS** for semantic search.

---

## ✅ Features

- Upload and process PDFs
- Use sentence-transformer embeddings for document chunking
- Create a FAISS vector store for efficient retrieval
- Use LLaMA 2 model locally for answering questions
- Simple Streamlit web interface

---

## 📁 Project Structure
─ app.py # Main Streamlit app
─ store_index.py # Preprocess and build FAISS index
─ requirements.txt # Python dependencies
─ Data/ # Place your PDFs here
─ Model/ # Place your LLaMA model file here
─ src/
  ─ helper.py # PDF loading, splitting, embeddings
  ─ prompt.py # Prompt template

