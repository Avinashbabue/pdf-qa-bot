# 🧠 PDF Question Answering Bot using RAG and LLaMA

This is a simple chatbot application that allows users to ask questions based on the content of PDF files. It uses **Retrieval-Augmented Generation (RAG)** with a **local LLaMA model** and **FAISS** for semantic search.

---

## ✅ Features

- 📄 Upload and process PDF documents  
- 🧩 Chunk documents and embed them using `sentence-transformers` (`all-MiniLM-L6-v2`)  
- 🔍 Semantic similarity search using FAISS  
- 🤖 Local inference with LLaMA 2 via [Ollama](https://ollama.com/)  
- 💬 Streamlit UI for querying and displaying answers  
- 📚 Displays relevant context used to generate the answer  

---

## 📁 Project Structure

```text
.
├── app.py                 # Streamlit app (loads index, answers questions)
├── store_index.py         # (Optional) Script to manually build FAISS index
├── requirements.txt       # Python dependencies
├── Data/                  # Place your PDF files here
├── Model/                 # Place LLaMA model .bin file here
└── src/
    ├── helper.py          # PDF loading, chunking, embeddings
    └── prompt.py          # Prompt template
