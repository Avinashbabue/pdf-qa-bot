# 🧠 PDF Question Answering Bot using RAG + LLaMA2

This is a lightweight and efficient **PDF-based Question Answering** system built using **Retrieval-Augmented Generation (RAG)**. It uses:

- Local **LLaMA 2** model via [Ollama](https://ollama.com/)
- **Hugging Face sentence embeddings** (`all-MiniLM-L6-v2`)
- **FAISS** for fast semantic search
- **Streamlit** as the front-end UI

---
## Assumptions

- All PDF documents are placed inside the `data/` folder.
- The LLaMA2 model (`llama2`) is already pulled using `ollama pull llama2`.
- The user has a compatible system with sufficient memory to run LLaMA2 locally via Ollama.
- The semantic search retrieves the top 2 most relevant chunks using FAISS.
- Each document is split into chunks of approximately 500 characters with 20 characters of overlap.
- Ollama is expected to be running as a local server before launching the Streamlit app.
- No external APIs (e.g., OpenAI, Cohere) are used — the system runs fully offline.
- PDFs are loaded and indexed at app startup. If new PDFs are added, the app must be restarted.
- The chatbot answers only one query at a time. There is no persistent memory or chat history tracking.


## ✅ Features

-  Upload and process local PDF documents
-  Generate semantic embeddings for document chunks
-  Perform fast retrieval using FAISS
-  Ask questions and get context-aware answers using LLaMA2
-  Caching for efficient repeated queries
- 🖥 Fully local & offline (no API keys required)

---

## 📁 Project Structure

```text
.
├── app.py                 # Main Streamlit app
├── store_index.py         # (Optional) Script to build FAISS index ahead of time
├── requirements.txt       # Python dependencies
├── data/                  # Place your PDF files here
├── Model/                 # Optional: place GGML/GGUF model files here if not using Ollama
├── src/
│   ├── helper.py          # PDF loading, text splitting, and embeddings
│   └── prompt.py          # Custom prompt template
└── README.md              # You're reading it!

