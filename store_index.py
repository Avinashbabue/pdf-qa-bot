from src.helper import load_pdf, split_text, download_hugging_face_embeddings
from langchain.vectorstores import FAISS

# Load PDF documents from the specified folder
extracted_data = load_pdf('Data/')  # Make sure 'data/' matches your folder name

# Split the loaded documents into text chunks
text_chunks = split_text(extracted_data)

# Download HuggingFace embeddings model
embeddings = download_hugging_face_embeddings()

# Create FAISS vector store from your text chunks and embeddings
docsearch = FAISS.from_documents(text_chunks, embeddings)

# Save the FAISS index for later use
docsearch.save_local("faiss_index")
