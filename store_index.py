from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import FAISS

extracted_data = load_pdf('Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Create FAISS vector store from your text chunks and embeddings
docsearch = FAISS.from_documents(text_chunks, embeddings)
