from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Extract Data from pdf
def load_pdf(pdf_folder):
    loader = DirectoryLoader(pdf_folder, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

# text splitter
def split_text(documents, chunk_size=500, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# embeddings download
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
