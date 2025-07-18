import streamlit as st
from src.helper import load_pdf, split_text, download_hugging_face_embeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import Ollama  
from src.prompt import prompt_template

# Load and process documents (do this only once using st.cache_resource)
@st.cache_resource
def build_vectorstore():
    docs = load_pdf("data")  # Change "data" to your PDF folder path if needed
    chunks = split_text(docs)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = download_hugging_face_embeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

vectorstore = build_vectorstore()

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}


llm = Ollama(model="llama2") 

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

st.title("🧠 RAG QA Chatbot")

user_input = st.text_input("Ask a question:")
if st.button("Submit") and user_input:
    result = qa({"query": user_input})
    st.markdown(f"**Answer:** {result['result']}")
    # Optionally show source context
    with st.expander("Show context"):
        for doc in result['source_documents']:
            st.write(doc.page_content)
