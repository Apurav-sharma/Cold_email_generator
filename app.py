import streamlit as st
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
import langchain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("Question Answering Bot")

llm = ChatGroq(
    model_name = "llama-3.3-70b-versatile",
    temperature = 0.9,
    groq_api_key = "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"
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def output (urls, question):
    try:
        loader = UnstructuredURLLoader(urls=urls)
        doc = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        text = text_splitter.split_documents(doc)


    except Exception as e:
        return str(e)





question = st.text_input("Enter Your Question")
# st.sidebar.title("Enter Your Urls : ")

url1 = st.sidebar.text_input("Enter Your Url1 ")

url2 = st.sidebar.text_input("Enter Your Url2 ")
url3 = st.sidebar.text_input("Enter Your Url3 ")
