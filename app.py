import streamlit as st
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
import langchain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain

st.title("Question Answering Bot")

def get_llm():
    llm = ChatGroq(
        model_name = "llama-3.3-70b-versatile",
        temperature = 0.9,
        groq_api_key = "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"
    )

    return llm

def load_embedder():
    model = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")
    return model

def output (urls, question):
    try:
        loader = UnstructuredURLLoader(urls=urls)
        doc = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        text = text_splitter.split_documents(doc)

        vectorIndex = FAISS.from_documents(text, embeddings)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())

        output = chain({"question": question}, return_only_outputs=True)

        return output['answer']


    except Exception as e:
        return str(e)

question = st.text_input("Enter Your Question")
# st.sidebar.title("Enter Your Urls : ")

url1 = st.sidebar.text_input("Enter Your Url1 ")

url2 = st.sidebar.text_input("Enter Your Url2 ")
url3 = st.sidebar.text_input("Enter Your Url3 ")

button = st.sidebar.button("Give Answer")


if button:
    # result = output([url1, url2, url3], question)
    if(not question):
        st.write("Please enter a question.")
    else: 
        llm = get_llm()
        embeddings = load_embedder()

        urls = []
        if(url1):
            urls.append(url1)
        if(url2):
            urls.append(url2)
        if(url3):
            urls.append(url3)
        
        if(not urls):
            st.write("Please enter at least one URL.")
        else:
            result = output(urls, question)
            st.write(result)

