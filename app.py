import streamlit as st
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain

st.title("Question Answering Bot")

@st.cache_resource
def get_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.9,
        groq_api_key="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"
    )

@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")

def process_documents(urls):
    try:
        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()
        if not docs:
            return None, "No content found in the URLs."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_text = text_splitter.split_documents(docs)

        embeddings = load_embedder()
        vector_index = FAISS.from_documents(split_text, embeddings)

        return vector_index, None
    except Exception as e:
        return None, str(e)

def get_answer(question):
    if "vector_index" not in st.session_state:
        return "No data loaded yet. Please enter URLs first."

    try:
        llm = get_llm()
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=st.session_state.vector_index.as_retriever())
        output = chain({"question": question}, return_only_outputs=True)
        return output.get("answer", "No answer found.")
    except Exception as e:
        return str(e)

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "last_fetched_urls" not in st.session_state:
    st.session_state.last_fetched_urls = []

st.sidebar.title("Enter Your URLs")
url1 = st.sidebar.text_input("Enter URL 1", key="url1")
url2 = st.sidebar.text_input("Enter URL 2", key="url2")
url3 = st.sidebar.text_input("Enter URL 3", key="url3")

urls = [url.strip() for url in [url1, url2, url3] if url.strip()]
if urls and st.session_state.last_fetched_urls != urls:
    with st.spinner("Fetching data..."):
        vector_index, error = process_documents(urls)
        if vector_index:
            st.session_state.vector_index = vector_index
            st.session_state.last_fetched_urls = urls
            st.sidebar.success("Data fetched successfully!")
        else:
            st.sidebar.error(error)

question = st.text_input("Enter Your Question")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        st.write("Generating answer...")
        result = get_answer(question)
        st.success(result)
