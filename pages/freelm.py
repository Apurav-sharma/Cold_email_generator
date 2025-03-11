import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

def get_llm():
    return  ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.9,
        groq_api_key="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"
    )

def get_answer(question):
    llm = get_llm()

    output = llm.invoke(question)
    return output

st.title("OpenLM")

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] ul {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader("Ask anything")

question = st.text_input("Enter your question")
button = st.button("Get answer")

if button:
    if question.strip():
        with st.spinner("Wait for it..."):
            answer = get_answer(question)
            st.write("Generating answer...")
            st.success(answer.content)
            # st.write("Generated answer")
    else:
        st.warning("Please enter a question.")