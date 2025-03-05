from langchain_groq import ChatGroq
from phi.agent import Agent
from phi.tools import duckduckgo

llm = ChatGroq(
    model_name = "llama-3.3-70b-versatile",
    temperature = 0.9,
    groq_api_key = "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"
)