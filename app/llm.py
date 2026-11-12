import os
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()


def get_llm(tools: list[BaseTool] | None = None) -> BaseChatModel:
    primary = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_retries=1,
    )
    fallback = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        max_retries=1,
    )

    if tools:
        primary = primary.bind_tools(tools)
        fallback = fallback.bind_tools(tools)

    return primary.with_fallbacks([fallback])