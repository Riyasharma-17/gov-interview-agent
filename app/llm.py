import os
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv(override=True)


def get_llm(tools: list[BaseTool] | None = None) -> BaseChatModel:
    models: list[BaseChatModel] = [
        ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0,
            reasoning_effort="low",
            max_retries=1,
        ),
        ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0,
            reasoning_effort="low",
            max_retries=1,
        ),
        ChatGroq(
            model="qwen/qwen3.8-27b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0,
            max_retries=1,
        ),
        ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0,
            max_retries=1,
        ),
    ]

    if tools:
        models = [m.bind_tools(tools) for m in models]

    return models[0].with_fallbacks(models[1:])