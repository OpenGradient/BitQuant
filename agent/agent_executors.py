import os

from openai import OpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from agent.tools import create_investor_agent_toolkit, create_analytics_agent_toolkit
from onchain.tokens.metadata import TokenMetadataRepo
from server import config

##
# Subnet LLM Configuration
##

LOCAL_LLM_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
LOCAL_LLM_BASE_URL = "http://localhost:8000/v1"

##
# OpenRouter LLM Configuration
##

GOOGLE_GEMINI_25_MODEL = "google/gemini-2.5-pro-exp-03-25:free"  # Free
GOOGLE_GEMINI_20_FLASH_MODEL = (
    "gemini-2.0-flash"  # $0.1/M input tokens; $0.4/M output tokens
)
GOOGLE_GEMINI_FLASH_15_8B_MODEL = (
    "gemini-1.5-flash"  # $0.0375/M input tokens; $0.15/M output tokens
)
LLAMA_3_1_405B_MODEL = (
    "meta-llama/llama-3.1-405b-instruct"  # $0.8/M input tokens; $0.8/M output tokens
)
DEEPSEEK_CHAT_V3_MODEL = (
    "deepseek/deepseek-chat-v3-0324"  # $0.27/M input tokens; $1.1/M output tokens
)
GROK_MODEL = "x-ai/grok-2-1212"  # $2/M input tokens; $10/M output tokens


# Select model based on configuration
if not config.SUBNET_MODE:
    SUGGESTIONS_MODEL = GOOGLE_GEMINI_20_FLASH_MODEL
    ROUTING_MODEL = GOOGLE_GEMINI_FLASH_15_8B_MODEL
    REASONING_MODEL = GOOGLE_GEMINI_20_FLASH_MODEL
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"
    API_KEY = os.getenv("GEMINI_API_KEY")
else:
    SUGGESTIONS_MODEL = LOCAL_LLM_MODEL
    ROUTING_MODEL = LOCAL_LLM_MODEL
    REASONING_MODEL = LOCAL_LLM_MODEL
    BASE_URL = LOCAL_LLM_BASE_URL
    API_KEY = "dummy_key"


def create_routing_model() -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=ROUTING_MODEL,
        temperature=0.0,
        google_api_key=API_KEY,
        max_tokens=500,
    )


def create_suggestions_model() -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=SUGGESTIONS_MODEL,
        temperature=0.3,
        google_api_key=API_KEY,
        max_tokens=1000,
    )


def create_investor_executor() -> CompiledGraph:
    openai_model = ChatGoogleGenerativeAI(
        model=REASONING_MODEL,
        temperature=0.0,
        google_api_key=API_KEY,
        max_tokens=4096,
    )
    agent_executor = create_react_agent(
        model=openai_model, tools=create_investor_agent_toolkit()
    )

    return agent_executor


def create_analytics_executor(token_metadata_repo: TokenMetadataRepo) -> CompiledGraph:
    openai_model = ChatGoogleGenerativeAI(
        model=REASONING_MODEL,
        temperature=0.0,
        google_api_key=API_KEY,
        max_tokens=4096,
    )
    analytics_executor = create_react_agent(
        model=openai_model,
        tools=create_analytics_agent_toolkit(token_metadata_repo),
    )

    return analytics_executor
