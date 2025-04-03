import os

from openai import OpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph
from langchain_openai import ChatOpenAI

from agent.tools import create_investor_agent_toolkit, create_analytics_agent_toolkit
from onchain.tokens.metadata import TokenMetadataRepo

# Using local LLM and local base URL. 
# This assumes that your local LLM uses the OpenAI API and is hosted on port 8000.
LOCAL_LLM_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
LOCAL_LLM_BASE_URL = "http://localhost:8000/v1"

SUGGESTIONS_MODEL = LOCAL_LLM_MODEL
ROUTING_MODEL = LOCAL_LLM_MODEL
REASONING_MODEL = LOCAL_LLM_MODEL
BASE_URL = LOCAL_LLM_BASE_URL
API_KEY = "abc123"


def create_routing_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=ROUTING_MODEL,
        temperature=0.0,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        request_timeout=60,
        default_headers={"X-Title": "two-ligma"},
    )


def create_suggestions_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=SUGGESTIONS_MODEL,
        temperature=0.3,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        request_timeout=60,
        max_tokens=500,
        streaming=False,
        default_headers={"X-Title": "two-ligma"},
    )


def create_investor_executor() -> CompiledGraph:
    openai_model = ChatOpenAI(
        model=REASONING_MODEL,
        temperature=0.0,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        request_timeout=60,
        max_tokens=4096,
        streaming=False,
        default_headers={"X-Title": "two-ligma"},
    )
    agent_executor = create_react_agent(
        model=openai_model, tools=create_investor_agent_toolkit()
    )

    return agent_executor


def create_analytics_executor(token_metadata_repo: TokenMetadataRepo) -> CompiledGraph:
    openai_model = ChatOpenAI(
        model=REASONING_MODEL,
        temperature=0.0,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=API_KEY,
        request_timeout=60,
        max_tokens=4096,
        streaming=False,
        default_headers={"X-Title": "two-ligma"},
    )
    analytics_executor = create_react_agent(
        model=openai_model,
        tools=create_analytics_agent_toolkit(token_metadata_repo),
    )

    return analytics_executor
