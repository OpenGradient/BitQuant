import os

from openai import OpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph
from langchain_openai import ChatOpenAI

from agent.tools import create_investor_agent_toolkit, create_analytics_agent_toolkit


REASONING_MODEL = "deepseek/deepseek-chat-v3-0324"


def create_suggestions_executor() -> CompiledGraph:
    openai_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    agent_executor = create_react_agent(
        model=openai_model,
        tools=[],
    )

    return agent_executor


def create_investor_executor() -> CompiledGraph:
    openai_model = ChatOpenAI(
        model=REASONING_MODEL,
        temperature=0.0,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    agent_executor = create_react_agent(
        model=openai_model, tools=create_investor_agent_toolkit()
    )

    return agent_executor


def create_analytics_executor() -> CompiledGraph:
    openai_model = ChatOpenAI(
        model=REASONING_MODEL,
        temperature=0.0,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    analytics_executor = create_react_agent(
        model=openai_model,
        tools=create_analytics_agent_toolkit(),
    )

    return analytics_executor
