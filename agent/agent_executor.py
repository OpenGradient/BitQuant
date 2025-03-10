import os

from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph
from opengradient.llm import langchain_adapter
from opengradient import LLM
from langchain_openai import ChatOpenAI

from agent.tools import create_agent_toolkit, create_analytics_agent_toolkit

MODEL = LLM.QWEN_2_5_72B_INSTRUCT
MAX_TOKENS = 1000


def create_suggestions_executor() -> CompiledGraph:
    openai_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # Create agent
    agent_executor = create_react_agent(model=openai_model, tools=[])

    return agent_executor


def create_agent_executor() -> CompiledGraph:
    openai_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    agent_executor = create_react_agent(
        model=openai_model,
        tools=create_agent_toolkit(),
    )

    return agent_executor

def create_analytics_executor() -> CompiledGraph:
    openai_model = ChatOpenAI(model="o3-mini")

    analytics_executor = create_react_agent(
        model=openai_model,
        tools=create_analytics_agent_toolkit(),  
    )

    return analytics_executor
