import os
from typing import List

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.chat_agent_executor import AgentState

from opengradient.llm import langchain_adapter

from agent.tools import create_agent_toolkit
from api_types.types import DepositAction, WithdrawAction, Action

MODEL = "Qwen/Qwen2.5-72B-Instruct"


def create_agent_executor() -> CompiledGraph:
    private_key = os.environ.get("PRIVATE_KEY")
    if not private_key:
        raise Exception("Must set PRIVATE_KEY env var")

    # Initialize LLM
    llm = langchain_adapter(private_key=private_key, model_cid=MODEL)

    # Create agent
    agent_executor = create_react_agent(
        model=llm,
        tools=create_agent_toolkit(),
    )

    return agent_executor
