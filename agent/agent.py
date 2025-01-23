import os
from typing import List

from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph

from opengradient.llm import langchain_adapter
from opengradient import LLM

from agent.tools import create_agent_toolkit

MODEL = LLM.QWEN_2_5_72B_INSTRUCT


def create_agent_executor() -> CompiledGraph:
    private_key = os.environ.get("PRIVATE_KEY")
    if not private_key:
        raise Exception("Must set PRIVATE_KEY env var")

    # Initialize LLM
    llm = langchain_adapter(private_key=private_key, model_cid=MODEL, max_tokens=400)

    # Create agent
    agent_executor = create_react_agent(
        model=llm,
        tools=create_agent_toolkit(),
    )

    return agent_executor
