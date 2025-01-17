import os

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph

from opengradient.llm import langchain_adapter

from agent.prompts import get_agent_prompt
from agent.tools import create_agent_toolkit

MODEL = "Qwen/Qwen2.5-72B-Instruct"


def create_agent_executor() -> CompiledGraph:
    private_key = os.environ.get("PRIVATE_KEY")
    if not private_key:
        raise Exception("Must set PRIVATE_KEY env var")

    # Initialize LLM
    llm = langchain_adapter(private_key=private_key, model_cid=MODEL)

    # Build system prompt
    system_prompt = get_agent_prompt()

    # Create agent
    agent_executor = create_react_agent(
        model=llm, tools=create_agent_toolkit(), messages_modifier=system_prompt
    )

    return agent_executor
