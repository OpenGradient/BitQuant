import os

from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph
from opengradient.llm import langchain_adapter
from opengradient import LLM
from langchain_openai import ChatOpenAI

from agent.tools import create_agent_toolkit

MODEL = LLM.QWEN_2_5_72B_INSTRUCT
MAX_TOKENS = 1000


def create_agent_executor() -> CompiledGraph:
    private_key = os.environ.get("OG_PRIVATE_KEY")
    if not private_key:
        raise Exception("Must set OG_PRIVATE_KEY env var")

    # Initialize LLMs
    og_model = langchain_adapter(
        private_key=private_key, model_cid=MODEL, max_tokens=MAX_TOKENS
    )
    openai_model = ChatOpenAI(model="gpt-4o", temperature=0)

    # Create agent
    agent_executor = create_react_agent(
        model=openai_model,
        tools=create_agent_toolkit(),
    )

    return agent_executor
