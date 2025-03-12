import os

from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph
from opengradient.llm import langchain_adapter
from opengradient import LLM
from langchain_openai import ChatOpenAI

from agent.tools import create_agent_toolkit, create_analytics_agent_toolkit
from agent.og_alphasense_tools import create_alphasense_toolkit, initialize_og_sdk
from agent.prompts import get_agent_prompt, get_suggestions_prompt, get_analytics_prompt, get_alphasense_prompt


def create_suggestions_executor() -> CompiledGraph:
    openai_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
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


def create_alphasense_executor() -> CompiledGraph:
    """
    Create and return the AlphaSense agent executor.
    
    This function initializes the OpenGradient SDK with the private key from
    environment variables, and creates a ReactAgent with access to the
    AlphaSense tools (read_workflow and run_model).
    
    Returns:
        CompiledGraph: The AlphaSense agent executor
    """
    # Initialize the OpenGradient SDK
    initialize_og_sdk()
    
    # Create the language model
    openai_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    
    # Create the agent with AlphaSense tools
    alphasense_executor = create_react_agent(
        model=openai_model,
        tools=create_alphasense_toolkit(),
    )
    
    return alphasense_executor
