"""
AlphaSense agent that provides tools for interacting with OpenGradient AlphaSense capabilities.
"""
from typing import Dict, Any, List, Optional, Type

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent

from agent.og_alphasense_tools import (
    create_alphasense_toolkit,
    initialize_og_sdk
)
from agent.llm_provider import get_llm


class AlphaSenseAgent:
    """
    An agent specialized in handling OpenGradient AlphaSense operations.
    
    This agent provides tools for interacting with AlphaSense workflows and models.
    It automatically initializes the OpenGradient SDK on creation.
    """
    def __init__(self):
        # Initialize the OpenGradient SDK
        try:
            initialize_og_sdk()
        except Exception as e:
            print(f"Warning: Failed to initialize OpenGradient SDK: {e}")
            print("Make sure to set the OG_EMAIL, OG_PASSWORD, and OG_PRIVATE_KEY environment variables.")
        
        # Get tools
        self.tools = create_alphasense_toolkit()
        
        # Get LLM
        self.llm = get_llm()
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI assistant that helps users interact with OpenGradient AlphaSense capabilities.
You can help users read workflow results and run machine learning models using AlphaSense.
Use the tools available to you to help the user accomplish their tasks.
Always provide clear explanations of what you're doing and the results you find."""),
            ("human", "{input}"),
        ])
        
        # Create agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def run(self, input_text: str) -> str:
        """
        Run the AlphaSense agent with the given input.
        
        Args:
            input_text: The user's input text
            
        Returns:
            The agent's response
        """
        return self.agent_executor.invoke({"input": input_text})["output"] 