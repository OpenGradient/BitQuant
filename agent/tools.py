import logging
from typing import List

from langchain_core.tools import BaseTool, tool


@tool
def deposit_to_pool(pool_address: str) -> str:
    """Deposits into the given pool"""
    logging.info(f"Depositing into {pool_address}")

    return "Deposited"


@tool
def withdraw_from_pool(pool_address: str) -> str:
    """Withdraws from the given pool"""
    logging.info(f"Withdrawing from {pool_address}")

    return "Withdrawed"


# Define the tools the agent can use
def create_agent_toolkit() -> List[BaseTool]:
    tools = [deposit_to_pool, withdraw_from_pool]

    return tools
