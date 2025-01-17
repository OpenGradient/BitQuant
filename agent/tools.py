import logging
from typing import List

from langchain_core.tools import BaseTool, tool

@tool
def deposit_to_pool(pool_address: str):
    """Deposits into the given pool"""
    logging.info(f"Depositing into {pool_address}")

    return "Deposited"

@tool
def withdraw_from_pool(pool_address: str):
    """Withdraws from the given pool"""
    logging.info(f"Withdrawing from {pool_address}")

    return "Withdrawed"

@tool
def swap_token(symbol_a: str, symbol_b):
    """Swaps token A into token B"""
    logging.info(f"Swapping {symbol_a} into {symbol_b}")

    return "Swapped"


# Define the tools the agent can use
def create_agent_toolkit() -> List[BaseTool]:
    tools = [deposit_to_pool, withdraw_from_pool, swap_token]

    return tools
