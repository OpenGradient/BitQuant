import logging
from typing import List, Tuple, Dict

from api_types.types import DepositAction, WithdrawAction, Action

from langchain_core.tools import BaseTool, tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig


@tool(response_format="content_and_artifact")
def recommend_deposit_to_pool(pool_address: str) -> Tuple[str, Dict]:
    """Recommends depositing into the given pool"""
    action = DepositAction(poolId=pool_address, amount=100, asset="USDC").model_dump()

    return "Recommendation recorded for user", action


@tool(response_format="content_and_artifact")
def recommend_withdraw_from_pool(pool_address: str) -> Tuple[str, Dict]:
    """Recommends withdrawal from the given pool"""
    action = WithdrawAction(poolId=pool_address, amount=100, asset="USDC").model_dump()

    return "Recommendation recorded for user", action


# Define the tools the agent can use
def create_agent_toolkit() -> List[BaseTool]:
    tools = [recommend_deposit_to_pool, recommend_withdraw_from_pool]

    return tools
