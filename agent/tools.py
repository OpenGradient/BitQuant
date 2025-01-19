from typing import List, Tuple, Dict

from api_types.types import DepositAction, WithdrawAction, Action

from langchain_core.tools import BaseTool, tool


@tool(response_format="content_and_artifact")
def recommend_deposit_to_pool(pool: str, token: str, amount: float) -> Tuple[str, Dict]:
    """Recommends depositing into the given pool"""
    action = DepositAction(pool=pool, amount=amount, asset=token).model_dump()

    return "Recommendation recorded for user", action


@tool(response_format="content_and_artifact")
def recommend_withdraw_from_pool(
    pool: str, token: str, amount: float
) -> Tuple[str, Dict]:
    """Recommends withdrawal from the given pool"""
    action = WithdrawAction(pool=pool, amount=amount, asset=token).model_dump()

    return "Recommendation recorded for user", action


# Define the tools the agent can use
def create_agent_toolkit() -> List[BaseTool]:
    tools = [recommend_deposit_to_pool, recommend_withdraw_from_pool]

    return tools
