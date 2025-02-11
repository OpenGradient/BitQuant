from typing import List, Tuple, Dict

from plugins.types import DepositAction, WithdrawAction
from strategies.strategy import Strategy
from strategies.registry import STRATEGIES

from langchain_core.tools import BaseTool, tool, StructuredTool


@tool(response_format="content_and_artifact")
def recommend_deposit_to_pool(pool: str, token: str, amount: float) -> Tuple[str, Dict]:
    """Recommends depositing into the given pool"""
    action = DepositAction(pool=pool, tokens={token: amount}).model_dump()

    return "Recommendation recorded for user", action


@tool(response_format="content_and_artifact")
def recommend_withdraw_from_pool(
    pool: str, token: str, amount: float
) -> Tuple[str, Dict]:
    """Recommends withdrawal from the given pool"""
    action = WithdrawAction(pool=pool, tokens={token: amount}).model_dump()

    return "Recommendation recorded for user", action


def convert_strategy_to_tool(strategy: Strategy) -> StructuredTool:
    return StructuredTool.from_function(
        func=strategy.allocate,
        name=strategy.name(),
        description=strategy.description(),
        response_format="content_and_artifact",
    )


# Define the tools the agent can use
def create_agent_toolkit() -> List[BaseTool]:
    tools = [recommend_deposit_to_pool, recommend_withdraw_from_pool]

    for s in STRATEGIES:
        tools.append(convert_strategy_to_tool(s))

    return tools
