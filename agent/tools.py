from typing import List, Tuple, Dict, Any, Type

from plugins.types import DepositAction, WithdrawAction, Action
from strategies.strategy import Strategy
from strategies.registry import STRATEGIES

from langgraph.graph.graph import RunnableConfig
from langchain_core.tools import BaseTool, tool, StructuredTool
from pydantic import BaseModel


@tool(response_format="content_and_artifact")
def recommend_deposit_to_pool(pool: str, token: str, amount: float) -> Tuple[str, List]:
    """Recommends depositing into the given pool"""
    action = DepositAction(pool=pool, tokens={token: amount}).model_dump()

    return "Recommendation recorded for user", [action]


@tool(response_format="content_and_artifact")
def recommend_withdraw_from_pool(
    pool: str, token: str, amount: float
) -> Tuple[str, List]:
    """Recommends withdrawal from the given pool"""
    action = WithdrawAction(pool=pool, tokens={token: amount}).model_dump()

    return "Recommendation recorded for user", [action]


def convert_strategy_to_tool(
    strategy: Strategy, args_schema: Type[BaseModel]
) -> StructuredTool:

    # Tool runnable
    def execute_strategy(options: Any, config: RunnableConfig) -> Tuple[str, List]:
        actions: List[Action] = strategy.allocate(
            tokens=config["configurable"]["tokens"],
            positions=config["configurable"]["positions"],
            available_pools=config["configurable"]["available_pools"],
            options=options,
        )

        return "Recorded allocations", [action.model_dump() for action in actions]

    return StructuredTool.from_function(
        func=execute_strategy,
        name=strategy.name(),
        description=strategy.description(),
        response_format="content_and_artifact",
        args_schema=args_schema,
    )


# Define the tools the agent can use
def create_agent_toolkit() -> List[BaseTool]:
    tools = [recommend_deposit_to_pool, recommend_withdraw_from_pool]

    for strategy, options in STRATEGIES:
        tools.append(convert_strategy_to_tool(strategy, options))

    return tools
