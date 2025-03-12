from typing import List, Tuple, Callable

from langgraph.graph.graph import RunnableConfig, CompiledGraph
from langchain_core.tools import BaseTool, tool, Tool
from pydantic import BaseModel, Field

from defi.analytics.defillama_tools import (
    show_defi_llama_protocols,
    show_defi_llama_protocol,
    show_defi_llama_global_tvl,
    show_defi_llama_chain_tvl,
    show_defi_llama_top_pools,
)
from api.api_types import Pool, WalletTokenHolding
from defi.analytics.binance_tools import (
    get_binance_price_history,
    analyze_price_trend,
    compare_assets,
)


@tool(response_format="content_and_artifact")
def show_pools(pool_ids: List[str], config: RunnableConfig) -> Tuple[str, List]:
    """Displays the pools to the user with the given IDs"""
    configurable = config["configurable"]
    available_pools: List[Pool] = configurable["available_pools"]

    pools = [pool.model_dump() for pool in available_pools if pool.id in pool_ids]

    return f"Showing pools to user: {pool_ids}", pools


def create_agent_toolkit(analytics_agent_func: Callable) -> List[BaseTool]:
    """Create tools that the main agent can use."""

    @tool
    def run_analytics(query: str, config: RunnableConfig) -> str:
        """
        Call the DeFi analytics agent to perform data analysis on protocols, tokens, or market conditions
        """
        configurable = config["configurable"]
        tokens: List[WalletTokenHolding] = configurable["tokens"]

        return analytics_agent_func(query=query, tokens=tokens, positions=[])

    return [
        show_pools,
        run_analytics,
    ]


def create_analytics_agent_toolkit() -> List[BaseTool]:
    return [
        show_defi_llama_protocols,
        show_defi_llama_protocol,
        show_defi_llama_global_tvl,
        show_defi_llama_chain_tvl,
        show_defi_llama_top_pools,
        get_binance_price_history,
        analyze_price_trend,
        compare_assets,
    ]
