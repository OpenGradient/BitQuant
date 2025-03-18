from typing import List, Tuple, Callable

from langgraph.graph.graph import RunnableConfig, CompiledGraph
from langchain_core.tools import BaseTool, tool, Tool
from pydantic import BaseModel, Field
import traceback

from defi.analytics.defillama_tools import (
    show_defi_llama_protocols,
    show_defi_llama_protocol,
    show_defi_llama_global_tvl,
    show_defi_llama_chain_tvl,
    show_defi_llama_top_pools,
)
from api.api_types import Pool, WalletTokenHolding, Chain, PoolQuery
from defi.analytics.binance_tools import (
    get_binance_price_history,
    analyze_price_trend,
    compare_assets,
)
from defi.analytics.financial_analytics_tools import (
    max_drawdown,
    portfolio_value,
    portfolio_volatility,
    portfolio_summary,
    analyze_volatility_trend,
)
from defi.pools.protocol import ProtocolRegistry


@tool(response_format="content_and_artifact")
def show_pools(pool_ids: List[str], config: RunnableConfig) -> Tuple[str, List]:
    """Displays the pools to the user with the given IDs"""
    configurable = config["configurable"]
    available_pools: List[Pool] = configurable["available_pools"]

    pools = [pool.model_dump() for pool in available_pools if pool.id in pool_ids]

    return f"Showing pools to user: {pool_ids}", pools


@tool()
def retrieve_pools(
    tokens: List[str] = None,
    protocols: List[str] = None,
    is_stablecoin: bool = None,
    impermanent_loss_risk: bool = None,
    config: RunnableConfig = None,
) -> List[Pool]:
    """Retrieves pools matching the specified criteria for internal agent analysis.
    This tool is for the agent to analyze pools without displaying them to the user.

    Args:
        tokens: List of token addresses to filter by
        protocols: List of protocol names to filter by
        is_stablecoin: Whether to filter for stablecoin pools
        impermanent_loss_risk: Whether to filter for pools with impermanent loss risk
        config: The runnable config containing available pools

    Returns:
        List of matching pools
    """
    configurable = config["configurable"]
    protocol_registry: ProtocolRegistry = configurable["protocol_registry"]

    # Create a query to filter pools
    query = PoolQuery(
        chain=Chain.SOLANA,  # Currently only supporting Solana
        tokens=tokens or [],
        protocols=protocols or [],
        isStableCoin=is_stablecoin,
        impermanentLossRisk=impermanent_loss_risk,
    )

    # Use ProtocolRegistry to get matching pools
    return protocol_registry.get_pools(query)


def create_agent_toolkit() -> List[BaseTool]:
    """Create tools that the main agent can use."""
    return [
        show_pools,
        retrieve_pools,
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
        max_drawdown,
        portfolio_value,
        portfolio_volatility,
        portfolio_summary,
        analyze_volatility_trend,
    ]
