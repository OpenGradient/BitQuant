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


@tool
def retrieve_pools(
    tokens: List[str] = None,
    config: RunnableConfig = None,
) -> List[Pool]:
    """
    Retrieves pools matching the specified criteria.
    """
    configurable = config["configurable"]
    user_tokens: List[WalletTokenHolding] = configurable["tokens"]
    protocol_registry: ProtocolRegistry = configurable["protocol_registry"]

    # Create a query to filter pools
    query = PoolQuery(
        chain=Chain.SOLANA,  # Currently only supporting Solana
        tokens=tokens or [],
        user_tokens=user_tokens,  # Pass user's actual token holdings
    )

    # Use ProtocolRegistry to get matching pools
    return protocol_registry.get_pools(query)


def create_agent_toolkit() -> List[BaseTool]:
    """Create tools that the main agent can use."""
    return [
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
