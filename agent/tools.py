from typing import List

from langgraph.graph.graph import RunnableConfig
from langchain_core.tools import BaseTool, tool

from onchain.analytics.defillama_tools import (
    show_defi_llama_protocol,
    show_defi_llama_global_tvl,
    show_defi_llama_chain_tvl,
    show_defi_llama_top_pools,
    show_defi_llama_historical_global_tvl,
    show_defi_llama_historical_chain_tvl,
)
from api.api_types import Pool, WalletTokenHolding, Chain, PoolQuery
from onchain.analytics.analytics_tools import (
    max_drawdown_for_token,
    portfolio_volatility,
    analyze_volatility_trend,
    analyze_price_trend,
    compare_assets,
    analyze_wallet_portfolio,
)
from onchain.memecoins.trending import get_trending_tokens
from onchain.pools.protocol import ProtocolRegistry


@tool
def retrieve_solana_pools(
    tokens: List[str] = None,
    config: RunnableConfig = None,
) -> List[Pool]:
    """
    Retrieves Solana pools matching the specified criteria that the user can invest in.
    """
    configurable = config["configurable"]
    user_tokens: List[WalletTokenHolding] = configurable["tokens"]
    protocol_registry: ProtocolRegistry = configurable["protocol_registry"]

    # Create a query to filter pools
    query = PoolQuery(
        chain=Chain.SOLANA,  # Currently only supporting Solana
        tokens=tokens or [],
        user_tokens=user_tokens,
    )

    return protocol_registry.get_pools(query)


def create_investor_agent_toolkit() -> List[BaseTool]:
    return [
        retrieve_solana_pools,
    ]


def create_analytics_agent_toolkit() -> List[BaseTool]:
    return [
        show_defi_llama_global_tvl,
        show_defi_llama_historical_global_tvl,
        show_defi_llama_historical_chain_tvl,
        show_defi_llama_top_pools,
        analyze_price_trend,
        compare_assets,
        max_drawdown_for_token,
        portfolio_volatility,
        analyze_volatility_trend,
        analyze_wallet_portfolio,
        get_trending_tokens,
    ]
