from typing import List

from langgraph.graph.graph import RunnableConfig
from langchain_core.tools import BaseTool, tool

from defi.analytics.defillama_tools import (
    show_defi_llama_protocol,
    show_defi_llama_global_tvl,
    show_defi_llama_chain_tvl,
    show_defi_llama_top_pools,
    show_defi_llama_historical_global_tvl,
    show_defi_llama_historical_chain_tvl,
)
from api.api_types import Pool, WalletTokenHolding, Chain, PoolQuery
from defi.analytics.analytics_tools import (
    max_drawdown_for_token,
    portfolio_volatility,
    analyze_volatility_trend,
    analyze_price_trend,
    compare_assets,
    analyze_wallet_portfolio,
    get_coinmarketcap_price_history,
    analyze_price_trend_cmc,
    compare_assets_cmc,
    max_drawdown_for_token_cmc,
    analyze_volatility_trend_cmc,
)
from defi.pools.protocol import ProtocolRegistry


@tool
def retrieve_solana_pools(
    tokens: List[str] = None,
    config: RunnableConfig = None,
) -> List[Pool]:
    configurable = config["configurable"]
    user_tokens: List[WalletTokenHolding] = configurable["tokens"]
    protocol_registry: ProtocolRegistry = configurable["protocol_registry"]

    query = PoolQuery(
        chain=Chain.SOLANA,
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
        show_defi_llama_protocol,
        show_defi_llama_global_tvl,
        show_defi_llama_historical_global_tvl,
        show_defi_llama_historical_chain_tvl,
        show_defi_llama_top_pools,
        get_coinmarketcap_price_history,
        analyze_price_trend_cmc,
        compare_assets_cmc,
        max_drawdown_for_token_cmc,
        analyze_volatility_trend_cmc,
        analyze_wallet_portfolio,
        portfolio_volatility,
    ]
