from typing import List, Optional

from langgraph.graph.graph import RunnableConfig
from langchain_core.tools import BaseTool, tool
from server.metrics import track_tool_usage

from onchain.tokens.metadata import TokenMetadataRepo, TokenMetadata

from onchain.analytics.defillama_tools import (
    show_defi_llama_top_pools,
    show_defi_llama_historical_global_tvl,
    show_defi_llama_historical_chain_tvl,
)
from api.api_types import Pool, WalletTokenHolding, Chain, PoolQuery
from onchain.analytics.analytics_tools import (
    max_drawdown_for_token,
    portfolio_volatility,
    analyze_price_trend,
    analyze_wallet_portfolio,
    get_coingecko_current_price,
)
from onchain.tokens.trending import (
    get_trending_tokens,
    evaluate_token_risk,
    get_top_token_holders,
)
from onchain.pools.protocol import ProtocolRegistry


@tool
@track_tool_usage("retrieve_solana_pools")
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

    pools = protocol_registry.get_pools(query)
    if len(pools) == 0:
        return "No pools found."

    return pools


def create_investor_agent_toolkit() -> List[BaseTool]:
    return [
        retrieve_solana_pools,
    ]


def create_analytics_agent_toolkit(
    token_metadata_repo: TokenMetadataRepo,
) -> List[BaseTool]:

    @tool
    @track_tool_usage("search_token")
    def search_token(
        token: str, chain: Optional[str] = None
    ) -> Optional[TokenMetadata]:
        """Search for a token by name or symbol. Returns metadata for the first token found."""
        return token_metadata_repo.search_token(token, chain)

    return [
        show_defi_llama_historical_global_tvl,
        show_defi_llama_historical_chain_tvl,
        show_defi_llama_top_pools,
        analyze_price_trend,
        max_drawdown_for_token,
        portfolio_volatility,
        analyze_wallet_portfolio,
        get_trending_tokens,
        get_coingecko_current_price,
        evaluate_token_risk,
        search_token,
        get_top_token_holders,
    ]
