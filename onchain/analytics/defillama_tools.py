from typing import Dict, Any, List
from langchain_core.tools import tool
from onchain.analytics.defillama_metrics import DefiLlamaMetrics
from agent.telemetry import track_tool_usage

defillama = DefiLlamaMetrics()


@tool()
@track_tool_usage("show_defi_llama_protocol")
def show_defi_llama_protocol(protocol_slug: str) -> Dict[str, Any]:
    """
    Get details for a DeFi protocol by slug
    """
    return defillama.get_protocol(protocol_slug)


@tool()
@track_tool_usage("show_defi_llama_pool")
def show_defi_llama_pool(pool_id: str) -> Dict[str, Any]:
    """
    Get details for a DeFi pool by ID
    """
    return defillama.get_pool(pool_id)


@tool()
@track_tool_usage("show_defi_llama_global_tvl")
def show_defi_llama_global_tvl() -> Dict[str, Any]:
    """
    Get current global TVL across all DeFi protocols
    """
    tvl = defillama.get_global_tvl()
    return {"global_tvl": tvl}


@tool()
@track_tool_usage("show_defi_llama_chain_tvl")
def show_defi_llama_chain_tvl(chain: str) -> Dict[str, Any]:
    """
    Get TVL for a specific chain
    """
    tvl = defillama.get_chain_tvl(chain)
    return {"chain": chain, "tvl": tvl}


@tool()
@track_tool_usage("show_defi_llama_top_pools")
def show_defi_llama_top_pools(
    chain: str = None, min_tvl: float = 500000, max_apy: float = 1000
) -> List[Dict[str, Any]]:
    """
    Get top DeFi pools ranked by APY with TVL filters
    """
    return defillama.get_top_pools(chain, 10, min_tvl, max_apy)


@tool()
@track_tool_usage("show_defi_llama_historical_global_tvl")
def show_defi_llama_historical_global_tvl(num_months: int = 12) -> Dict[str, Any]:
    """
    Get historical TVL data for all DeFi protocols
    """
    return defillama.get_historical_global_tvl(num_months)


@tool()
@track_tool_usage("show_defi_llama_historical_chain_tvl")
def show_defi_llama_historical_chain_tvl(
    chain: str, num_months: int = 12
) -> Dict[str, Any]:
    """
    Get historical TVL data for a specific blockchain
    """
    return defillama.get_historical_chain_tvl(chain, num_months)
