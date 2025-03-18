from typing import Dict, Any, List
from langchain_core.tools import tool
from defi.analytics.defillama_source import DefiLlamaMetrics

defi_metrics = DefiLlamaMetrics()


@tool()
def show_defi_llama_protocols() -> List[Dict[str, Any]]:
    """Show a list of all DeFi protocols"""
    return defi_metrics.get_protocols()


@tool()
def show_defi_llama_protocol(protocol_slug: str) -> Dict[str, Any]:
    """Show details for a specific DeFi protocol by slug"""
    return defi_metrics.get_protocol(protocol_slug)


@tool()
def show_defi_llama_global_tvl() -> Dict[str, Any]:
    """Show current global TVL across all DeFi protocols"""
    tvl = defi_metrics.get_global_tvl()
    return {"global_tvl": tvl}


@tool()
def show_defi_llama_chain_tvl(chain: str) -> Dict[str, Any]:
    """Show TVL for a specific blockchain"""
    tvl = defi_metrics.get_chain_tvl(chain)
    return {"chain": chain, "tvl": tvl}


@tool()
def show_defi_llama_top_pools(limit: int = 10) -> List[Dict[str, Any]]:
    """Show top DeFi pools ranked by APY"""
    return defi_metrics.get_top_pools(limit)


@tool()
def show_defi_llama_pool(pool_id: str) -> Dict[str, Any]:
    """Show details for a specific DeFi pool by ID"""
    return defi_metrics.get_pool(pool_id)
