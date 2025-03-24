from typing import Dict, Any, List
from langchain_core.tools import tool
from defi.analytics.defillama_metrics import DefiLlamaMetrics

defillama = DefiLlamaMetrics()


@tool()
def show_defi_llama_protocol(protocol_slug: str) -> Dict[str, Any]:
    """Show details for a specific DeFi protocol by slug"""
    return defillama.get_protocol(protocol_slug)


@tool()
def show_defi_llama_global_tvl() -> Dict[str, Any]:
    """Show current global TVL across all DeFi protocols"""
    tvl = defillama.get_global_tvl()
    return {"global_tvl": tvl}


@tool()
def show_defi_llama_chain_tvl(chain: str) -> Dict[str, Any]:
    """Show TVL for a specific blockchain"""
    tvl = defillama.get_chain_tvl(chain)
    return {"chain": chain, "tvl": tvl}


@tool()
def show_defi_llama_top_pools(chain: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Show top DeFi pools ranked by APY for a specific chain (eg Solana, Ethereum, BSC)"""
    return defillama.get_top_pools(chain, limit)


@tool()
def show_defi_llama_pool(pool_id: str) -> Dict[str, Any]:
    """Show details for a specific DeFi pool by ID"""
    return defillama.get_pool(pool_id)
