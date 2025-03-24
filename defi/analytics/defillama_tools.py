from typing import Dict, Any, List
from langchain_core.tools import tool
from defi.analytics.defillama_metrics import DefiLlamaMetrics

defillama = DefiLlamaMetrics()


@tool()
def show_defi_llama_protocol(protocol_slug: str) -> Dict[str, Any]:
    """Show details for a specific DeFi protocol by slug"""
    return defillama.get_protocol(protocol_slug)


@tool()
def show_defi_llama_pool(pool_id: str) -> Dict[str, Any]:
    """Show details for a specific DeFi pool by ID"""
    return defillama.get_pool(pool_id)


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
def show_defi_llama_historical_global_tvl(months: int = None, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """Show historical TVL data for all DeFi protocols across all chains over time
    
    Args:
        months (int, optional): Number of months of history to return. Defaults to None.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to None.
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to None.
    """
    return defillama.get_historical_global_tvl(months, start_date, end_date)


@tool()
def show_defi_llama_historical_chain_tvl(chain: str, months: int = None, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """Show historical TVL data for a specific blockchain over time
    
    Args:
        chain (str): The target blockchain name.
        months (int, optional): Number of months of history to return. Defaults to None.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to None.
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to None.
    """
    return defillama.get_historical_chain_tvl(chain, months, start_date, end_date)
