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
def show_defi_llama_top_pools(chain: str = None, limit: int = 10, min_tvl: float = 500000, max_apy: float = 1000) -> List[Dict[str, Any]]:
    """Show top DeFi pools ranked by APY with configurable TVL threshold.
    
    Args:
        chain (str, optional): Filter by blockchain (e.g., Solana, Ethereum, BSC). 
                              If None, returns pools from all chains.
        limit (int, optional): Maximum number of pools to return. Defaults to 10.
        min_tvl (float, optional): Minimum TVL threshold in USD. Defaults to 500000 ($500k).
                                  Higher TVL generally indicates lower risk.
        max_apy (float, optional): Maximum APY threshold in percentage. Defaults to 1000 (1000%).
                            Lower values filter out potentially unreliable high-yield pools.
    
    Returns pools with realistic APY values (under 200%) to filter out extremely high 
    but potentially unreliable options.
    """
    return defillama.get_top_pools(chain, limit, min_tvl, max_apy)


@tool()
def show_defi_llama_historical_global_tvl(num_months: int = 3) -> Dict[str, Any]:
    """Show historical TVL data for all DeFi protocols across all chains
    
    Args:
        num_months (int, optional): Number of months of history to include. Defaults to 3 months.
    """
    return defillama.get_historical_global_tvl(num_months)


@tool()
def show_defi_llama_historical_chain_tvl(chain: str, num_months: int = 3) -> Dict[str, Any]:
    """Show historical TVL data for a specific blockchain
    
    Args:
        chain (str): The target blockchain name.
        num_months (int, optional): Number of months of history to include. Defaults to 3 months.
    """
    return defillama.get_historical_chain_tvl(chain, num_months)
