from typing import Dict, Any, List
from langchain_core.tools import tool
from defi.stats import DefiMetrics

defi_metrics = DefiMetrics()

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
    return defi_metrics.get_global_tvl()

@tool()
def show_defi_llama_chain_tvl(chain: str) -> Dict[str, Any]:
    """Show TVL for a specific blockchain"""
    return defi_metrics.get_chain_tvl(chain)

@tool()
def show_defi_llama_top_pools(limit: int = 10) -> List[Dict[str, Any]]:
    """Show top DeFi pools ranked by APY"""
    return defi_metrics.get_top_pools(limit)