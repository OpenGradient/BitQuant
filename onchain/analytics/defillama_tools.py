from typing import Dict, Any, List
from langchain_core.tools import tool
from onchain.analytics.defillama_metrics import DefiLlamaMetrics

defillama = DefiLlamaMetrics()


@tool()
def show_defi_llama_protocol(protocol_slug: str) -> Dict[str, Any]:
    """Get detailed information about a specific DeFi protocol using its slug identifier.
    
    Args:
        protocol_slug (str): The unique identifier (slug) of the DeFi protocol
        
    Returns:
        Dict[str, Any]: Dictionary containing protocol details including TVL, tokens, etc.
    """
    return defillama.get_protocol(protocol_slug)


@tool()
def show_defi_llama_pool(pool_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific DeFi liquidity pool using its ID.
    
    Args:
        pool_id (str): The unique identifier of the DeFi pool
        
    Returns:
        Dict[str, Any]: Dictionary containing pool details including TVL, APY, etc.
    """
    return defillama.get_pool(pool_id)


@tool()
def show_defi_llama_global_tvl() -> Dict[str, Any]:
    """Get the current Total Value Locked (TVL) across all DeFi protocols.
    
    Returns:
        Dict[str, Any]: Dictionary containing the global TVL value
    """
    tvl = defillama.get_global_tvl()
    return {"global_tvl": tvl}


@tool()
def show_defi_llama_chain_tvl(chain: str) -> Dict[str, Any]:
    """Get the Total Value Locked (TVL) for a specific blockchain.
    
    Args:
        chain (str): The name of the blockchain (e.g., 'ethereum', 'bsc')
        
    Returns:
        Dict[str, Any]: Dictionary containing the chain name and its TVL
    """
    tvl = defillama.get_chain_tvl(chain)
    return {"chain": chain, "tvl": tvl}


@tool()
def show_defi_llama_top_pools(
    chain: str = None, min_tvl: float = 500000, max_apy: float = 1000
) -> List[Dict[str, Any]]:
    """Get a list of top DeFi pools ranked by APY with customizable filters.
    
    Args:
        chain (str, optional): Filter pools by specific blockchain. Defaults to None.
        min_tvl (float, optional): Minimum TVL threshold in USD. Defaults to 500000.
        max_apy (float, optional): Maximum APY threshold. Defaults to 1000.
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing pool details
    """
    return defillama.get_top_pools(chain, 10, min_tvl, max_apy)


@tool()
def show_defi_llama_historical_global_tvl(num_months: int = 12) -> Dict[str, Any]:
    """Get historical TVL data for all DeFi protocols over a specified time period.
    
    Args:
        num_months (int, optional): Number of months of historical data to retrieve. Defaults to 12.
        
    Returns:
        Dict[str, Any]: Dictionary containing historical TVL data points
    """
    return defillama.get_historical_global_tvl(num_months)


@tool()
def show_defi_llama_historical_chain_tvl(
    chain: str, num_months: int = 12
) -> Dict[str, Any]:
    """Get historical TVL data for a specific blockchain over a specified time period.
    
    Args:
        chain (str): The name of the blockchain
        num_months (int, optional): Number of months of historical data to retrieve. Defaults to 12.
        
    Returns:
        Dict[str, Any]: Dictionary containing historical TVL data points for the specified chain
    """
    return defillama.get_historical_chain_tvl(chain, num_months)
