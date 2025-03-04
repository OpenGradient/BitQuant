from typing import Dict, Any, List
from defi.stats import DefiMetrics
from defi.types import Chain, Pool, PoolQuery
from defi.defillama_tools import (
    show_defi_llama_protocols,
    show_defi_llama_protocol,
    show_defi_llama_global_tvl,
    show_defi_llama_chain_tvl,
    show_defi_llama_top_pools
)

class DeFiDataScientistAgent:
    """
    Specialized agent focused on DeFi data analysis and insights.
    Encapsulates all DeFi-related tools to prevent overloading the main agent.
    """
    
    def __init__(self):
        """
        Initialize the DeFi data scientist agent with its specialized tools.
        
        Sets up the underlying metrics engine and prepares the agent for DeFi analysis tasks.
        """
        self.name = "DeFi Data Scientist"
        self.metrics = DefiMetrics()
        
    def get_tools(self):
        """
        Return the set of DeFi-specific tools this agent can use.
        
        Returns:
            List of tools available to this agent for DeFi analysis tasks
        """
        return [
            # DefiLlama Tools
            show_defi_llama_protocols,
            show_defi_llama_protocol,
            show_defi_llama_global_tvl,
            show_defi_llama_chain_tvl,
            show_defi_llama_top_pools,
        ]
    
    def get_protocol_insights(self, protocol_slug: str) -> Dict[str, Any]:
        """
        Get comprehensive insights about a specific DeFi protocol.
        
        Args:
            protocol_slug: The unique identifier for the protocol (e.g., "aave-v3", "uniswap-v3")
            
        Returns:
            Dictionary containing detailed protocol data including TVL, description, chains, and more
        """
        return show_defi_llama_protocol.invoke({"protocol_slug": protocol_slug})
    
    def compare_pools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Compare top yield-generating pools across different protocols.
        
        Args:
            limit: Maximum number of pools to return, sorted by APY
            
        Returns:
            List of dictionaries containing pool data with APY, TVL, and other metrics
        """
        return show_defi_llama_top_pools.invoke({"limit": limit})
    
    def get_pools(self, chain: Chain, protocols: List[str] = None) -> List[Pool]:
        """
        Get pools from a specific blockchain and optionally filter by protocols.
        
        Args:
            chain: The blockchain to query pools from (e.g., Chain.ETHEREUM, Chain.SOLANA)
            protocols: Optional list of protocol names to filter the results
            
        Returns:
            List of Pool objects containing detailed information about each liquidity pool
        """
        self.metrics.refresh_metrics()
        return self.metrics.get_pools(
            PoolQuery(
                chain=chain,
                protocols=protocols,
            )
        )
        
    def get_global_tvl(self) -> Dict[str, Any]:
        """
        Get the current global Total Value Locked (TVL) across all DeFi protocols.
        
        Returns:
            Dictionary containing global TVL data including historical data and current totals
        """
        return show_defi_llama_global_tvl.invoke({})

    def get_chain_tvl(self, chain: str) -> Dict[str, Any]:
        """
        Get the Total Value Locked (TVL) for a specific blockchain.
        
        Args:
            chain: The blockchain to get TVL for (e.g., "ethereum", "solana", "arbitrum")
            
        Returns:
            Dictionary containing chain-specific TVL data including historical data and current totals
        """
        return show_defi_llama_chain_tvl.invoke({"chain": chain})