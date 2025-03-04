from typing import Dict, Any, List, Optional
from defi.stats import DefiMetrics
from defi.types import Chain, Pool, PoolQuery
from defi.defillama_tools import (
    show_defi_llama_protocols,
    show_defi_llama_protocol,
    show_defi_llama_global_tvl,
    show_defi_llama_chain_tvl,
    show_defi_llama_top_pools
)
from defi.binance_tools import get_binance_price_history

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
    
    # New Binance-related methods
    
    def get_price_history(self, pair: str = "BTCUSDT", interval: str = "1d", limit: int = 365) -> Dict[str, Any]:
        """
        Retrieve historical price data for a cryptocurrency trading pair.
        
        Args:
            pair: The trading pair (e.g., "BTCUSDT", "ETHUSDT")
            interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
            limit: Number of candlesticks to retrieve (max 1000)
            
        Returns:
            Dictionary containing the historical price data.
        """
        return get_binance_price_history(pair, interval, limit)
    
    def analyze_price_trend(self, pair: str = "BTCUSDT", interval: str = "1d", limit: int = 30) -> Dict[str, Any]:
        """
        Analyze price trends for a cryptocurrency pair.
        
        Args:
            pair: The trading pair (e.g., "BTCUSDT", "ETHUSDT")
            interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
            limit: Number of candlesticks to analyze
            
        Returns:
            Dictionary containing trend analysis including moving averages,
            volatility metrics, and basic technical indicators.
        """
        # Get the price history first
        price_data = self.get_price_history(pair, interval, limit)
        
        # If there was an error, return the error
        if "error" in price_data:
            return price_data
        
        # Extract relevant data for analysis
        # This is a simplified analysis - in a real implementation you would 
        # calculate moving averages, RSI, volatility, etc.
        raw_data = price_data["data"]
        
        # Extract close prices as floats
        close_prices = [float(candle[4]) for candle in raw_data]
        
        # Simple trend analysis
        if len(close_prices) < 2:
            trend = "Not enough data"
        else:
            recent_change = ((close_prices[-1] / close_prices[0]) - 1) * 100
            trend = "Upward" if recent_change > 0 else "Downward"
        
        return {
            "pair": price_data["pair"],
            "period": f"{interval} x {limit}",
            "trend": trend,
            "change_percent": round(recent_change, 2) if len(close_prices) >= 2 else None,
            "current_price": close_prices[-1] if close_prices else None,
            "price_range": {
                "min": min(close_prices) if close_prices else None,
                "max": max(close_prices) if close_prices else None
            },
            "raw_data": price_data
        }
    
    def compare_assets(self, pairs: List[str], interval: str = "1d", limit: int = 30) -> Dict[str, Any]:
        """
        Compare performance of multiple cryptocurrency assets.
        
        Args:
            pairs: List of trading pairs to compare (e.g., ["BTCUSDT", "ETHUSDT"])
            interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
            limit: Number of candlesticks to analyze
            
        Returns:
            Dictionary with comparative performance metrics.
        """
        results = {}
        
        for pair in pairs:
            analysis = self.analyze_price_trend(pair, interval, limit)
            
            # Skip if there was an error
            if "error" in analysis:
                results[pair] = {"error": analysis["error"]}
                continue
                
            results[pair] = {
                "trend": analysis["trend"],
                "change_percent": analysis["change_percent"],
                "current_price": analysis["current_price"]
            }
        
        # Determine which asset performed best
        valid_pairs = {p: data for p, data in results.items() 
                      if "error" not in data and data.get("change_percent") is not None}
        
        if valid_pairs:
            best_performer = max(valid_pairs.items(), 
                                key=lambda x: x[1].get("change_percent", float("-inf")))
            worst_performer = min(valid_pairs.items(), 
                                 key=lambda x: x[1].get("change_percent", float("inf")))
            
            # Add comparative analysis
            return {
                "assets": results,
                "best_performer": {
                    "pair": best_performer[0],
                    "change_percent": best_performer[1]["change_percent"]
                },
                "worst_performer": {
                    "pair": worst_performer[0],
                    "change_percent": worst_performer[1]["change_percent"]
                },
                "period": f"{interval} x {limit}"
            }
        
        return {
            "assets": results,
            "error": "No valid assets for comparison"
        }