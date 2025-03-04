from typing import List, Tuple, Dict, Any, Type, Optional, Union

from langgraph.graph.graph import RunnableConfig
from langchain_core.tools import BaseTool, tool, StructuredTool

from defi.types import Pool
import numpy as np
import pandas as pd
from binance.spot import Spot
from datetime import datetime
import traceback


@tool(response_format="content_and_artifact")
def show_pools(pool_ids: List[str], config: RunnableConfig) -> Tuple[str, List]:
    """Displays the pools to the user with the given IDs"""
    configurable = config["configurable"]
    available_pools: List[Pool] = configurable["available_pools"]

    pools = [pool.model_dump() for pool in available_pools if pool.id in pool_ids]

    return f"Showing pools to user: {pool_ids}", pools

@tool()
def show_binance_price_history(pair: str = "BTCUSDT", interval: str = "1d", limit: int = 365) -> Dict[str, Any]:
    """
    Retrieves historical price data for a cryptocurrency from Binance.
    
    Args:
        pair: The trading pair (e.g., "BTCUSDT", "ETHUSDT")
        interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
        limit: Number of candlesticks to retrieve (max 1000)
        
    Returns:
        Dictionary containing price history data, current price, and price metrics
    """
    # Enforce limit parameter - properly limit the number of data points
    limit = min(max(2, int(limit)), 1000)
    
    try:
        # Initialize client with Binance.US base URL - no API key needed for public endpoints
        client = Spot(base_url="https://api.binance.us")
        
        # Get historical klines using the Binance connector
        data = client.klines(symbol=pair.upper(), interval=interval, limit=limit)
        
        # Check if the data is valid
        if not isinstance(data, list) or not data:
            raise ValueError(f"Invalid response from Binance API for {pair}")
        
        # Extract close prices and timestamps
        dates = []
        close_prices = []
        
        for candle in data:
            # Binance returns timestamp as the first element (in milliseconds)
            timestamp = int(candle[0]) / 1000
            # Close price is the 5th element (index 4)
            close_price = float(candle[4])
            
            date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            dates.append(date_str)
            close_prices.append(close_price)
        
        # Calculate current price and percentage changes
        current_price = close_prices[-1]
        
        # Calculate percentage changes if we have enough data points
        percent_change_24h = ((close_prices[-1] / close_prices[-2]) - 1) * 100 if len(close_prices) >= 2 else None
        percent_change_week = ((close_prices[-1] / close_prices[-7]) - 1) * 100 if len(close_prices) >= 7 else None
        percent_change_month = ((close_prices[-1] / close_prices[-30]) - 1) * 100 if len(close_prices) >= 30 else None
        
        # Calculate statistics
        stats = {
            "mean": float(np.mean(close_prices)),
            "median": float(np.median(close_prices)),
            "min": float(np.min(close_prices)),
            "max": float(np.max(close_prices)),
            "std_dev": float(np.std(close_prices))
        }
        
        # Calculate volatility
        stats["volatility"] = float(stats["std_dev"] / stats["mean"]) if stats["mean"] > 0 else 0
        
        # Return the formatted response
        return {
            "pair": pair,
            "interval": interval,
            "current_price": current_price,
            "percent_change_24h": percent_change_24h,
            "percent_change_week": percent_change_week,
            "percent_change_month": percent_change_month,
            "stats": stats,
            "dates": dates,
            "close_prices": close_prices,
            "data_points": len(close_prices)
        }
    
    except Exception as e:
        error_msg = str(e)
        
        # Specific error handling for invalid trading pairs
        if "Invalid symbol" in error_msg or "Unknown symbol" in error_msg or "too many indices for array" in error_msg:
            raise ValueError(f"Invalid trading pair: {pair}")
        
        # Return error information for other exceptions
        return {
            "error": f"Error fetching Binance data for {pair}: {error_msg}",
            "traceback": traceback.format_exc()
        }

# Define the tools the agent can use
def create_agent_toolkit() -> List[BaseTool]:
    tools = [
        show_pools,
        show_binance_price_history
    ]

    return tools
