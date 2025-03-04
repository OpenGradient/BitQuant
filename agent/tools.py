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
    Retrieves historical price data for a cryptocurrency directly from Binance API.
    
    Args:
        pair: The trading pair (e.g., "BTCUSDT", "ETHUSDT")
        interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
        limit: Number of candlesticks to retrieve (max 1000)
        
    Returns:
        Dictionary containing the raw klines data from Binance API
    """
    # Enforce limit parameter
    limit = min(max(2, int(limit)), 1000)
    
    try:
        # Initialize client with Binance.US base URL
        client = Spot(base_url="https://api.binance.us")
        
        # Get klines data directly from the API
        klines = client.klines(symbol=pair.upper(), interval=interval, limit=limit)
        
        # Return the raw data with minimal processing
        return {
            "pair": pair,
            "interval": interval,
            "limit": limit,
            "data": klines,
            "columns": [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "count",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "ignore"
            ]
        }
        
    except Exception as e:
        error_msg = str(e)
        
        # Specific error handling for invalid trading pairs
        if "Invalid symbol" in error_msg or "Unknown symbol" in error_msg:
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
