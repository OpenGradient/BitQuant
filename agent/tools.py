from typing import List, Tuple, Dict, Any, Type, Optional, Union

from langgraph.graph.graph import RunnableConfig
from langchain_core.tools import BaseTool, tool, StructuredTool

from defi.types import Pool
import requests
import numpy as np
import pandas as pd


@tool(response_format="content_and_artifact")
def show_pools(pool_ids: List[str], config: RunnableConfig) -> Tuple[str, List]:
    """Displays the pools to the user with the given IDs"""
    configurable = config["configurable"]
    available_pools: List[Pool] = configurable["available_pools"]

    pools = [pool.model_dump() for pool in available_pools if pool.id in pool_ids]

    return f"Showing pools to user: {pool_ids}", pools


@tool()
def show_binance_price_history(pair: str, granularity: str = '1d', rows: int = 365) -> Dict[str, Any]:
    """
    Show historical price data from Binance for a specific trading pair.
    
    Args:
        pair: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
        granularity: Time interval for candles ('1m', '5m', '15m', '1h', '4h', '1d', '1w', etc.)
        rows: Number of candles to retrieve (max 1000)
    
    Returns:
        Dictionary with historical price data including close prices, percentage change, and summary statistics
    """
    try:
        # Get the candle data
        candles_df = get_binance_candles(pair, granularity, rows)
        
        # Get the close price series
        close_prices = single_close_price_series(pair, granularity, rows)
        
        # Calculate some useful metrics
        current_price = close_prices[-1]
        price_24h_ago = close_prices[-2] if len(close_prices) > 1 else None
        percent_change_24h = ((current_price - price_24h_ago) / price_24h_ago * 100) if price_24h_ago else None
        
        week_ago_idx = -7 if len(close_prices) >= 7 else 0
        price_week_ago = close_prices[week_ago_idx] if len(close_prices) > abs(week_ago_idx) else None
        percent_change_week = ((current_price - price_week_ago) / price_week_ago * 100) if price_week_ago else None
        
        month_ago_idx = -30 if len(close_prices) >= 30 else 0
        price_month_ago = close_prices[month_ago_idx] if len(close_prices) > abs(month_ago_idx) else None
        percent_change_month = ((current_price - price_month_ago) / price_month_ago * 100) if price_month_ago else None
        
        # Get some statistics
        stats = {
            "mean": float(np.mean(close_prices)),
            "median": float(np.median(close_prices)),
            "min": float(np.min(close_prices)),
            "max": float(np.max(close_prices)),
            "std_dev": float(np.std(close_prices)),
            "volatility": float(np.std(np.diff(close_prices) / close_prices[:-1]) * 100) if len(close_prices) > 1 else None
        }
        
        # Format the dates for the response
        formatted_dates = candles_df['timestamp'].dt.strftime('%Y-%m-%d').tolist() if 'timestamp' in candles_df.columns else []
        
        # Create a response object
        response = {
            "pair": pair,
            "granularity": granularity,
            "data_points": len(close_prices),
            "current_price": float(current_price),
            "percent_change_24h": float(percent_change_24h) if percent_change_24h is not None else None,
            "percent_change_week": float(percent_change_week) if percent_change_week is not None else None,
            "percent_change_month": float(percent_change_month) if percent_change_month is not None else None,
            "stats": stats,
            "close_prices": close_prices.tolist(),
            "dates": formatted_dates
        }
        
        return response
    except Exception as e:
        print(f"Warning: Error fetching Binance data for {pair}: {e}")
        return {"error": str(e), "pair": pair}

# Helper functions (add these outside of the tool functions)
def get_binance_candles(
    pair,
    granularity = '1d',
    rows = 365):

    url = 'https://data-api.binance.vision/api/v3/klines?'+\
        'symbol='+pair+'&interval='+granularity+'&limit='+str(rows+1)
    data = np.array(requests.get(url).json())[:-1]
    columns = [
        'open_timestamp',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'close_timestamp',
        'quote_volume',
        'n_trades',
        'taker_buy_base_volume',
        'taker_buy_quote_volume',
        'unused'
        ]
    df = pd.DataFrame(
        data = np.array(data,dtype=float),
        columns = columns
        )
    for col in ['open_timestamp','close_timestamp','n_trades']:
        df[col] = df[col].astype(int)
    timestamp = pd.to_datetime(df['open_timestamp'],unit='ms')
    df['timestamp'] = timestamp
    df = df.sort_values('timestamp')
    return df

def single_close_price_series(
        pair,
        granularity = '1d',
        rows = 365
        ):
    candles_data = get_binance_candles(
        pair = pair,
        granularity = granularity,
        rows = rows)
    close_series = np.array(candles_data['close'])
    return close_series

# Define the tools the agent can use
def create_agent_toolkit() -> List[BaseTool]:
    tools = [
        show_pools,
        show_binance_price_history
    ]

    return tools
