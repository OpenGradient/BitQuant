from typing import Dict, Any
import traceback
from binance.spot import Spot

def get_binance_price_history(pair: str = "BTCUSDT", interval: str = "1d", limit: int = 365) -> Dict[str, Any]:
    """
    Retrieves historical price data for a cryptocurrency directly from Binance API.
    
    Args:
        pair: The trading pair (e.g., "BTCUSDT", "ETHUSDT")
        interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
        limit: Number of candlesticks to retrieve (max 1000)
        
    Returns:
        Dictionary containing the raw klines data from Binance API
    """
    limit = min(max(2, int(limit)), 1000)
    
    try:
        client = Spot(base_url="https://api.binance.us")
        
        klines = client.klines(symbol=pair.upper(), interval=interval, limit=limit)
        
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