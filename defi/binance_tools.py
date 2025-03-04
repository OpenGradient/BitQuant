from typing import Dict, Any, List
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

def analyze_price_trend(pair: str = "BTCUSDT", interval: str = "1d", limit: int = 30) -> Dict[str, Any]:
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
    price_data = get_binance_price_history(pair, interval, limit)
    
    # If there was an error, return the error
    if "error" in price_data:
        return price_data
    
    # Extract relevant data for analysis
    raw_data = price_data["data"]
    
    # Extract close prices as floats
    close_prices = [float(candle[4]) for candle in raw_data]
    
    # Simple trend analysis
    if len(close_prices) < 2:
        trend = "Not enough data"
        recent_change = 0
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

def compare_assets(pairs: List[str], interval: str = "1d", limit: int = 30) -> Dict[str, Any]:
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
        analysis = analyze_price_trend(pair, interval, limit)
        
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