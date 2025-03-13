from typing import Dict, Any, List
import numpy as np
import traceback
from langchain_core.tools import tool
from defi.analytics.binance_tools import get_binance_price_history


@tool()
def max_drawdown(
        symbol: str,
        interval: str = "1d",
        limit: int = 90) -> Dict[str, Any]:
    '''
    Calculates the maximum drawdown for a cryptocurrency using Binance price data
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
        limit: Number of candlesticks to retrieve (max 1000)
        
    Returns:
        Dictionary containing the maximum drawdown value
    '''
    try:
        # Get price data from Binance
        price_data = get_binance_price_history.invoke({"pair": symbol, "interval": interval, "limit": limit})
        
        if "error" in price_data:
            return {
                "error": f"Failed to fetch price data for {symbol}: {price_data['error']}"
            }
        
        # Extract closing prices
        price_series = [float(candle[4]) for candle in price_data["data"]]
        
        # Calculate max drawdown
        price_series = np.array(price_series)
        rolling_max = np.maximum.accumulate(price_series)
        drawdowns = (rolling_max - price_series) / rolling_max
        max_dd = float(drawdowns.max())
        
        return {
            "symbol": symbol,
            "period": f"{interval} x {limit}",
            "max_drawdown": max_dd,
            "max_drawdown_percent": f"{max_dd * 100:.2f}%",
            "explanation": "Maximum drawdown represents the largest percentage drop from a peak to a subsequent trough"
        }
    except Exception as e:
        return {
            "error": f"Error calculating maximum drawdown: {str(e)}",
            "traceback": traceback.format_exc()
        }


@tool()
def portfolio_value(
        symbols: List[str],
        holding_qty: List[float],
        interval: str = "1d",
        limit: int = 90) -> Dict[str, Any]:
    '''
    Creates the time series of portfolio total value using Binance price data
    
    Args:
        symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
        holding_qty: List with quantities of each asset held
        interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
        limit: Number of candlesticks to retrieve (max 1000)
        
    Returns:
        Dictionary containing time series of portfolio total value
    '''
    try:
        if len(symbols) != len(holding_qty):
            return {
                "error": "Number of symbols must match number of holdings"
            }
        
        # Fetch price data for each asset
        all_price_data = []
        
        for symbol in symbols:
            price_data = get_binance_price_history.invoke({"pair": symbol, "interval": interval, "limit": limit})
            
            if "error" in price_data:
                return {
                    "error": f"Failed to fetch price data for {symbol}: {price_data['error']}"
                }
            
            # Extract closing prices
            close_prices = [float(candle[4]) for candle in price_data["data"]]
            all_price_data.append(close_prices)
        
        # Convert to numpy arrays and transpose to get [time][asset] format
        prices = np.array(all_price_data).T
        holding_qty = np.array(holding_qty)
        
        # Calculate portfolio values
        weighted_values = holding_qty * prices
        portfolio_values = weighted_values.sum(axis=1)
        
        # Format asset names for output
        asset_names = [symbol.replace("USDT", "") for symbol in symbols]
        
        return {
            "assets": asset_names,
            "portfolio_values": portfolio_values.tolist(),
            "initial_value": float(portfolio_values[0]),
            "final_value": float(portfolio_values[-1]),
            "change_percent": f"{((portfolio_values[-1] / portfolio_values[0]) - 1) * 100:.2f}%",
            "period": f"{interval} x {limit}"
        }
    except Exception as e:
        return {
            "error": f"Error calculating portfolio value: {str(e)}",
            "traceback": traceback.format_exc()
        }


@tool()
def portfolio_volatility(
        symbols: List[str],
        holding_qty: List[float],
        interval: str = "1d",
        limit: int = 90) -> Dict[str, Any]:
    '''
    Calculates the volatility (standard deviation of returns) of a portfolio using Binance price data
    
    Args:
        symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
        holding_qty: List with quantities of each asset held
        interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
        limit: Number of candlesticks to retrieve (max 1000)
        
    Returns:
        Dictionary containing portfolio volatility metrics
    '''
    try:
        if len(symbols) != len(holding_qty):
            return {
                "error": "Number of symbols must match number of holdings"
            }
        
        # Fetch price data for each asset
        all_price_data = []
        
        for symbol in symbols:
            price_data = get_binance_price_history.invoke({"pair": symbol, "interval": interval, "limit": limit})
            
            if "error" in price_data:
                return {
                    "error": f"Failed to fetch price data for {symbol}: {price_data['error']}"
                }
            
            # Extract closing prices
            close_prices = [float(candle[4]) for candle in price_data["data"]]
            all_price_data.append(close_prices)
        
        # Convert to numpy arrays and transpose to get [time][asset] format
        prices = np.array(all_price_data).T
        holding_qty = np.array(holding_qty)
        
        # Calculate portfolio values
        weighted_values = holding_qty * prices
        portfolio_values = weighted_values.sum(axis=1)
        
        # Calculate returns and volatility
        portfolio_returns = portfolio_values[1:]/portfolio_values[:-1] - 1
        portfolio_sd = float(portfolio_returns.std())
        
        # Format asset names for output
        asset_names = [symbol.replace("USDT", "") for symbol in symbols]
        
        return {
            "assets": asset_names,
            "portfolio_volatility": portfolio_sd,
            "annualized_volatility": float(portfolio_sd * np.sqrt(252)),  # Assuming daily data, annualized
            "annualized_volatility_percent": f"{float(portfolio_sd * np.sqrt(252) * 100):.2f}%",
            "returns_mean": float(portfolio_returns.mean()),
            "risk_analysis": "Higher volatility indicates higher risk but potentially higher returns",
            "period": f"{interval} x {limit}"
        }
    except Exception as e:
        return {
            "error": f"Error calculating portfolio volatility: {str(e)}",
            "traceback": traceback.format_exc()
        }


@tool()
def portfolio_summary(
        symbols: List[str],
        holding_qty: List[float],
        interval: str = "1d",
        limit: int = 90) -> Dict[str, Any]:
    '''
    Provides a comprehensive summary of a portfolio including value, volatility, and drawdown metrics using Binance price data
    
    Args:
        symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
        holding_qty: List with quantities of each asset held
        interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
        limit: Number of candlesticks to retrieve (max 1000)
        
    Returns:
        Dictionary containing portfolio summary metrics
    '''
    try:
        if len(symbols) != len(holding_qty):
            return {
                "error": "Number of symbols must match number of holdings"
            }
        
        # Fetch price data for each asset
        all_price_data = []
        
        for symbol in symbols:
            price_data = get_binance_price_history.invoke({"pair": symbol, "interval": interval, "limit": limit})
            
            if "error" in price_data:
                return {
                    "error": f"Failed to fetch price data for {symbol}: {price_data['error']}"
                }
            
            # Extract closing prices
            close_prices = [float(candle[4]) for candle in price_data["data"]]
            all_price_data.append(close_prices)
        
        # Convert to numpy arrays and transpose to get [time][asset] format
        prices = np.array(all_price_data).T
        holding_qty = np.array(holding_qty)
        
        # Format asset names for output
        asset_names = [symbol.replace("USDT", "") for symbol in symbols]
        
        # Calculate portfolio values
        weighted_values = holding_qty * prices
        portfolio_values = weighted_values.sum(axis=1)
        
        # Calculate allocation percentages
        latest_values = holding_qty * prices[-1]
        total_value = latest_values.sum()
        allocations = latest_values / total_value
        
        # Calculate returns
        portfolio_returns = portfolio_values[1:]/portfolio_values[:-1] - 1
        
        # Calculate drawdown
        rolling_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (rolling_max - portfolio_values) / rolling_max
        max_dd = float(drawdowns.max())
        
        # Asset allocation summary
        asset_allocation = []
        for i, asset in enumerate(asset_names):
            asset_allocation.append({
                "asset": asset,
                "quantity": float(holding_qty[i]),
                "value": float(latest_values[i]),
                "allocation_percent": f"{allocations[i] * 100:.2f}%"
            })
        
        return {
            "portfolio_summary": {
                "period": f"{interval} x {limit}",
                "total_value": float(total_value),
                "asset_count": len(asset_names),
                "performance": {
                    "initial_value": float(portfolio_values[0]),
                    "final_value": float(portfolio_values[-1]),
                    "total_return": f"{((portfolio_values[-1] / portfolio_values[0]) - 1) * 100:.2f}%",
                    "volatility": float(portfolio_returns.std()),
                    "annualized_volatility": f"{float(portfolio_returns.std() * np.sqrt(252) * 100):.2f}%",
                    "max_drawdown": f"{max_dd * 100:.2f}%",
                    "sharpe_ratio": float(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else None
                },
                "asset_allocation": asset_allocation
            }
        }
    except Exception as e:
        return {
            "error": f"Error calculating portfolio summary: {str(e)}",
            "traceback": traceback.format_exc()
        } 