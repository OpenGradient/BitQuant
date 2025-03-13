from typing import Dict, Any, List, Optional
import numpy as np
import traceback
import json
import os
from langchain_core.tools import tool
from defi.analytics.binance_tools import get_binance_price_history
from langgraph.graph.graph import RunnableConfig


# Load token list mapping from address to symbol
def load_token_list():
    try:
        token_list_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      "static", "tokenlist.json")
        with open(token_list_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading token list: {e}")
        return {}


# Token address to symbol mapping
TOKEN_LIST = load_token_list()


# Helper function to extract tokens from config
def extract_tokens_from_config(config: RunnableConfig):
    """Extract token holdings from the configurable context"""
    try:
        configurable = config["configurable"]
        tokens = configurable.get("tokens", [])
        
        symbols = []
        quantities = []
        
        for token in tokens:
            address = token.get("address")
            amount = token.get("amount", 0)
            
            # Skip if address is missing or amount is 0
            if not address or amount <= 0:
                continue
                
            # Look up token symbol from the address
            token_info = TOKEN_LIST.get(address)
            if not token_info:
                continue
                
            symbol = token_info.get("symbol")
            if not symbol:
                continue
            
            # Convert to Binance trading pair format by default
            trading_pair = f"{symbol}USDT"
            
            # Add the token to our analysis list
            symbols.append(trading_pair)
            quantities.append(amount)
            
        return symbols, quantities
    except Exception as e:
        print(f"Error extracting tokens: {e}")
        return [], []


@tool()
def max_drawdown_for_token(
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
def analyze_wallet_portfolio(
        config: RunnableConfig,
        interval: str = "1d",
        limit: int = 90,
        custom_symbols: Optional[List[str]] = None,
        custom_quantities: Optional[List[float]] = None) -> Dict[str, Any]:
    '''
    Analyzes the user's wallet portfolio or a custom portfolio using Binance price data
    
    Args:
        config: The runnable config containing user's wallet tokens
        interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
        limit: Number of candlesticks to retrieve (max 1000)
        custom_symbols: Optional list of trading pairs to override wallet tokens
        custom_quantities: Optional list of quantities to override wallet amounts
        
    Returns:
        Dictionary containing comprehensive portfolio analysis
    '''
    try:
        # Determine if we're using custom portfolio or wallet tokens
        if custom_symbols and custom_quantities:
            symbols = custom_symbols
            holding_qty = custom_quantities
        else:
            # Extract tokens from config
            symbols, holding_qty = extract_tokens_from_config(config)
            
            # Check if we found any tokens
            if not symbols or not holding_qty:
                return {
                    "error": "No analyzable tokens found in wallet. Your wallet may contain only stablecoins or tokens not supported on Binance. Please provide custom_symbols and custom_quantities."
                }
        
        if len(symbols) != len(holding_qty):
            return {
                "error": "Number of symbols must match number of holdings"
            }
        
        # Fetch price data for each asset
        all_price_data = []
        valid_symbols = []
        valid_quantities = []
        
        for i, symbol in enumerate(symbols):
            price_data = get_binance_price_history.invoke({"pair": symbol, "interval": interval, "limit": limit})
            
            if "error" in price_data:
                print(f"Warning: Failed to fetch price data for {symbol}: {price_data.get('error')}")
                continue  # Skip this token but continue with others
            
            # Extract closing prices
            close_prices = [float(candle[4]) for candle in price_data["data"]]
            all_price_data.append(close_prices)
            valid_symbols.append(symbol)
            valid_quantities.append(holding_qty[i])
        
        if not all_price_data:
            return {
                "error": "Could not fetch price data for any of the tokens in the wallet. Please try with custom symbols that are available on Binance."
            }
        
        # Convert to numpy arrays and transpose to get [time][asset] format
        prices = np.array(all_price_data).T
        holding_qty = np.array(valid_quantities)
        
        # Format asset names for output
        asset_names = [symbol.replace("USDT", "") for symbol in valid_symbols]
        
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
            "error": f"Error analyzing portfolio: {str(e)}",
            "traceback": traceback.format_exc()
        }


# Keep existing tools for backwards compatibility
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
    return max_drawdown_for_token(symbol, interval, limit)


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