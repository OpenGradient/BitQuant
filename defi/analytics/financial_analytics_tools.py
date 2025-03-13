from typing import Dict, Any, List
import numpy as np
import traceback
from langchain_core.tools import tool


@tool()
def max_drawdown(price_series: List[float]) -> Dict[str, Any]:
    '''
    Calculates the maximum drawdown in terms of return
    in the price series. Drawdown is the greatest amount
    of negative return following a rolling maximum, and
    can be helpful in assessing risk

    Args:
        price_series: 1-d series of prices as a list of floats

    Returns:
        Dictionary containing the maximum drawdown value
    '''
    try:
        price_series = np.array(price_series)
        rolling_max = np.maximum.accumulate(price_series)
        drawdowns = (rolling_max - price_series) / rolling_max
        max_dd = float(drawdowns.max())
        
        return {
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
        holding_qty: List[float],
        prices: List[List[float]]) -> Dict[str, Any]:
    '''
    Creates the time series of portfolio total value given
    a constant quantity of each asset

    Args:
        holding_qty: 1-d array with a float for the quantity of each asset held
        prices: 2-d array with prices of each asset over time where
               axis 0 (row) is time and axis 1 (column) is asset,
               most recent price last

    Returns:
        Dictionary containing time series of portfolio total value
    '''
    try:
        holding_qty = np.array(holding_qty)
        prices = np.array(prices)
        
        weighted_values = holding_qty * prices
        portfolio_values = weighted_values.sum(axis=1)
        
        return {
            "portfolio_values": portfolio_values.tolist(),
            "initial_value": float(portfolio_values[0]),
            "final_value": float(portfolio_values[-1]),
            "change_percent": f"{((portfolio_values[-1] / portfolio_values[0]) - 1) * 100:.2f}%"
        }
    except Exception as e:
        return {
            "error": f"Error calculating portfolio value: {str(e)}",
            "traceback": traceback.format_exc()
        }


@tool()
def portfolio_volatility(
        holding_qty: List[float],
        prices: List[List[float]]) -> Dict[str, Any]:
    '''
    Calculates the volatility (standard deviation of returns) of a portfolio

    Args:
        holding_qty: 1-d array with a float for the quantity of each asset held
        prices: 2-d array with prices of each asset over time where
               axis 0 (row) is time and axis 1 (column) is asset,
               most recent price last

    Returns:
        Dictionary containing portfolio volatility metrics
    '''
    try:
        holding_qty = np.array(holding_qty)
        prices = np.array(prices)
        
        weighted_values = holding_qty * prices
        portfolio_values = weighted_values.sum(axis=1)
        portfolio_returns = portfolio_values[1:]/portfolio_values[:-1] - 1
        portfolio_sd = float(portfolio_returns.std())
        
        return {
            "portfolio_volatility": portfolio_sd,
            "annualized_volatility": float(portfolio_sd * np.sqrt(252)),  # Assuming daily data, annualized
            "returns_mean": float(portfolio_returns.mean()),
            "risk_analysis": "Higher volatility indicates higher risk but potentially higher returns"
        }
    except Exception as e:
        return {
            "error": f"Error calculating portfolio volatility: {str(e)}",
            "traceback": traceback.format_exc()
        }


@tool()
def portfolio_summary(
        asset_list: List[str],
        holding_qty: List[float],
        prices: List[List[float]]) -> Dict[str, Any]:
    '''
    Provides a comprehensive summary of a portfolio including value,
    volatility, and drawdown metrics

    Args:
        asset_list: List of asset names or identifiers
        holding_qty: List with quantities of each asset held
        prices: 2-d array with prices of each asset over time

    Returns:
        Dictionary containing portfolio summary metrics
    '''
    try:
        holding_qty = np.array(holding_qty)
        prices = np.array(prices)
        
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
        for i, asset in enumerate(asset_list):
            asset_allocation.append({
                "asset": asset,
                "quantity": float(holding_qty[i]),
                "value": float(latest_values[i]),
                "allocation_percent": f"{allocations[i] * 100:.2f}%"
            })
        
        return {
            "portfolio_summary": {
                "total_value": float(total_value),
                "asset_count": len(asset_list),
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