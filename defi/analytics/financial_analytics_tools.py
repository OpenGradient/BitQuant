from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import traceback
import json
import os
from langchain_core.tools import tool
from defi.analytics.binance_tools import get_binance_price_history
from langgraph.graph.graph import RunnableConfig
from sklearn.linear_model import LinearRegression


# Load token list mapping from address to symbol
def load_token_list():
    try:
        token_list_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "static",
            "tokenlist.json",
        )
        with open(token_list_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading token list: {e}")
        return {}


# Token address to symbol mapping
TOKEN_LIST = load_token_list()


# Helper function to extract tokens from config
def extract_tokens_from_config(config: RunnableConfig) -> Tuple[List[str], List[float]]:
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
    token_symbol: str, candle_interval: str = "1d", num_candles: int = 90
) -> Dict[str, Any]:
    """
    Calculates the maximum drawdown for a cryptocurrency using Binance price data

    Args:
        token_symbol: Token symbol (e.g., "BTC")
        candle_interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
        num_candles: Number of candlesticks to retrieve (max 1000)

    Returns:
        Dictionary containing the maximum drawdown value
    """
    try:
        # Get price data from Binance
        price_data = get_binance_price_history.invoke(
            {"token_symbol": token_symbol, "candle_interval": candle_interval, "num_candles": num_candles}
        )

        if "error" in price_data:
            return {
                "error": f"Failed to fetch price data for {token_symbol}: {price_data['error']}"
            }

        # Extract closing prices
        price_series = [float(candle[4]) for candle in price_data["data"]]

        # Calculate max drawdown
        price_series = np.array(price_series)
        rolling_max = np.maximum.accumulate(price_series)
        drawdowns = (rolling_max - price_series) / rolling_max
        max_dd = float(drawdowns.max())

        return {
            "token_symbol": token_symbol,
            "period": f"{candle_interval} x {num_candles}",
            "max_drawdown": max_dd,
            "max_drawdown_percent": f"{max_dd * 100:.2f}%",
            "explanation": "Maximum drawdown represents the largest percentage drop from a peak to a subsequent trough",
        }
    except Exception as e:
        return {
            "error": f"Error calculating maximum drawdown: {str(e)}",
            "traceback": traceback.format_exc(),
        }


@tool()
def analyze_wallet_portfolio(
    config: RunnableConfig,
    candle_interval: str = "1d",
    num_candles: int = 90,
) -> Dict[str, Any]:
    """
    Analyzes the user's wallet portfolio using Binance price data. Returns a dictionary containing comprehensive portfolio analysis.
    """
    try:

        # Extract tokens from config
        symbols, holding_qty = extract_tokens_from_config(config)

        # Check if we found any tokens
        if not symbols or not holding_qty:
            return {
                "error": "No analyzable tokens found in wallet. Your wallet may contain only stablecoins or tokens not supported on Binance. Please provide custom_symbols and custom_quantities."
            }

        if len(symbols) != len(holding_qty):
            return {"error": "Number of symbols must match number of holdings"}

        # Fetch price data for each asset
        all_price_data = []
        valid_symbols = []
        valid_quantities = []

        for i, symbol in enumerate(symbols):
            price_data = get_binance_price_history.invoke(
                {"token_symbol": symbol, "candle_interval": candle_interval, "num_candles": num_candles}
            )

            if "error" in price_data:
                print(
                    f"Warning: Failed to fetch price data for {symbol}: {price_data.get('error')}"
                )
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
        portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1

        # Calculate drawdown
        rolling_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (rolling_max - portfolio_values) / rolling_max
        max_dd = float(drawdowns.max())

        # Asset allocation summary
        asset_allocation = []
        for i, asset in enumerate(asset_names):
            asset_allocation.append(
                {
                    "asset": asset,
                    "quantity": float(holding_qty[i]),
                    "value": float(latest_values[i]),
                    "allocation_percent": f"{allocations[i] * 100:.2f}%",
                }
            )

        return {
            "portfolio_summary": {
                "period": f"{candle_interval} x {num_candles}",
                "total_value": float(total_value),
                "asset_count": len(asset_names),
                "performance": {
                    "initial_value": float(portfolio_values[0]),
                    "final_value": float(portfolio_values[-1]),
                    "total_return": f"{((portfolio_values[-1] / portfolio_values[0]) - 1) * 100:.2f}%",
                    "volatility": float(portfolio_returns.std()),
                    "annualized_volatility": f"{float(portfolio_returns.std() * np.sqrt(252) * 100):.2f}%",
                    "max_drawdown": f"{max_dd * 100:.2f}%",
                    "sharpe_ratio": (
                        float(
                            portfolio_returns.mean()
                            / portfolio_returns.std()
                            * np.sqrt(252)
                        )
                        if portfolio_returns.std() > 0
                        else None
                    ),
                },
                "asset_allocation": asset_allocation,
            }
        }
    except Exception as e:
        return {
            "error": f"Error analyzing portfolio: {str(e)}",
            "traceback": traceback.format_exc(),
        }


@tool()
def portfolio_value(
    token_symbols: List[str], token_quantities: List[float], candle_interval: str = "1d", num_candles: int = 90
) -> Dict[str, Any]:
    """
    Creates the time series of portfolio total value using Binance price data over the specified time period.
    """
    try:
        if len(token_symbols) != len(token_quantities):
            return {"error": "Number of symbols must match number of holdings"}

        # Fetch price data for each asset
        all_price_data = []

        for token_symbol in token_symbols:
            price_data = get_binance_price_history.invoke(
                {"token_symbol": token_symbol, "candle_interval": candle_interval, "num_candles": num_candles}
            )

            if "error" in price_data:
                return {
                    "error": f"Failed to fetch price data for {token_symbol}: {price_data['error']}"
                }

            # Extract closing prices
            close_prices = [float(candle[4]) for candle in price_data["data"]]
            all_price_data.append(close_prices)

        # Convert to numpy arrays and transpose to get [time][asset] format
        prices = np.array(all_price_data).T
        holding_qty = np.array(token_quantities)

        # Calculate portfolio values
        weighted_values = holding_qty * prices
        portfolio_values = weighted_values.sum(axis=1)

        return {
            "assets": token_symbols,
            "portfolio_values": portfolio_values.tolist(),
            "initial_value": float(portfolio_values[0]),
            "final_value": float(portfolio_values[-1]),
            "change_percent": f"{((portfolio_values[-1] / portfolio_values[0]) - 1) * 100:.2f}%",
            "period": f"{candle_interval} x {num_candles}",
        }
    except Exception as e:
        return {
            "error": f"Error calculating portfolio value: {str(e)}",
            "traceback": traceback.format_exc(),
        }


@tool()
def portfolio_volatility(
    token_symbols: List[str], token_quantities: List[float], candle_interval: str = "1d", num_candles: int = 90
) -> Dict[str, Any]:
    """
    Calculates the volatility (standard deviation of returns) of a portfolio using Binance price data over the specified time period.
    """
    try:
        if len(token_symbols) != len(token_quantities):
            return {"error": "Number of symbols must match number of holdings"}

        # Fetch price data for each asset
        all_price_data = []

        for token_symbol in token_symbols:
            price_data = get_binance_price_history.invoke(
                {"token_symbol": token_symbol, "candle_interval": candle_interval, "num_candles": num_candles}
            )

            if "error" in price_data:
                return {
                    "error": f"Failed to fetch price data for {token_symbol}: {price_data['error']}"
                }

            # Extract closing prices
            close_prices = [float(candle[4]) for candle in price_data["data"]]
            all_price_data.append(close_prices)

        # Convert to numpy arrays and transpose to get [time][asset] format
        prices = np.array(all_price_data).T
        holding_qty = np.array(token_quantities)

        # Calculate portfolio values
        weighted_values = holding_qty * prices
        portfolio_values = weighted_values.sum(axis=1)

        # Calculate returns and volatility
        portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1
        portfolio_sd = float(portfolio_returns.std())

        return {
            "assets": token_symbols,
            "portfolio_volatility": portfolio_sd,
            "annualized_volatility": float(
                portfolio_sd * np.sqrt(252)
            ),  # Assuming daily data, annualized
            "annualized_volatility_percent": f"{float(portfolio_sd * np.sqrt(252) * 100):.2f}%",
            "returns_mean": float(portfolio_returns.mean()),
            "risk_analysis": "Higher volatility indicates higher risk but potentially higher returns",
            "period": f"{candle_interval} x {num_candles}",
        }
    except Exception as e:
        return {
            "error": f"Error calculating portfolio volatility: {str(e)}",
            "traceback": traceback.format_exc(),
        }


@tool()
def portfolio_summary(
    token_symbols: List[str], token_quantities: List[float], candle_interval: str = "1d", num_candles: int = 90
) -> Dict[str, Any]:
    """
    Provides a comprehensive summary of a portfolio including value, volatility, and drawdown metrics using Binance price data over the specified time period.
    """
    try:
        if len(token_symbols) != len(token_quantities):
            return {"error": "Number of symbols must match number of holdings"}

        # Fetch price data for each asset
        all_price_data = []

        for token_symbol in token_symbols:
            price_data = get_binance_price_history.invoke(
                {"token_symbol": token_symbol, "candle_interval": candle_interval, "num_candles": num_candles}
            )

            if "error" in price_data:
                return {
                    "error": f"Failed to fetch price data for {token_symbol}: {price_data['error']}"
                }

            # Extract closing prices
            close_prices = [float(candle[4]) for candle in price_data["data"]]
            all_price_data.append(close_prices)

        # Convert to numpy arrays and transpose to get [time][asset] format
        prices = np.array(all_price_data).T
        holding_qty = np.array(token_quantities)

        # Calculate portfolio values
        weighted_values = holding_qty * prices
        portfolio_values = weighted_values.sum(axis=1)

        # Calculate allocation percentages
        latest_values = holding_qty * prices[-1]
        total_value = latest_values.sum()
        allocations = latest_values / total_value

        # Calculate returns
        portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1

        # Calculate drawdown
        rolling_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (rolling_max - portfolio_values) / rolling_max
        max_dd = float(drawdowns.max())

        # Asset allocation summary
        asset_allocation = []
        for i, asset in enumerate(token_symbols):
            asset_allocation.append(
                {
                    "asset": asset,
                    "quantity": float(holding_qty[i]),
                    "value": float(latest_values[i]),
                    "allocation_percent": f"{allocations[i] * 100:.2f}%",
                }
            )

        return {
            "portfolio_summary": {
                "period": f"{candle_interval} x {num_candles}",
                "total_value": float(total_value),
                "asset_count": len(token_symbols),
                "performance": {
                    "initial_value": float(portfolio_values[0]),
                    "final_value": float(portfolio_values[-1]),
                    "total_return": f"{((portfolio_values[-1] / portfolio_values[0]) - 1) * 100:.2f}%",
                    "volatility": float(portfolio_returns.std()),
                    "annualized_volatility": f"{float(portfolio_returns.std() * np.sqrt(252) * 100):.2f}%",
                    "max_drawdown": f"{max_dd * 100:.2f}%",
                    "sharpe_ratio": (
                        float(
                            portfolio_returns.mean()
                            / portfolio_returns.std()
                            * np.sqrt(252)
                        )
                        if portfolio_returns.std() > 0
                        else None
                    ),
                },
                "asset_allocation": asset_allocation,
            }
        }
    except Exception as e:
        return {
            "error": f"Error calculating portfolio summary: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def volatility_trend(price_series):
    """
    Gives some information about trend in volatility by calculating
    the slope of the line fit to the square root of
    absolute value of returns over time.

    Parameters
    ----------
    price_series : array
        1-d array of prices.

    Returns
    -------
    float
        Linear regression coefficient interpreted as
        the average increase or decrease of
        square root of absolute value of returns per day.
    """
    return_series = price_series[1:] / price_series[:-1] - 1
    log_abs_returns = np.log(np.abs(return_series) + 1e-12)
    index_series = np.arange(log_abs_returns.shape[0]).reshape(-1, 1)
    linreg = LinearRegression().fit(index_series, log_abs_returns)
    # preds = linreg.predict(index_series)
    # se = np.sqrt(((preds-log_abs_returns)**2).sum() / ((log_abs_returns.shape[0]-2)*
    #                   np.sum((log_abs_returns-log_abs_returns.mean())**2)))
    # tstat = linreg.coef_[0]/se
    return linreg.coef_[0]


@tool()
def analyze_volatility_trend(
    token_symbol: str, candle_interval: str = "1d", num_candles: int = 90
) -> Dict[str, Any]:
    """
    Analyzes the trend in volatility for a cryptocurrency over the specified time period.
    """
    try:
        # Get price data from Binance
        price_data = get_binance_price_history.invoke(
            {"token_symbol": token_symbol, "candle_interval": candle_interval, "num_candles": num_candles}
        )

        if "error" in price_data:
            return {
                "error": f"Failed to fetch price data for {token_symbol}: {price_data['error']}"
            }

        # Extract closing prices
        price_series = [float(candle[4]) for candle in price_data["data"]]

        # Calculate volatility trend
        price_series = np.array(price_series)
        vol_trend = volatility_trend(price_series)

        # Interpret the trend
        if vol_trend > 0:
            interpretation = "Increasing volatility trend"
            direction = "upward"
        elif vol_trend < 0:
            interpretation = "Decreasing volatility trend"
            direction = "downward"
        else:
            interpretation = "Stable volatility"
            direction = "stable"

        # Calculate standard volatility for comparison
        returns = price_series[1:] / price_series[:-1] - 1
        std_volatility = float(returns.std())

        return {
            "token_symbol": token_symbol,
            "period": f"{candle_interval} x {num_candles}",
            "volatility_trend_coefficient": float(vol_trend),
            "trend_direction": direction,
            "current_volatility": std_volatility,
            "annualized_volatility": f"{float(std_volatility * np.sqrt(252) * 100):.2f}%",
            "interpretation": interpretation,
            "explanation": "A positive coefficient indicates volatility is increasing over time, while a negative coefficient indicates decreasing volatility.",
        }
    except Exception as e:
        return {
            "error": f"Error analyzing volatility trend: {str(e)}",
            "traceback": traceback.format_exc(),
        }
