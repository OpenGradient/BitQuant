from typing import Dict, Any, List
from langchain_core.tools import tool
from langgraph.graph.graph import RunnableConfig
from sklearn.linear_model import LinearRegression
import traceback
import numpy as np
import time

from langchain_core.tools import tool
from binance.spot import Spot  # type: ignore

from defi.analytics import extract_tokens_from_config


@tool()
def get_binance_price_history(
    token_symbol: str, candle_interval: str, num_candles: int
) -> Dict[str, Any]:
    """
    Retrieves historical price data for a token.
    """
    # Min value of 2 ensures we have at least two data points for calculating trends
    num_candles = min(max(2, int(num_candles)), 1000)

    trading_pair = f"{token_symbol.upper()}USDT"

    try:
        client = Spot(base_url="https://api.binance.us")
        klines = client.klines(
            symbol=trading_pair, interval=candle_interval, limit=num_candles
        )

        return {
            "token_symbol": token_symbol,
            "candle_interval": candle_interval,
            "num_candles": num_candles,
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
                "ignore",
            ],
        }

    except Exception as e:
        return {"error": f"Error fetching Binance data for {token_symbol}: {e}"}


@tool()
def analyze_price_trend(
    token_symbol: str, candle_interval: str, num_candles: int
) -> Dict[str, Any]:
    """
    Analyzes price trend for a token including moving averages, volatility metrics, 
    and basic technical indicators over the specified time period.
    """
    try:
        # Get the price history first
        price_data = get_binance_price_history.invoke(
            {
                "token_symbol": token_symbol,
                "candle_interval": candle_interval,
                "num_candles": num_candles,
            }
        )

        # Extract relevant data for analysis
        raw_data = price_data["data"]

        close_prices = [float(candle[4]) for candle in raw_data]

        # Calculate technical indicators and moving averages
        # 7-day simple moving average
        sma7 = []
        # 21-day simple moving average
        sma21 = []
        # Relative Strength Index (simplified)
        rsi = None

        if len(close_prices) >= 7:
            for i in range(6, len(close_prices)):
                sma7.append(sum(close_prices[i - 6 : i + 1]) / 7)

        if len(close_prices) >= 21:
            for i in range(20, len(close_prices)):
                sma21.append(sum(close_prices[i - 20 : i + 1]) / 21)

        # Calculate RSI (simplified version)
        if len(close_prices) >= 14:
            gains = []
            losses = []
            for i in range(1, 14):
                change = close_prices[i] - close_prices[i - 1]
                if change >= 0:
                    gains.append(float(change))
                    losses.append(0.0)
                else:
                    gains.append(0)
                    losses.append(float(abs(change)))

            avg_gain = float(sum(gains) / 14 if gains else 0)
            avg_loss = float(sum(losses) / 14 if losses else 0)

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100  # No losses means RSI = 100

        # Simple trend analysis
        if len(close_prices) < 2:
            trend = "Not enough data"
            recent_change = 0
        else:
            recent_change = ((close_prices[-1] / close_prices[0]) - 1) * 100
            trend = "Upward" if recent_change > 0 else "Downward"

        # Determine trend strength based on consistency
        trend_strength = "Neutral"
        if len(close_prices) >= 7:
            up_days = sum(
                1
                for i in range(1, len(close_prices))
                if close_prices[i] > close_prices[i - 1]
            )
            down_days = sum(
                1
                for i in range(1, len(close_prices))
                if close_prices[i] < close_prices[i - 1]
            )

            if up_days > len(close_prices) * 0.7:
                trend_strength = "Strong Upward"
            elif up_days > len(close_prices) * 0.55:
                trend_strength = "Moderate Upward"
            elif down_days > len(close_prices) * 0.7:
                trend_strength = "Strong Downward"
            elif down_days > len(close_prices) * 0.55:
                trend_strength = "Moderate Downward"

        return {
            "trend": trend,
            "trend_strength": trend_strength,
            "change_percent": (
                round(recent_change, 2) if len(close_prices) >= 2 else None
            ),
            "current_price": close_prices[-1] if close_prices else None,
            "price_range": {
                "min": min(close_prices) if close_prices else None,
                "max": max(close_prices) if close_prices else None,
            },
            "technical_indicators": {
                "sma7": sma7[-1] if sma7 else None,
                "sma21": sma21[-1] if sma21 else None,
                "sma_trend": (
                    "Upward"
                    if sma7 and sma21 and sma7[-1] > sma21[-1]
                    else "Downward" if sma7 and sma21 else None
                ),
                "rsi": round(rsi, 2) if rsi is not None else None,
            },
            "raw_data": price_data,
        }
    except Exception as e:
        return {
            "error": f"Error analyzing price trend for {token_symbol}: {e}",
        }


@tool()
def compare_assets(
    token_symbols: List[str], candle_interval: str, num_candles: int
) -> Dict[str, Any]:
    """
    Compare performance of multiple tokens, including detailed price trends, technical indicators,
    relative performance metrics, volatility analysis, and correlation data over the specified time period.
    """
    results = {}
    detailed_results = {}
    all_price_data = {}

    # Step 1: Collect individual asset data
    for token_symbol in token_symbols:
        try:
            analysis = analyze_price_trend.invoke(
                {
                    "token_symbol": token_symbol,
                    "candle_interval": candle_interval,
                    "num_candles": num_candles,
                }
            )

            # Skip if there was an error
            if "error" in analysis:
                results[token_symbol] = {"error": analysis["error"]}
                continue

            # Store detailed analysis results
            detailed_results[token_symbol] = analysis

            # Extract close prices for correlation analysis
            raw_data = analysis["raw_data"]["data"]
            all_price_data[token_symbol] = [float(candle[4]) for candle in raw_data]

            # Store basic metrics in results
            results[token_symbol] = {
                "trend": analysis["trend"],
                "trend_strength": analysis["trend_strength"],
                "change_percent": analysis["change_percent"],
                "current_price": analysis["current_price"],
                "volatility": calculate_volatility(all_price_data[token_symbol]),
                "technical_indicators": analysis["technical_indicators"],
            }

        except Exception as e:
            results[token_symbol] = {
                "error": f"Error analyzing {token_symbol}: {str(e)}",
                "traceback": traceback.format_exc(),
            }

    # Step 2: Calculate comparative metrics
    valid_pairs = {
        p: data
        for p, data in results.items()
        if "error" not in data and data.get("change_percent") is not None
    }

    comparative_analysis = {}

    if valid_pairs:
        # Performance rankings
        ranked_by_change = sorted(
            valid_pairs.items(),
            key=lambda x: x[1].get("change_percent", float("-inf")),
            reverse=True,
        )

        # Volatility rankings
        ranked_by_volatility = sorted(
            valid_pairs.items(),
            key=lambda x: x[1].get("volatility", float("-inf")),
            reverse=True,
        )

        # Calculate correlations between assets
        correlations = calculate_correlations(all_price_data)

        # Risk-adjusted returns (simple Sharpe ratio approximation)
        risk_adjusted = {}
        for symbol, data in valid_pairs.items():
            if data.get("volatility", 0) > 0:
                risk_adjusted[symbol] = data.get("change_percent", 0) / data.get(
                    "volatility", 1
                )
            else:
                risk_adjusted[symbol] = 0

        ranked_by_risk_adjusted = sorted(
            risk_adjusted.items(), key=lambda x: x[1], reverse=True
        )

        # Construct the comparative analysis
        comparative_analysis = {
            "performance_ranking": [
                {"symbol": pair[0], "change_percent": pair[1]["change_percent"]}
                for pair in ranked_by_change
            ],
            "volatility_ranking": [
                {"symbol": pair[0], "volatility": pair[1]["volatility"]}
                for pair in ranked_by_volatility
            ],
            "risk_adjusted_ranking": [
                {"symbol": symbol, "risk_adjusted_return": value}
                for symbol, value in ranked_by_risk_adjusted
            ],
            "correlations": correlations,
            "best_performer": {
                "token_symbol": ranked_by_change[0][0],
                "change_percent": ranked_by_change[0][1]["change_percent"],
            },
            "worst_performer": {
                "token_symbol": ranked_by_change[-1][0],
                "change_percent": ranked_by_change[-1][1]["change_percent"],
            },
            "most_volatile": {
                "token_symbol": ranked_by_volatility[0][0],
                "volatility": ranked_by_volatility[0][1]["volatility"],
            },
            "least_volatile": {
                "token_symbol": ranked_by_volatility[-1][0],
                "volatility": ranked_by_volatility[-1][1]["volatility"],
            },
            "best_risk_adjusted": {
                "token_symbol": ranked_by_risk_adjusted[0][0],
                "value": ranked_by_risk_adjusted[0][1],
            },
        }

        # Calculate basket performance (if we were to invest equally in all assets)
        if len(valid_pairs) > 0:
            avg_change = sum(
                data["change_percent"] for _, data in valid_pairs.items()
            ) / len(valid_pairs)
            comparative_analysis["basket_performance"] = {
                "average_change_percent": avg_change,
                "outperformers": [
                    symbol
                    for symbol, data in valid_pairs.items()
                    if data["change_percent"] > avg_change
                ],
                "underperformers": [
                    symbol
                    for symbol, data in valid_pairs.items()
                    if data["change_percent"] < avg_change
                ],
            }

    # Step 3: Construct the final return object
    return {
        "individual_assets": results,
        "comparative_analysis": comparative_analysis,
        "period": f"Past {num_candles} {candle_interval}",
        "analysis_timestamp": int(time.time()),
    }


def calculate_volatility(prices: List[float]) -> float:
    """
    Calculate a simple volatility metric (standard deviation of daily returns).

    Args:
        prices: List of closing prices

    Returns:
        Volatility value
    """
    if len(prices) < 2:
        return 0

    # Calculate daily returns
    returns = [(prices[i] / prices[i - 1] - 1) * 100 for i in range(1, len(prices))]

    # Calculate standard deviation of returns
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)

    return round(variance**0.5, 2)  # Standard deviation


def calculate_correlations(
    price_data: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate Pearson correlation coefficients between assets.

    Args:
        price_data: Dictionary with token symbols as keys and price lists as values

    Returns:
        Nested dictionary with correlation values
    """

    correlations = {}
    symbols = list(price_data.keys())

    # Ensure all price series have the same length by truncating to shortest
    min_length = min(len(prices) for prices in price_data.values())
    normalized_prices = {
        symbol: prices[-min_length:] for symbol, prices in price_data.items()
    }

    # Calculate returns instead of using absolute prices
    returns = {}
    for symbol, prices in normalized_prices.items():
        if len(prices) < 2:
            continue
        returns[symbol] = [
            (prices[i] / prices[i - 1] - 1) for i in range(1, len(prices))
        ]

    # Calculate correlations between each pair
    for i, symbol1 in enumerate(symbols):
        if symbol1 not in returns:
            continue

        correlations[symbol1] = {}

        for symbol2 in symbols:
            if symbol2 not in returns:
                continue

            if symbol1 == symbol2:
                correlations[symbol1][symbol2] = 1.0
                continue

            # Calculate Pearson correlation
            r1 = np.array(returns[symbol1])
            r2 = np.array(returns[symbol2])

            # Calculate correlation coefficient
            correlation = np.corrcoef(r1, r2)[0, 1]
            correlations[symbol1][symbol2] = round(float(correlation), 2)

    return correlations


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
            {
                "token_symbol": token_symbol,
                "candle_interval": candle_interval,
                "num_candles": num_candles,
            }
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
    candle_interval: str = "1d",
    num_candles: int = 90,
    config: RunnableConfig = None,
) -> Dict[str, Any]:
    """
    Analyzes the user's wallet portfolio using Binance price data. Returns a dictionary containing comprehensive portfolio analysis.
    """

    try:
        # Extract tokens from config
        symbols, holding_qty = extract_tokens_from_config(config)

        if not symbols or not holding_qty:
            return {
                "error": "No analyzable tokens found in wallet. Your wallet may contain only stablecoins or tokens not supported on Binance. Please provide custom_symbols and custom_quantities."
            }

        # Fetch price data for each asset
        all_price_data = []
        valid_symbols = []
        valid_quantities = []

        for i, symbol in enumerate(symbols):
            price_data = get_binance_price_history.invoke(
                {
                    "token_symbol": symbol,
                    "candle_interval": candle_interval,
                    "num_candles": num_candles,
                }
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
        return {"error": f"Error analyzing portfolio: {e}"}


@tool()
def portfolio_value(
    token_symbols: List[str],
    token_quantities: List[float],
    candle_interval: str = "1d",
    num_candles: int = 90,
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
                {
                    "token_symbol": token_symbol,
                    "candle_interval": candle_interval,
                    "num_candles": num_candles,
                }
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
    token_symbols: List[str],
    token_quantities: List[float],
    candle_interval: str = "1d",
    num_candles: int = 90,
) -> Dict[str, Any]:
    """
    Calculates the volatility (standard deviation of returns) of a portfolio over the specified time period. Do not pass in stablecoins.
    """
    try:
        if len(token_symbols) != len(token_quantities):
            return {"error": "Number of symbols must match number of holdings"}

        # Fetch price data for each asset
        all_price_data = []

        for token_symbol in token_symbols:
            price_data = get_binance_price_history.invoke(
                {
                    "token_symbol": token_symbol,
                    "candle_interval": candle_interval,
                    "num_candles": num_candles,
                }
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
    token_symbols: List[str],
    token_quantities: List[float],
    candle_interval: str = "1d",
    num_candles: int = 90,
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
                {
                    "token_symbol": token_symbol,
                    "candle_interval": candle_interval,
                    "num_candles": num_candles,
                }
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
            {
                "token_symbol": token_symbol,
                "candle_interval": candle_interval,
                "num_candles": num_candles,
            }
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
