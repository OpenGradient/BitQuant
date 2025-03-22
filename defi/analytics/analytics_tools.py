from typing import Dict, Any, List
from langchain_core.tools import tool
from langgraph.graph.graph import RunnableConfig
from sklearn.linear_model import LinearRegression
import traceback
import numpy as np
import time
from enum import StrEnum

from langchain_core.tools import tool
from binance.spot import Spot  # type: ignore

from defi.analytics.utils import extract_tokens_from_config


class CandleInterval(StrEnum):
    DAY = "1d"
    HOUR = "1h"
    WEEK = "1w"


@tool()
def get_binance_price_history(
    token_symbol: str, candle_interval: CandleInterval, num_candles: int
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
    token_symbol: str, candle_interval: CandleInterval, num_candles: int
) -> Dict[str, Any]:
    """
    Analyzes price trend for a token including moving averages, volatility metrics, 
    and enhanced technical indicators over the specified time period.
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

        # Process price data
        close_prices = [float(candle[4]) for candle in raw_data]
        open_prices = [float(candle[1]) for candle in raw_data]
        high_prices = [float(candle[2]) for candle in raw_data]
        low_prices = [float(candle[3]) for candle in raw_data]
        volumes = [float(candle[5]) for candle in raw_data]
        timestamps = [int(candle[0]) for candle in raw_data]

        # Simple trend analysis
        if len(close_prices) < 2:
            trend = "Not enough data"
            recent_change = 0
        else:
            recent_change = ((close_prices[-1] / close_prices[0]) - 1) * 100
            trend = "Upward" if recent_change > 0 else "Downward"

        # Calculate simple moving averages
        sma7 = []
        sma21 = []
        if len(close_prices) >= 7:
            for i in range(6, len(close_prices)):
                sma7.append(sum(close_prices[i - 6 : i + 1]) / 7)
        
        if len(close_prices) >= 21:
            for i in range(20, len(close_prices)):
                sma21.append(sum(close_prices[i - 20 : i + 1]) / 21)

        # 1. Bollinger Bands (20-period SMA with 2 standard deviations)
        bollinger_bands = {"upper": None, "middle": None, "lower": None}
        if len(close_prices) >= 20:
            # Middle band is 20-period SMA
            middle_band = sum(close_prices[-20:]) / 20
            # Calculate standard deviation
            std_dev = (sum((price - middle_band) ** 2 for price in close_prices[-20:]) / 20) ** 0.5
            # Upper and lower bands
            upper_band = middle_band + (2 * std_dev)
            lower_band = middle_band - (2 * std_dev)
            
            bollinger_bands = {
                "upper": round(upper_band, 2),
                "middle": round(middle_band, 2),
                "lower": round(lower_band, 2),
                "width": round((upper_band - lower_band) / middle_band, 4),  # Normalized width
                "position": round((close_prices[-1] - lower_band) / (upper_band - lower_band), 2) if upper_band != lower_band else 0.5
            }

        # 2. MACD (12-period EMA, 26-period EMA, 9-period signal)
        macd = {"value": None, "signal": None, "histogram": None}
        if len(close_prices) >= 26:
            # Calculate 12-period EMA
            ema12 = close_prices[0]
            k = 2 / (12 + 1)
            for i in range(1, len(close_prices)):
                ema12 = close_prices[i] * k + ema12 * (1 - k)

            # Calculate 26-period EMA
            ema26 = close_prices[0]
            k = 2 / (26 + 1)
            for i in range(1, len(close_prices)):
                ema26 = close_prices[i] * k + ema26 * (1 - k)

            # MACD line
            macd_line = ema12 - ema26

            # Signal line (9-period EMA of MACD)
            signal_line = macd_line  # Simplified calculation for brevity
            
            macd = {
                "value": round(macd_line, 4),
                "signal": round(signal_line, 4),
                "histogram": round(macd_line - signal_line, 4),
                "trend": "Bullish" if macd_line > signal_line else "Bearish"
            }

        # 4. Fibonacci Retracement Levels (based on recent high and low)
        fibonacci = {}
        if len(close_prices) >= 10:
            recent_high = max(high_prices[-20:]) if len(high_prices) >= 20 else max(high_prices)
            recent_low = min(low_prices[-20:]) if len(low_prices) >= 20 else min(low_prices)
            price_range = recent_high - recent_low
            
            fibonacci = {
                "levels": {
                    "0.0": round(recent_low, 2),
                    "0.236": round(recent_low + 0.236 * price_range, 2),
                    "0.382": round(recent_low + 0.382 * price_range, 2),
                    "0.5": round(recent_low + 0.5 * price_range, 2),
                    "0.618": round(recent_low + 0.618 * price_range, 2),
                    "0.786": round(recent_low + 0.786 * price_range, 2),
                    "1.0": round(recent_high, 2)
                },
                "current_position": "None"
            }
            
            # Identify nearest Fibonacci level to current price
            current_price = close_prices[-1]
            levels = list(fibonacci["levels"].items())
            levels.sort(key=lambda x: abs(current_price - x[1]))
            fibonacci["current_position"] = levels[0][0]

        # 5. Volume Analysis
        volume_analysis = {"trend": "Neutral", "avg_volume": None, "current_vs_avg": None}
        if len(volumes) >= 7:
            avg_volume = sum(volumes[-7:]) / 7
            current_volume = volumes[-1]
            
            volume_trend = "Increasing" if current_volume > avg_volume else "Decreasing"
            # Check if volume confirms price trend
            price_up = close_prices[-1] > open_prices[-1]
            volume_confirms = (price_up and volume_trend == "Increasing") or (not price_up and volume_trend == "Decreasing")
            
            volume_analysis = {
                "trend": volume_trend,
                "avg_volume": round(avg_volume, 2),
                "current_volume": round(current_volume, 2),
                "current_vs_avg": round((current_volume / avg_volume - 1) * 100, 2),
                "confirms_price": volume_confirms
            }

        # 7. Support & Resistance Levels (simple method based on recent price action)
        support_resistance = {"support": [], "resistance": []}
        if len(close_prices) >= 30:
            # Simplified algorithm to find support/resistance
            window_size = min(30, len(close_prices) // 3)
            for i in range(window_size, len(close_prices) - window_size):
                # Check if this point is a local minimum (support)
                if all(low_prices[i] <= low_prices[j] for j in range(i-window_size, i)) and \
                   all(low_prices[i] <= low_prices[j] for j in range(i+1, i+window_size+1)):
                    support_resistance["support"].append(round(low_prices[i], 2))
                
                # Check if this point is a local maximum (resistance)
                if all(high_prices[i] >= high_prices[j] for j in range(i-window_size, i)) and \
                   all(high_prices[i] >= high_prices[j] for j in range(i+1, i+window_size+1)):
                    support_resistance["resistance"].append(round(high_prices[i], 2))
            
            # Limit to top 3 strongest levels for each
            support_resistance["support"] = sorted(support_resistance["support"])[:3]
            support_resistance["resistance"] = sorted(support_resistance["resistance"], reverse=True)[:3]

        # 9. Token-specific metrics
        token_metrics = {}
        if len(close_prices) > 0 and len(volumes) > 0:
            current_price = close_prices[-1]
            avg_daily_volume = sum(volumes[-min(7, len(volumes)):]) / min(7, len(volumes))
            
            token_metrics = {
                "price": round(current_price, 4),
                "avg_daily_volume_usd": round(avg_daily_volume * current_price, 2),
                "volatility": round(((max(close_prices[-7:]) / min(close_prices[-7:]) - 1) * 100), 2) if len(close_prices) >= 7 else None,
                "liquidity_score": "High" if avg_daily_volume * current_price > 1000000 else "Medium" if avg_daily_volume * current_price > 100000 else "Low"
            }

        return {
            "token_symbol": token_symbol,
            "trend": trend,
            "change_percent": round(recent_change, 2) if len(close_prices) >= 2 else None,
            "current_price": close_prices[-1] if close_prices else None,
            "price_range": {
                "min": round(min(close_prices), 4) if close_prices else None,
                "max": round(max(close_prices), 4) if close_prices else None,
            },
            "simple_indicators": {
                "sma7": round(sma7[-1], 4) if sma7 else None,
                "sma21": round(sma21[-1], 4) if sma21 else None,
                "sma_trend": "Bullish" if sma7 and sma21 and sma7[-1] > sma21[-1] else "Bearish" if sma7 and sma21 else None,
            },
            "technical_indicators": {
                "bollinger_bands": bollinger_bands,
                "macd": macd,
                "fibonacci": fibonacci,
                "volume_analysis": volume_analysis,
                "support_resistance": support_resistance
            },
            "token_metrics": token_metrics,
            "analysis_summary": get_analysis_summary(
                trend, sma7, sma21, bollinger_bands, macd, volume_analysis
            )
        }
    except Exception as e:
        return {
            "error": f"Error analyzing price trend for {token_symbol}: {e}",
        }


def get_analysis_summary(
    trend, sma7, sma21, bollinger_bands, macd, volume_analysis
):
    """Generate a simple summary of the analysis results."""
    summary = []
    
    # Trend summary
    trend_desc = f"The overall trend is {trend.lower()}."
    summary.append(trend_desc)
    
    # Moving average summary
    if sma7 and sma21:
        if sma7[-1] > sma21[-1]:
            ma_desc = "Short-term average above long-term average suggests bullish momentum."
            summary.append(ma_desc)
        else:
            ma_desc = "Short-term average below long-term average suggests bearish momentum."
            summary.append(ma_desc)
    
    # Bollinger Bands summary
    if bollinger_bands["upper"] is not None:
        position = bollinger_bands["position"]
        if position > 0.8:
            bb_desc = "Price near upper Bollinger Band suggests overbought conditions."
            summary.append(bb_desc)
        elif position < 0.2:
            bb_desc = "Price near lower Bollinger Band suggests oversold conditions."
            summary.append(bb_desc)
    
    # MACD summary
    if macd["value"] is not None:
        macd_desc = f"MACD indicates {macd['trend'].lower()} momentum."
        summary.append(macd_desc)
    
    # Volume confirmation
    if volume_analysis["avg_volume"] is not None:
        if volume_analysis["confirms_price"]:
            vol_desc = "Volume confirms price movement."
        else:
            vol_desc = "Volume does not confirm price movement, suggesting potential reversal."
        summary.append(vol_desc)
    
    # Combine into a paragraph
    return " ".join(summary)


@tool()
def compare_assets(
    token_symbols: List[str], candle_interval: CandleInterval, num_candles: int
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
    token_symbol: str, candle_interval: CandleInterval = CandleInterval.DAY, num_candles: int = 90
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
    candle_interval: CandleInterval = CandleInterval.DAY,
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
    candle_interval: CandleInterval = CandleInterval.DAY,
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
    candle_interval: CandleInterval = CandleInterval.DAY,
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
    candle_interval: CandleInterval = CandleInterval.DAY,
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
    token_symbol: str, candle_interval: CandleInterval = CandleInterval.DAY, num_candles: int = 90
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
