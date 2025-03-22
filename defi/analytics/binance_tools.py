import traceback
import numpy as np
import time
from typing import Dict, Any, List

from langchain_core.tools import tool
from binance.spot import Spot  # type: ignore


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
    Analyzes price trend for a token including moving averages, volatility metrics, and basic technical indicators.
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
    relative performance metrics, volatility analysis, and correlation data.

    Args:
        token_symbols: List of token symbols to compare (e.g. ["BTC", "ETH", "SOL"])
        candle_interval: Time interval for candles (e.g. "1d", "4h", "1w")
        num_candles: Number of candles to retrieve for analysis
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
        "market_context": {
            "period": f"{candle_interval} x {num_candles}",
            "start_date": (
                detailed_results[token_symbols[0]]["raw_data"]["data"][0][0]
                if token_symbols and token_symbols[0] in detailed_results
                else None
            ),
            "end_date": (
                detailed_results[token_symbols[0]]["raw_data"]["data"][-1][0]
                if token_symbols and token_symbols[0] in detailed_results
                else None
            ),
        },
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
