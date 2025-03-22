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
    """
    One of 1d, 1h, 1w
    """
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

        # keep only the columns we need
        klines = [candle[:6] for candle in klines]

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
        
        # Enhanced moving averages calculations with more periods and types
        def calculate_sma(prices, period):
            """Calculate Simple Moving Average for a given period"""
            if len(prices) < period:
                return []
            
            sma_values = []
            # Use a sliding window approach
            for i in range(period - 1, len(prices)):
                window = prices[i - (period - 1) : i + 1]
                sma_values.append(sum(window) / period)
            return sma_values
        
        # Calculate key moving averages (both SMA and EMA)
        # Short-term moving averages
        sma7 = calculate_sma(close_prices, 7)
        sma20 = calculate_sma(close_prices, 20)
        sma50 = calculate_sma(close_prices, 50)
        sma200 = calculate_sma(close_prices, 200)

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

        # 9. Token-specific metrics
        token_metrics = {}
        if len(close_prices) > 0 and len(volumes) > 0:
            current_price = close_prices[-1]
            avg_daily_volume = sum(volumes[-min(7, len(volumes)):]) / min(7, len(volumes))
            
            token_metrics = {
                "price": round(current_price, 4),
                "avg_daily_volume_usd": round(avg_daily_volume * current_price, 2),
                "volatility": round(((max(close_prices[-7:]) / min(close_prices[-7:]) - 1) * 100), 2) if len(close_prices) >= 7 else None,
            }

        return {
            "token_symbol": token_symbol,
            "current_price": close_prices[-1] if close_prices else None,
            "price_range": {
                "min": round(min(close_prices), 4) if close_prices else None,
                "max": round(max(close_prices), 4) if close_prices else None,
                "open": round(open_prices[0], 4) if open_prices else None,
                "close": round(close_prices[-1], 4) if close_prices else None,
            },
            "moving_averages": {
                # Simple Moving Averages
                "sma7": round(sma7[-1], 4) if sma7 else None,
                "sma20": round(sma20[-1], 4) if sma20 else None,
                "sma50": round(sma50[-1], 4) if sma50 else None,
                "sma200": round(sma200[-1], 4) if sma200 else None,
                # Key crossovers and trends
                "golden_cross": "Yes" if sma50 and sma200 and sma50[-1] > sma200[-1] and (len(sma50) > 1 and len(sma200) > 1 and sma50[-2] <= sma200[-2]) else "No",
                "death_cross": "Yes" if sma50 and sma200 and sma50[-1] < sma200[-1] and (len(sma50) > 1 and len(sma200) > 1 and sma50[-2] >= sma200[-2]) else "No",
                "short_trend": "Bullish" if sma7 and sma20 and sma7[-1] > sma20[-1] else "Bearish" if sma7 and sma20 else "Neutral",
                "long_trend": "Bullish" if sma50 and sma200 and sma50[-1] > sma200[-1] else "Bearish" if sma50 and sma200 else "Neutral",
            },
            "technical_indicators": {
                "bollinger_bands": bollinger_bands,
                "fibonacci": fibonacci,
                "volume_analysis": volume_analysis,
            },
            "token_metrics": token_metrics,
            "analysis_summary": get_analysis_summary(
                sma7, sma20, sma50, sma200, bollinger_bands, volume_analysis
            )
        }
    except Exception as e:
        return {
            "error": f"Error analyzing price trend for {token_symbol}: {e}",
        }


def get_analysis_summary(
    sma7, sma20, sma50, sma200, bollinger_bands, volume_analysis
):
    """Generate a simple summary of the analysis results."""
    summary = []
    
    # Moving average summary - Enhanced with multiple timeframes
    # Short-term trend
    if sma7 and sma20:
        if sma7[-1] > sma20[-1]:
            ma_short_desc = "Short-term moving averages indicate bullish momentum."
            summary.append(ma_short_desc)
        else:
            ma_short_desc = "Short-term moving averages indicate bearish momentum."
            summary.append(ma_short_desc)
    
    # Long-term trend and major crossovers
    if sma50 and sma200:
        # Check for golden cross (50-day crosses above 200-day)
        if len(sma50) > 1 and len(sma200) > 1:
            if sma50[-1] > sma200[-1] and sma50[-2] <= sma200[-2]:
                summary.append("Golden Cross detected - a strong bullish signal.")
            # Check for death cross (50-day crosses below 200-day)
            elif sma50[-1] < sma200[-1] and sma50[-2] >= sma200[-2]:
                summary.append("Death Cross detected - a strong bearish signal.")
    
    # Bollinger Bands summary
    if bollinger_bands["upper"] is not None:
        position = bollinger_bands["position"]
        if position > 0.8:
            bb_desc = "Price near upper Bollinger Band suggests overbought conditions."
            summary.append(bb_desc)
        elif position < 0.2:
            bb_desc = "Price near lower Bollinger Band suggests oversold conditions."
            summary.append(bb_desc)
    
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
    Compare performance of multiple crypto assets with simplified insights for average investors.
    
    Args:
        token_symbols: List of token symbols (e.g. ["BTC", "ETH", "SOL"])
        candle_interval: Time interval for candles (1h, 1d, 1w)
        num_candles: Number of historical candles to analyze
        
    Returns:
        Dictionary with individual asset analyses, comparative metrics, and investment insights
    """
    results = {}
    detailed_results = {}
    
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
            
            # Extract price data
            price_data = get_binance_price_history.invoke(
                {
                    "token_symbol": token_symbol,
                    "candle_interval": candle_interval,
                    "num_candles": num_candles,
                }
            )
            
            # Calculate actual price change
            price_history = price_data.get("data", [])
            if len(price_history) >= 2:
                start_price = float(price_history[0][4])  # Close price of first candle
                end_price = float(price_history[-1][4])   # Close price of last candle
                price_change_pct = ((end_price / start_price) - 1) * 100
            else:
                price_change_pct = 0
                
            # Store useful metrics for average investors
            results[token_symbol] = {
                "current_price": analysis.get("current_price"),
                "price_change_pct": round(price_change_pct, 2),
                "moving_averages": {
                    "short_term": analysis.get("moving_averages", {}).get("short_trend"),
                    "long_term": analysis.get("moving_averages", {}).get("long_trend"),
                },
                "volatility": analysis.get("token_metrics", {}).get("volatility"),
                "volume_trend": analysis.get("technical_indicators", {})
                    .get("volume_analysis", {}).get("trend"),
                "key_signals": []
            }
            
            # Add key signals for average investors
            moving_averages = analysis.get("moving_averages", {})
            if moving_averages.get("golden_cross") == "Yes":
                results[token_symbol]["key_signals"].append("BULLISH: Golden Cross detected")
            if moving_averages.get("death_cross") == "Yes":
                results[token_symbol]["key_signals"].append("BEARISH: Death Cross detected")
                
            # Add Bollinger Band signals
            bb = analysis.get("technical_indicators", {}).get("bollinger_bands", {})
            if bb.get("position") is not None:
                position = bb.get("position")
                if position > 0.8:
                    results[token_symbol]["key_signals"].append("CAUTION: Potentially overbought")
                elif position < 0.2:
                    results[token_symbol]["key_signals"].append("OPPORTUNITY: Potentially oversold")
                    
        except Exception as e:
            results[token_symbol] = {
                "error": f"Error analyzing {token_symbol}: {str(e)}"
            }
    
    # Step 2: Calculate investor-friendly comparative metrics
    valid_tokens = {
        symbol: data for symbol, data in results.items() 
        if "error" not in data and data.get("current_price") is not None
    }
    
    comparative_analysis = {}
    
    if valid_tokens:
        # Performance ranking
        performance_ranking = sorted(
            valid_tokens.items(),
            key=lambda x: x[1].get("price_change_pct", 0),
            reverse=True
        )
        
        # Volatility ranking (lower is better for risk-averse investors)
        volatility_ranking = sorted(
            valid_tokens.items(),
            key=lambda x: x[1].get("volatility", float("inf"))
        )
        
        # Calculate simple risk-adjusted returns
        risk_adjusted = {}
        for symbol, data in valid_tokens.items():
            volatility = data.get("volatility", 0)
            price_change = data.get("price_change_pct", 0)
            
            if volatility and volatility > 0:
                risk_adjusted[symbol] = price_change / volatility
            else:
                risk_adjusted[symbol] = 0
                
        risk_adjusted_ranking = sorted(
            risk_adjusted.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create a more beginner-friendly comparative analysis
        comparative_analysis = {
            "best_performer": {
                "token": performance_ranking[0][0],
                "return": f"{performance_ranking[0][1].get('price_change_pct', 0):.2f}%"
            } if performance_ranking else None,
            
            "worst_performer": {
                "token": performance_ranking[-1][0],
                "return": f"{performance_ranking[-1][1].get('price_change_pct', 0):.2f}%"
            } if performance_ranking else None,
            
            "most_stable": {
                "token": volatility_ranking[0][0],
                "volatility": volatility_ranking[0][1].get("volatility", 0)
            } if volatility_ranking else None,
            
            "best_risk_adjusted": {
                "token": risk_adjusted_ranking[0][0],
                "value": round(risk_adjusted_ranking[0][1], 2)
            } if risk_adjusted_ranking else None,
            
            "all_tokens_ranked_by_performance": [
                {
                    "token": symbol,
                    "return": f"{data.get('price_change_pct', 0):.2f}%",
                    "short_term_trend": data.get("moving_averages", {}).get("short_term", "Neutral"),
                    "key_signals": data.get("key_signals", [])
                }
                for symbol, data in performance_ranking
            ]
        }
        
        # Calculate average market trend
        bullish_count = sum(
            1 for symbol, data in valid_tokens.items()
            if data.get("moving_averages", {}).get("long_term") == "Bullish"
        )
        
        bearish_count = sum(
            1 for symbol, data in valid_tokens.items()
            if data.get("moving_averages", {}).get("long_term") == "Bearish"
        )
        
        if bullish_count > bearish_count:
            market_trend = "Bullish"
        elif bearish_count > bullish_count:
            market_trend = "Bearish"
        else:
            market_trend = "Mixed"
            
        comparative_analysis["market_trend"] = market_trend
    
    # Step 3: Create actionable insights for investors
    investment_insights = []
    
    if comparative_analysis:
        # Add general market insight
        market_trend = comparative_analysis.get("market_trend")
        if market_trend == "Bullish":
            investment_insights.append(
                "Overall market trend appears bullish across analyzed tokens."
            )
        elif market_trend == "Bearish":
            investment_insights.append(
                "Overall market trend appears bearish across analyzed tokens."
            )
        else:
            investment_insights.append(
                "Market shows mixed signals, consider diversification."
            )
            
        # Add risk-based suggestions
        best_risk_adjusted = comparative_analysis.get("best_risk_adjusted")
        if best_risk_adjusted:
            investment_insights.append(
                f"{best_risk_adjusted['token']} shows the best balance of return vs risk."
            )
        
        # Add momentum-based suggestion
        best_performer = comparative_analysis.get("best_performer")
        if best_performer:
            token = best_performer['token']
            trend = valid_tokens[token].get("moving_averages", {}).get("short_term")
            if trend == "Bullish":
                investment_insights.append(
                    f"{token} shows strong momentum with the highest return and a bullish trend."
                )
                
        # Add stability suggestion
        most_stable = comparative_analysis.get("most_stable")
        if most_stable:
            investment_insights.append(
                f"{most_stable['token']} has the lowest volatility, potentially suitable for risk-averse investors."
            )
            
    # Format period in user-friendly terms
    period_text = f"{num_candles} "
    if candle_interval == "1d":
        period_text += "days"
    elif candle_interval == "1h":
        period_text += "hours"
    elif candle_interval == "1w":
        period_text += "weeks"
    
    # Step 4: Return the final results
    return {
        "individual_tokens": results,
        "comparative_analysis": comparative_analysis,
        "investment_insights": investment_insights,
        "period": period_text,
        "disclaimer": "This analysis is for informational purposes only. Past performance is not indicative of future results. Always do your own research before investing."
    }


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
