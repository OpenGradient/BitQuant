from typing import Dict, Any, List, Tuple, Optional
from langchain_core.tools import tool
from langgraph.graph.graph import RunnableConfig
import traceback
import numpy as np
from enum import StrEnum
import os
import csv
import requests
from time import sleep
from datetime import datetime, timedelta, UTC
from cachetools import TTLCache

from server.metrics import track_tool_usage
from api.api_types import WalletTokenHolding

class CandleInterval(StrEnum):
    """Candle interval options for price data"""

    DAY = "1d"
    HOUR = "1h"


# Initialize API settings
COINGECKO_API_KEY = os.environ.get("COINGECKO_API_KEY", "")
COINGECKO_BASE_URL = "https://pro-api.coingecko.com/api/v3"
COINGECKO_HEADERS = {
    "accept": "application/json",
    "x-cg-pro-api-key": COINGECKO_API_KEY,
}

# Cache for price data (10-minute TTL)
price_data_cache = TTLCache(maxsize=1000, ttl=600)

# Common token mappings for better user experience
PREFERRED_TOKEN_IDS = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "link": "chainlink",
    "uni": "uniswap",
    "aave": "aave",
    "matic": "polygon",
    "sol": "solana",
    "doge": "dogecoin",
    "shib": "shiba-inu",
    "ada": "cardano",
    "dot": "polkadot",
    "avax": "avalanche-2",
    "bnb": "binancecoin",
    "usdt": "tether",
    "usdc": "usd-coin",
    "dai": "dai",
}

# Global maps for token resolution
SYMBOL_TO_ID_MAP = {}
ID_TO_NAME_MAP = {}
NAME_TO_ID_MAP = {}

def timestamp_to_date(timestamp: int) -> str:
    """Convert Unix timestamp to human-readable date string"""
    # Convert milliseconds to seconds if needed
    if timestamp > 10000000000:  # Threshold for millisecond timestamps
        timestamp = timestamp / 1000

    dt = datetime.fromtimestamp(timestamp, tz=UTC)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def date_to_timestamp(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> int:
    """Convert date components to Unix timestamp in seconds"""
    dt = datetime(year, month, day, hour, minute, second, tzinfo=UTC)
    return int(dt.timestamp())


def load_coingecko_id_mappings() -> Tuple[Dict, Dict, Dict, str]:
    """
    Load CoinGecko ID mappings from CSV file.

    Returns:
        Tuple containing symbol_to_id_map, id_to_name_map, name_to_id_map, and any error message
    """
    symbol_to_id_map = {}
    id_to_name_map = {}
    name_to_id_map = {}
    error_msg = None

    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "static",
        "coingecko_ids.csv",
    )

    try:
        with open(csv_path, "r", encoding="utf-8") as file:
            # Skip the first two header rows
            next(file)  # Skip the note about API
            next(file)  # Skip the column headers

            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) >= 3:
                    coingecko_id = row[0].strip()
                    symbol = row[1].strip().lower()
                    name = row[2].strip().lower()

                    # Create symbol->ID mapping
                    if symbol:
                        # Handle multiple symbols that map to the same ID
                        if symbol in symbol_to_id_map:
                            if not isinstance(symbol_to_id_map[symbol], list):
                                symbol_to_id_map[symbol] = [symbol_to_id_map[symbol]]
                            symbol_to_id_map[symbol].append(coingecko_id)
                        else:
                            symbol_to_id_map[symbol] = coingecko_id

                    # Create ID->name mapping
                    id_to_name_map[coingecko_id] = name

                    # Create name->ID mapping (case-insensitive)
                    name_to_id_map[name] = coingecko_id
    except Exception as e:
        error_msg = f"Error loading CoinGecko IDs: {str(e)}"

    return symbol_to_id_map, id_to_name_map, name_to_id_map, error_msg


# Load token mappings at module import time
SYMBOL_TO_ID_MAP, ID_TO_NAME_MAP, NAME_TO_ID_MAP, csv_load_error = (
    load_coingecko_id_mappings()
)


def make_coingecko_request(
    url: str,
    params: Optional[Dict] = None,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
) -> Any:
    """Make request to CoinGecko API with retry logic"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(
                url, params=params, headers=COINGECKO_HEADERS, timeout=10
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                sleep(retry_after)
                retry_count += 1
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException:
            retry_count += 1
            wait_time = backoff_factor * (2 ** (retry_count - 1))
            sleep(wait_time)

    # All retries failed
    raise Exception(f"Failed to make request to {url} after {max_retries} attempts")


def get_coingecko_id(token_input: str) -> Tuple[str, Optional[str]]:
    """
    Resolve token symbol/name/id to valid CoinGecko ID.

    Returns:
        Tuple of (coingecko_id, error_message)
    """
    if not token_input:
        return "", "Empty token input provided"

    token_input_lower = token_input.lower()

    # Check if input is already a valid CoinGecko ID
    if token_input in ID_TO_NAME_MAP:
        return token_input, None

    # Check for preferred mapping
    if token_input_lower in PREFERRED_TOKEN_IDS:
        return PREFERRED_TOKEN_IDS[token_input_lower], None

    # Try symbol match (case-insensitive)
    if token_input_lower in SYMBOL_TO_ID_MAP:
        symbol_match = SYMBOL_TO_ID_MAP[token_input_lower]

        # Handle multiple IDs matching a symbol
        if isinstance(symbol_match, list):
            return (
                symbol_match[0],
                f"Multiple CoinGecko IDs found for symbol {token_input_lower}: {symbol_match}. Using {symbol_match[0]}",
            )

        return symbol_match, None

    # Try name match (case-insensitive)
    if token_input_lower in NAME_TO_ID_MAP:
        return NAME_TO_ID_MAP[token_input_lower], None

    # Fallback to lowercase input
    return (
        token_input_lower,
        f"No exact CoinGecko ID match found for '{token_input}'. Using '{token_input_lower}' as a fallback",
    )


def format_ohlc_data(ohlc_data: List) -> List:
    """Format OHLC data from CoinGecko API response"""
    formatted_data = []
    for candle in ohlc_data:
        if len(candle) >= 5:
            timestamp, open_price, high, low, close = candle
            # Convert milliseconds to seconds if needed
            if timestamp > 10000000000:
                timestamp = timestamp / 1000

            # Ensure all price values are numeric
            open_price = float(open_price) if open_price is not None else None
            high = float(high) if high is not None else None
            low = float(low) if low is not None else None
            close = float(close) if close is not None else None

            formatted_data.append([timestamp, open_price, high, low, close])

    # Sort by timestamp
    formatted_data.sort(key=lambda x: x[0])
    return formatted_data


def get_coin_suggestions(token_symbol: str, token_id: str) -> Optional[str]:
    """Generate suggestions for similar coin IDs when a match fails"""
    suggestions = []
    input_lower = token_symbol.lower()

    for symbol in SYMBOL_TO_ID_MAP:
        if input_lower in symbol or symbol in input_lower:
            coin_id = SYMBOL_TO_ID_MAP[symbol]
            if isinstance(coin_id, list):
                for cid in coin_id:
                    coin_name = ID_TO_NAME_MAP.get(cid, symbol)
                    suggestions.append(f"{symbol} -> {cid} ({coin_name})")
            else:
                coin_name = ID_TO_NAME_MAP.get(coin_id, symbol)
                suggestions.append(f"{symbol} -> {coin_id} ({coin_name})")

            if len(suggestions) >= 5:
                break

    if suggestions:
        return f"Suggested coins: {', '.join(suggestions[:5])}"
    return None

@tool("get_coingecko_categories_list")
def get_coingecko_categories_list() -> list:
    """
    Fetch the list of all coin categories (category_id and name) from CoinGecko.
    Useful for letting users sift through available categories.
    """
    url = f"{COINGECKO_BASE_URL}/coins/categories/list"
    try:
        categories = make_coingecko_request(url, params=None)
        return categories
    except Exception as e:
        return [{"error": str(e)}]

@tool("get_coingecko_categories_info")
def get_coingecko_categories_info() -> list:
    """
    Fetch detailed info for all coin categories from CoinGecko.
    Each entry includes category_id, name, market cap, volume, and top coins.
    """
    url = f"{COINGECKO_BASE_URL}/coins/categories"
    try:
        categories_info = make_coingecko_request(url, params=None)
        return categories_info
    except Exception as e:
        return [{"error": str(e)}]

@tool("get_trending_tokens_in_category")
def get_trending_tokens_in_category(category_id: str, page: int = 1, per_page: int = 10) -> list:
    """
    Fetch the top tokens in a CoinGecko category, sorted by market cap (i.e., trending by market cap).
    """
    url = f"{COINGECKO_BASE_URL}/coins/markets"
    params = {
        "vs_currency": "usd",
        "category": category_id,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false"
    }
    try:
        response = make_coingecko_request(url, params=params)
        return response
    except Exception as e:
        return [{"error": str(e)}]

@tool("get_coingecko_category_info")
def get_coingecko_category_info(category_id: str, page: int = 1, per_page: int = 10) -> list:
    """
    Fetch coins in a specific CoinGecko category by category_id.
    """
    url = f"{COINGECKO_BASE_URL}/coins/markets"
    params = {
        "vs_currency": "usd",
        "category": category_id,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false"
    }
    try:
        response = make_coingecko_request(url, params=params)
        return response
    except Exception as e:
        return [{"error": str(e)}]

@tool()
def get_coingecko_current_price(
    token_symbol: str, vs_currency: str = "usd", days: int = 1
) -> Dict[str, Any]:
    """
    Retrieve snapshot OHLC price data for a token over the specified number of days.
    """
    try:
        token_id, error_message = get_coingecko_id(token_symbol)
        if not token_id:
            return {"error": f"Failed to resolve CoinGecko ID for {token_symbol}"}

        # Construct request URL
        url = f"{COINGECKO_BASE_URL}/coins/{token_id}/ohlc"

        # Valid days values for the API
        valid_days = [1, 7, 14, 30, 90, 180, 365, "max"]
        if days not in valid_days:
            days = 1

        params = {"vs_currency": vs_currency, "days": days}

        # Make the request
        ohlc_data = make_coingecko_request(url, params=params)

        if not isinstance(ohlc_data, list) or len(ohlc_data) == 0:
            return {
                "error": "CoinGecko API returned invalid or empty data",
                "token_symbol": token_symbol,
                "token_id": token_id,
            }

        # Format the response data
        formatted_data = format_ohlc_data(ohlc_data)

        result = {
            "token_symbol": token_symbol,
            "vs_currency": vs_currency,
            "days": days,
            "num_candles": len(formatted_data),
            "data": formatted_data,
            "columns": ["timestamp", "open", "high", "low", "close"],
            "readable_dates": {
                "start": (
                    timestamp_to_date(formatted_data[0][0]) if formatted_data else None
                ),
                "end": (
                    timestamp_to_date(formatted_data[-1][0]) if formatted_data else None
                ),
            },
        }

        if error_message:
            result["warning"] = error_message

        return result

    except Exception as e:
        return {
            "error": f"Error fetching CoinGecko snapshot data for {token_symbol}: {str(e)}",
            "token_symbol": token_symbol,
            "traceback": traceback.format_exc(),
        }


def get_coingecko_price_data(
    token_symbol: str,
    candle_interval: CandleInterval,
    from_timestamp: Optional[int] = None,
    to_timestamp: Optional[int] = None,
    num_candles: Optional[int] = None,
    vs_currency: str = "usd",
) -> Dict[str, Any]:
    """
    Retrieve historical price data for a token using CoinGecko.

    Args:
        token_symbol: The token symbol to fetch data for
        candle_interval: The interval between candles (DAY or HOUR)
        from_timestamp: Optional start timestamp in seconds
        to_timestamp: Optional end timestamp in seconds
        num_candles: Optional number of candles to fetch (used if timestamps not provided)
        vs_currency: The currency to compare against (default: usd)

    Returns:
        Dict containing price data and metadata
    """
    # Check cache if using num_candles (history mode)
    if num_candles is not None and from_timestamp is None and to_timestamp is None:
        cache_key = f"{token_symbol}_{candle_interval}_{num_candles}"
        if cache_key in price_data_cache:
            return price_data_cache[cache_key]

    try:
        # Resolve token ID
        token_id, error_message = get_coingecko_id(token_symbol)
        if not token_id:
            return {"error": f"Failed to resolve CoinGecko ID for {token_symbol}"}

        # Validate and set up time parameters
        now = int(datetime.now(UTC).timestamp())

        if from_timestamp is None or to_timestamp is None:
            if num_candles is None:
                num_candles = 90  # Default to 90 candles

            # Calculate time range based on interval
            if candle_interval == CandleInterval.HOUR:
                seconds_per_candle = 60 * 60
                max_candles = 744  # API limit for hourly data
            else:
                seconds_per_candle = 60 * 60 * 24
                max_candles = 180  # API limit for daily data

            num_candles = min(num_candles, max_candles)
            time_range_seconds = num_candles * seconds_per_candle
            from_timestamp = now - time_range_seconds
            to_timestamp = now

        # Validate timestamp limits
        min_timestamp = (
            1518147224  # February 9, 2018 - CoinGecko data availability limit
        )
        if from_timestamp < min_timestamp:
            from_timestamp = min_timestamp

        # Construct request URL and parameters
        url = f"{COINGECKO_BASE_URL}/coins/{token_id}/ohlc/range"
        interval = "daily" if candle_interval == CandleInterval.DAY else "hourly"

        params = {
            "vs_currency": vs_currency,
            "from": from_timestamp,
            "to": to_timestamp,
            "interval": interval,
        }

        # Make the request
        ohlc_data = make_coingecko_request(url, params=params)

        # Handle API errors
        if isinstance(ohlc_data, dict) and "error" in ohlc_data:
            if "Could not find coin" in ohlc_data["error"]:
                suggestions = get_coin_suggestions(token_symbol, token_id)
                error_msg = f"CoinGecko API couldn't find coin with ID '{token_id}'."
                if suggestions:
                    error_msg += f" {suggestions}"
                return {
                    "error": error_msg,
                    "token_symbol": token_symbol,
                    "attempted_id": token_id,
                }
            return {
                "error": f"CoinGecko API returned an error: {ohlc_data['error']}",
                "token_symbol": token_symbol,
                "attempted_id": token_id,
            }

        # Validate response data
        if not isinstance(ohlc_data, list) or len(ohlc_data) == 0:
            return {
                "error": "CoinGecko API returned invalid or empty data",
                "token_symbol": token_symbol,
                "attempted_id": token_id,
            }

        # Format the response data
        formatted_data = format_ohlc_data(ohlc_data)

        # If using num_candles, limit to requested number of candles
        if num_candles is not None and len(formatted_data) > num_candles:
            formatted_data = formatted_data[-num_candles:]

        # Prepare result
        result = {
            "token_symbol": token_symbol,
            "candle_interval": candle_interval,
            "num_candles": len(formatted_data),
            "data": formatted_data,
            "columns": ["timestamp", "open", "high", "low", "close"],
            "readable_dates": {
                "start": (
                    timestamp_to_date(formatted_data[0][0]) if formatted_data else None
                ),
                "end": (
                    timestamp_to_date(formatted_data[-1][0]) if formatted_data else None
                ),
                "requested_range": {
                    "from": timestamp_to_date(from_timestamp),
                    "to": timestamp_to_date(to_timestamp),
                },
            },
        }

        if error_message:
            result["warning"] = error_message

        # Cache the result if using num_candles mode
        if num_candles is not None and from_timestamp is None and to_timestamp is None:
            cache_key = f"{token_symbol}_{candle_interval}_{num_candles}"
            price_data_cache[cache_key] = result

        return result

    except Exception as e:
        return {
            "error": f"Error fetching CoinGecko data for {token_symbol}: {str(e)}",
            "token_symbol": token_symbol,
            "traceback": traceback.format_exc(),
        }


@tool()
@track_tool_usage("analyze_price_trend")
def analyze_price_trend(token_symbol: str, num_days: int = 90) -> Dict[str, Any]:
    """
    Analyzes price trend for a token including moving averages, volatility metrics,
    and enhanced technical indicators over the specified time period.
    """
    try:
        # Get the price history first
        price_data = get_coingecko_price_data(
            token_symbol=token_symbol,
            candle_interval=CandleInterval.DAY,
            num_candles=num_days,
        )

        # Check for errors in price data
        if "error" in price_data:
            return {"error": price_data["error"]}

        # Extract relevant data for analysis
        raw_data = price_data["data"]

        # Process price data
        close_prices = [float(candle[4]) for candle in raw_data]
        open_prices = [float(candle[1]) for candle in raw_data]
        high_prices = [float(candle[2]) for candle in raw_data]
        low_prices = [float(candle[3]) for candle in raw_data]

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
            std_dev = (
                sum((price - middle_band) ** 2 for price in close_prices[-20:]) / 20
            ) ** 0.5
            # Upper and lower bands
            upper_band = middle_band + (2 * std_dev)
            lower_band = middle_band - (2 * std_dev)

            # Calculate normalized width and position safely to avoid division by zero
            width = 0
            if middle_band != 0:
                width = (upper_band - lower_band) / middle_band

            position = 0.5  # Default to middle if bands are identical
            if upper_band != lower_band:
                position = (close_prices[-1] - lower_band) / (upper_band - lower_band)

            bollinger_bands = {
                "upper": round(upper_band, 2),
                "middle": round(middle_band, 2),
                "lower": round(lower_band, 2),
                "width": round(width, 4),  # Normalized width
                "position": round(position, 2),
            }

        # 4. Fibonacci Retracement Levels (based on recent high and low)
        fibonacci = {}
        if len(close_prices) >= 10:
            recent_high = (
                max(high_prices[-20:]) if len(high_prices) >= 20 else max(high_prices)
            )
            recent_low = (
                min(low_prices[-20:]) if len(low_prices) >= 20 else min(low_prices)
            )
            price_range = recent_high - recent_low

            fibonacci = {
                "levels": {
                    "0.0": round(recent_low, 2),
                    "0.236": round(recent_low + 0.236 * price_range, 2),
                    "0.382": round(recent_low + 0.382 * price_range, 2),
                    "0.5": round(recent_low + 0.5 * price_range, 2),
                    "0.618": round(recent_low + 0.618 * price_range, 2),
                    "0.786": round(recent_low + 0.786 * price_range, 2),
                    "1.0": round(recent_high, 2),
                },
                "current_position": "None",
            }

            # Identify nearest Fibonacci level to current price
            current_price = close_prices[-1]
            levels = list(fibonacci["levels"].items())
            levels.sort(key=lambda x: abs(current_price - x[1]))
            fibonacci["current_position"] = levels[0][0]

        # 9. Token-specific metrics
        token_metrics = {}
        if len(close_prices) > 0:
            current_price = close_prices[-1]

            # Calculate volatility safely
            volatility = None
            if len(close_prices) >= 7:
                max_price = max(close_prices[-7:])
                min_price = min(close_prices[-7:])
                if min_price > 0:  # Avoid division by zero
                    volatility = round(((max_price / min_price - 1) * 100), 2)

            token_metrics = {
                "price": round(current_price, 4),
                "volatility": volatility,
            }

        # Add source information to the output
        result = {
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
                "golden_cross": (
                    "Yes"
                    if sma50
                    and sma200
                    and sma50[-1] > sma200[-1]
                    and (len(sma50) > 1 and len(sma200) > 1 and sma50[-2] <= sma200[-2])
                    else "No"
                ),
                "death_cross": (
                    "Yes"
                    if sma50
                    and sma200
                    and sma50[-1] < sma200[-1]
                    and (len(sma50) > 1 and len(sma200) > 1 and sma50[-2] >= sma200[-2])
                    else "No"
                ),
                "short_trend": (
                    "Bullish"
                    if sma7 and sma20 and sma7[-1] > sma20[-1]
                    else "Bearish" if sma7 and sma20 else "Neutral"
                ),
                "long_trend": (
                    "Bullish"
                    if sma50 and sma200 and sma50[-1] > sma200[-1]
                    else "Bearish" if sma50 and sma200 else "Neutral"
                ),
            },
            "technical_indicators": {
                "bollinger_bands": bollinger_bands,
                "fibonacci": fibonacci,
            },
            "token_metrics": token_metrics,
            "analysis_summary": get_analysis_summary(
                sma7, sma20, sma50, sma200, bollinger_bands
            ),
        }

        return result

    except Exception as e:
        return {"error": f"Error analyzing price trend for {token_symbol}: {e}"}


def get_analysis_summary(sma7, sma20, sma50, sma200, bollinger_bands):
    """Generate a simple summary of the analysis results"""
    summary = []

    # Short-term trend
    if sma7 and sma20:
        if sma7[-1] > sma20[-1]:
            summary.append("Short-term moving averages indicate bullish momentum.")
        else:
            summary.append("Short-term moving averages indicate bearish momentum.")

    # Long-term trend and major crossovers
    if sma50 and sma200 and len(sma50) > 1 and len(sma200) > 1:
        # Check for golden cross (50-day crosses above 200-day)
        if sma50[-1] > sma200[-1] and sma50[-2] <= sma200[-2]:
            summary.append("Golden Cross detected - a strong bullish signal.")
        # Check for death cross (50-day crosses below 200-day)
        elif sma50[-1] < sma200[-1] and sma50[-2] >= sma200[-2]:
            summary.append("Death Cross detected - a strong bearish signal.")

    # Bollinger Bands summary
    if bollinger_bands["upper"] is not None:
        position = bollinger_bands["position"]
        if position > 0.8:
            summary.append(
                "Price near upper Bollinger Band suggests overbought conditions."
            )
        elif position < 0.2:
            summary.append(
                "Price near lower Bollinger Band suggests oversold conditions."
            )

    # Combine into a paragraph
    return " ".join(summary)


@tool()
def compare_assets(
    token_symbols: List[str], candle_interval: CandleInterval, num_candles: int
) -> Dict[str, Any]:
    """
    Compare performance of multiple crypto assets with simplified insights for average investors.
    """
    # Generate a request ID for this comparison operation
    comparison_id = f"compare_{datetime.now(UTC).strftime('%H%M%S')}"

    results = {}
    detailed_results = {}
    error_count = 0
    successful_tokens = []

    # Step 1: Collect individual asset data
    for token_symbol in token_symbols:
        try:
            # Get price data first to check for errors
            price_data = get_coingecko_price_data(
                token_symbol=token_symbol,
                candle_interval=candle_interval,
                num_candles=num_candles,
            )

            # Check for errors in price data
            if "error" in price_data:
                results[token_symbol] = {"error": price_data["error"]}
                error_count += 1
                continue

            # Now get the analysis
            analysis = analyze_price_trend(
                token_symbol=token_symbol,
                candle_interval=candle_interval,
                num_candles=num_candles,
            )

            # Skip if there was an error
            if "error" in analysis:
                results[token_symbol] = {"error": analysis["error"]}
                error_count += 1
                continue

            # Store detailed analysis results
            detailed_results[token_symbol] = analysis

            # Calculate actual price change
            price_history = price_data.get("data", [])
            price_change_pct = 0

            if len(price_history) >= 2:
                start_price = float(price_history[0][4])  # Close price of first candle
                end_price = float(price_history[-1][4])  # Close price of last candle

                # Calculate percentage change safely
                if start_price > 0:  # Avoid division by zero
                    price_change_pct = ((end_price / start_price) - 1) * 100

            # Store useful metrics for average investors
            results[token_symbol] = {
                "current_price": analysis.get("current_price"),
                "price_change_pct": round(price_change_pct, 2),
                "moving_averages": {
                    "short_term": analysis.get("moving_averages", {}).get(
                        "short_trend"
                    ),
                    "long_term": analysis.get("moving_averages", {}).get("long_trend"),
                },
                "volatility": analysis.get("token_metrics", {}).get("volatility"),
                "key_signals": [],
            }

            # Add key signals for average investors
            moving_averages = analysis.get("moving_averages", {})
            if moving_averages.get("golden_cross") == "Yes":
                results[token_symbol]["key_signals"].append(
                    "BULLISH: Golden Cross detected"
                )
            if moving_averages.get("death_cross") == "Yes":
                results[token_symbol]["key_signals"].append(
                    "BEARISH: Death Cross detected"
                )

            # Add Bollinger Band signals
            bb = analysis.get("technical_indicators", {}).get("bollinger_bands", {})
            if bb.get("position") is not None:
                position = bb.get("position")
                if position > 0.8:
                    results[token_symbol]["key_signals"].append(
                        "CAUTION: Potentially overbought"
                    )
                elif position < 0.2:
                    results[token_symbol]["key_signals"].append(
                        "OPPORTUNITY: Potentially oversold"
                    )

            successful_tokens.append(token_symbol)

        except Exception as e:
            results[token_symbol] = {
                "error": f"Error analyzing {token_symbol}: {str(e)}"
            }
            error_count += 1

    # If all tokens had errors, return a general error
    if error_count == len(token_symbols):
        return {
            "error": "Could not analyze any of the requested tokens. The CoinGecko API may be rate-limited or temporarily unavailable. Please try again later."
        }

    # Step 2: Calculate investor-friendly comparative metrics
    valid_tokens = {
        symbol: data
        for symbol, data in results.items()
        if "error" not in data and data.get("current_price") is not None
    }

    comparative_analysis = {}

    if valid_tokens:
        # Performance ranking
        performance_ranking = sorted(
            valid_tokens.items(),
            key=lambda x: x[1].get("price_change_pct", 0),
            reverse=True,
        )

        # Volatility ranking (lower is better for risk-averse investors)
        volatility_ranking = sorted(
            [
                (symbol, data)
                for symbol, data in valid_tokens.items()
                if data.get("volatility", 0) is not None
            ],
            key=lambda x: x[1].get("volatility", float("inf")),
        )

        # Calculate simple risk-adjusted returns
        risk_adjusted = {}
        for symbol, data in valid_tokens.items():
            volatility = data.get("volatility")
            price_change = data.get("price_change_pct", 0)

            # Only calculate if volatility exists and is greater than zero
            if volatility is not None and volatility > 0:
                risk_adjusted[symbol] = price_change / volatility
            # Skip tokens with no volatility data or zero volatility

        risk_adjusted_ranking = sorted(
            risk_adjusted.items(), key=lambda x: x[1], reverse=True
        )

        # Create a more beginner-friendly comparative analysis
        best_performer = None
        if performance_ranking:
            best_performer = {
                "token": performance_ranking[0][0],
                "return": f"{performance_ranking[0][1].get('price_change_pct', 0):.2f}%",
            }

        worst_performer = None
        if performance_ranking and len(performance_ranking) > 1:
            worst_performer = {
                "token": performance_ranking[-1][0],
                "return": f"{performance_ranking[-1][1].get('price_change_pct', 0):.2f}%",
            }

        most_stable = None
        if volatility_ranking:
            most_stable = {
                "token": volatility_ranking[0][0],
                "volatility": volatility_ranking[0][1].get("volatility", 0),
            }

        best_risk_adjusted = None
        if risk_adjusted_ranking:
            best_risk_adjusted = {
                "token": risk_adjusted_ranking[0][0],
                "value": (
                    round(risk_adjusted_ranking[0][1], 2)
                    if risk_adjusted_ranking[0][1] != float("inf")
                    else "High"
                ),
            }

        comparative_analysis = {
            "best_performer": best_performer,
            "worst_performer": worst_performer,
            "most_stable": most_stable,
            "best_risk_adjusted": best_risk_adjusted,
            "all_tokens_ranked_by_performance": [
                {
                    "token": symbol,
                    "return": f"{data.get('price_change_pct', 0):.2f}%",
                    "short_term_trend": data.get("moving_averages", {}).get(
                        "short_term", "Neutral"
                    ),
                    "key_signals": data.get("key_signals", []),
                }
                for symbol, data in performance_ranking
            ],
        }

        # Calculate average market trend with safety checks
        bullish_count = sum(
            1
            for symbol, data in valid_tokens.items()
            if data.get("moving_averages", {}).get("long_term") == "Bullish"
        )

        bearish_count = sum(
            1
            for symbol, data in valid_tokens.items()
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
        if best_risk_adjusted:
            investment_insights.append(
                f"{best_risk_adjusted['token']} shows the best balance of return vs risk."
            )

        # Add momentum-based suggestion
        if best_performer:
            token = best_performer["token"]
            trend = valid_tokens[token].get("moving_averages", {}).get("short_term")
            if trend == "Bullish":
                investment_insights.append(
                    f"{token} shows strong momentum with the highest return and a bullish trend."
                )

        # Add stability suggestion
        if most_stable:
            investment_insights.append(
                f"{most_stable['token']} has the lowest volatility, potentially suitable for risk-averse investors."
            )

        # Format period in user-friendly terms
        period_text = f"{num_candles} "
        if candle_interval == CandleInterval.DAY:
            period_text += "days"
        elif candle_interval == CandleInterval.HOUR:
            period_text += "hours"

        # Return the final results
        return {
            "individual_tokens": results,
            "comparative_analysis": comparative_analysis,
            "investment_insights": investment_insights,
            "period": period_text,
            "error_count": error_count,
            "total_tokens": len(token_symbols),
            "successful_tokens": len(successful_tokens),
        }
    else:
        return {
            "error": "Could not generate comparative analysis due to insufficient valid data.",
            "individual_tokens": results,
            "error_count": error_count,
            "total_tokens": len(token_symbols),
        }


@tool()
def max_drawdown_for_token(
    token_symbol: str,
    candle_interval: CandleInterval = CandleInterval.DAY,
    num_candles: int = 90,
) -> Dict[str, Any]:
    """
    Calculates the maximum drawdown for a cryptocurrency using CoinGecko price data
    """
    try:
        # Get price data from CoinGecko
        price_data = get_coingecko_price_data(
            token_symbol=token_symbol,
            candle_interval=candle_interval,
            num_candles=num_candles,
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
    Provides a comprehensive analysis of a crypto wallet portfolio with investor-friendly insights and recommendations.
    """
    try:
        tokens: List[WalletTokenHolding] = config["configurable"]["tokens"]

        # Fetch price data for each asset
        all_price_data = []
        valid_symbols = []
        valid_quantities = []
        error_symbols = []

        for i, token in enumerate(tokens):
            # Handle both WalletTokenHolding objects and dicts
            if hasattr(token, "symbol"):
                symbol = token.symbol
                amount = token.amount
            else:
                # Handle dict format
                symbol = token.get("symbol")
                amount = token.get("amount")

            price_data = get_coingecko_price_data(
                token_symbol=symbol,
                candle_interval=candle_interval,
                num_candles=num_candles,
            )

            if "error" in price_data:
                error_symbols.append(symbol)
                continue  # Skip this token but continue with others

            # Extract closing prices
            close_prices = [float(candle[4]) for candle in price_data["data"]]
            all_price_data.append(close_prices)
            valid_symbols.append(symbol)
            valid_quantities.append(amount)

        if not all_price_data:
            return {
                "error": "Could not fetch price data for any of the tokens in your wallet. Please try with custom symbols that are available on CoinGecko."
            }

        # Convert to numpy arrays and transpose to get [time][asset] format
        prices = np.array(all_price_data).T
        holding_qty = np.array(valid_quantities)

        # Format asset names for output
        asset_names = [symbol for symbol in valid_symbols]

        # Calculate portfolio values over time
        weighted_values = holding_qty * prices
        portfolio_values = weighted_values.sum(axis=1)

        # Calculate allocation percentages
        latest_values = holding_qty * prices[-1]
        total_value = latest_values.sum()
        allocations = latest_values / total_value

        # Calculate returns (daily, weekly, monthly)
        daily_returns = portfolio_values[1:] / portfolio_values[:-1] - 1

        # Weekly returns (if we have sufficient data)
        weekly_returns = None
        if len(portfolio_values) >= 7:
            weekly_indices = list(range(0, len(portfolio_values), 7))
            if len(weekly_indices) >= 2:
                weekly_values = portfolio_values[weekly_indices]
                weekly_returns = weekly_values[1:] / weekly_values[:-1] - 1

        # Monthly returns (if we have sufficient data)
        monthly_returns = None
        if len(portfolio_values) >= 30:
            monthly_indices = list(range(0, len(portfolio_values), 30))
            if len(monthly_indices) >= 2:
                monthly_values = portfolio_values[monthly_indices]
                monthly_returns = monthly_values[1:] / monthly_values[:-1] - 1

        # Calculate drawdown
        rolling_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (rolling_max - portfolio_values) / rolling_max
        max_dd = float(drawdowns.max())

        # Calculate when max drawdown occurred
        max_dd_idx = np.argmax(drawdowns)
        peak_idx = np.argmax(portfolio_values[: max_dd_idx + 1])
        max_dd_duration = max_dd_idx - peak_idx

        # Calculate recovery after max drawdown
        if max_dd_idx < len(portfolio_values) - 1:
            recovery_pct = (
                portfolio_values[-1] / portfolio_values[max_dd_idx] - 1
            ) * 100
        else:
            recovery_pct = 0

        # Asset allocation summary with performance
        asset_allocation = []
        for i, asset in enumerate(asset_names):
            # Calculate individual asset performance
            asset_return = (prices[-1, i] / prices[0, i] - 1) * 100
            asset_volatility = np.std(prices[1:, i] / prices[:-1, i] - 1) * 100

            # Calculate relative strength vs. portfolio
            relative_return = asset_return - (
                (portfolio_values[-1] / portfolio_values[0] - 1) * 100
            )

            asset_allocation.append(
                {
                    "asset": asset,
                    "quantity": float(holding_qty[i]),
                    "current_price": float(prices[-1, i]),
                    "value": float(latest_values[i]),
                    "allocation_percent": f"{allocations[i] * 100:.2f}%",
                    "performance": {
                        "return_percent": f"{asset_return:.2f}%",
                        "volatility": f"{asset_volatility:.2f}%",
                        "relative_to_portfolio": f"{relative_return:+.2f}%",
                    },
                }
            )

        # Sort asset allocation by value (descending)
        asset_allocation = sorted(
            asset_allocation, key=lambda x: x["value"], reverse=True
        )

        # Calculate diversification score (0-100)
        # Higher when allocation is more even across assets
        herfindahl_index = np.sum(allocations**2)
        diversification_score = (1 - herfindahl_index) * 100

        # Risk assessment
        portfolio_volatility = float(daily_returns.std() * 100)
        annualized_volatility = portfolio_volatility * np.sqrt(252)

        # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = None
        if portfolio_volatility > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (
                portfolio_volatility * np.sqrt(252) / 100
            )

        # Determine risk level
        risk_level = "Unknown"
        if annualized_volatility < 15:
            risk_level = "Low"
        elif annualized_volatility < 30:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Generate personalized insights
        insights = []

        # Overall performance insight
        total_return_pct = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        if total_return_pct > 0:
            insights.append(
                f"Your portfolio has gained {total_return_pct:.2f}% over the past {num_candles} {candle_interval}s."
            )
        else:
            insights.append(
                f"Your portfolio has declined {abs(total_return_pct):.2f}% over the past {num_candles} {candle_interval}s."
            )

        # Diversification insight
        if diversification_score < 30:
            insights.append(
                f"Your portfolio is highly concentrated with a diversification score of {diversification_score:.1f}/100. Consider adding more assets to reduce risk."
            )
        elif diversification_score < 60:
            insights.append(
                f"Your portfolio has moderate diversification ({diversification_score:.1f}/100). Adding 1-2 uncorrelated assets could improve risk-adjusted returns."
            )
        else:
            insights.append(
                f"Your portfolio is well-diversified ({diversification_score:.1f}/100), which may help protect against volatility in individual assets."
            )

        # Risk insight
        if risk_level == "High":
            insights.append(
                f"Your portfolio shows high volatility ({annualized_volatility:.2f}% annualized). This may indicate higher risk, so consider your risk tolerance."
            )

        # Top/bottom performer insight
        if asset_allocation:
            # Find best and worst performers
            sorted_by_return = sorted(
                asset_allocation,
                key=lambda x: float(
                    x["performance"]["return_percent"].replace("%", "")
                ),
                reverse=True,
            )
            best_performer = sorted_by_return[0]
            worst_performer = sorted_by_return[-1]

            insights.append(
                f"{best_performer['asset']} is your best performer with a {best_performer['performance']['return_percent']} return, while {worst_performer['asset']} has returned {worst_performer['performance']['return_percent']}."
            )

        # Drawdown insight
        if max_dd > 0.1:  # Only show if drawdown is significant (>10%)
            insights.append(
                f"Your maximum drawdown was {max_dd*100:.2f}% over {max_dd_duration} {candle_interval}s. Since then, your portfolio has recovered {recovery_pct:.2f}%."
            )

        # Get time periods in user-friendly format
        period_text = ""
        if candle_interval == CandleInterval.DAY:
            period_text = f"{num_candles} days"
        elif candle_interval == CandleInterval.HOUR:
            period_text = f"{num_candles} hours"

        # Check for missing tokens
        missing_tokens_message = ""
        if error_symbols:
            missing_tokens_message = f"Note: Could not fetch data for these tokens: {', '.join(error_symbols)}"

        return {
            "portfolio_summary": {
                "total_value": f"${total_value:.2f}",
                "asset_count": len(asset_names),
                "period_analyzed": period_text,
                "missing_tokens": missing_tokens_message,
                "performance": {
                    "initial_value": f"${float(portfolio_values[0]):.2f}",
                    "current_value": f"${float(portfolio_values[-1]):.2f}",
                    "total_return": f"{total_return_pct:.2f}%",
                    "annualized_return": (
                        f"{((1 + total_return_pct/100) ** (365/(num_candles)) - 1) * 100:.2f}%"
                        if candle_interval == CandleInterval.DAY
                        else "N/A"
                    ),
                    "max_drawdown": f"{max_dd * 100:.2f}%",
                    "best_day_return": (
                        f"{max(daily_returns) * 100:.2f}%"
                        if len(daily_returns) > 0
                        else "N/A"
                    ),
                    "worst_day_return": (
                        f"{min(daily_returns) * 100:.2f}%"
                        if len(daily_returns) > 0
                        else "N/A"
                    ),
                },
                "risk_assessment": {
                    "risk_level": risk_level,
                    "volatility_daily": f"{portfolio_volatility:.2f}%",
                    "volatility_annualized": f"{annualized_volatility:.2f}%",
                    "sharpe_ratio": f"{sharpe_ratio:.2f}" if sharpe_ratio else "N/A",
                    "diversification_score": f"{diversification_score:.1f}/100",
                    "max_drawdown": f"{max_dd * 100:.2f}%",
                },
                "asset_allocation": asset_allocation,
            },
            "personalized_insights": insights,
        }
    except Exception as e:
        return {
            "error": f"Error analyzing portfolio: {str(e)}",
            "traceback": traceback.format_exc(),
        }


@tool()
def portfolio_volatility(
    candle_interval: CandleInterval = CandleInterval.DAY,
    num_candles: int = 90,
    config: RunnableConfig = None,
) -> Dict[str, Any]:
    """
    Calculates the volatility (standard deviation of returns) of a portfolio over the specified time period. Do not pass in stablecoins.
    """
    try:
        tokens: List[WalletTokenHolding] = config["configurable"]["tokens"]

        # Fetch price data for each asset
        all_price_data = []

        for token in tokens:
            price_data = get_coingecko_price_data(
                token_symbol=token.symbol,
                candle_interval=candle_interval,
                num_candles=num_candles,
            )

            if "error" in price_data:
                return {
                    "error": f"Failed to fetch price data for {token}: {price_data['error']}"
                }

            # Extract closing prices
            close_prices = [float(candle[4]) for candle in price_data["data"]]
            all_price_data.append(close_prices)

        # Convert to numpy arrays and transpose to get [time][asset] format
        prices = np.array(all_price_data).T
        holding_qty = np.array(token.amount)

        # Calculate portfolio values
        weighted_values = holding_qty * prices
        portfolio_values = weighted_values.sum(axis=1)

        # Calculate returns and volatility
        portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1
        portfolio_sd = float(portfolio_returns.std())

        return {
            "assets": [token.symbol for token in tokens],
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
