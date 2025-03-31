from typing import Dict, Any, List
from langchain_core.tools import tool
from langgraph.graph.graph import RunnableConfig
from sklearn.linear_model import LinearRegression
import traceback
import numpy as np
import time
from enum import StrEnum
import os
import csv
import requests
import random
from time import sleep
import logging
import json
from datetime import datetime
import sys
from functools import lru_cache, wraps
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger("coingecko_api")
logger.setLevel(logging.DEBUG)

# Create a file handler that logs to a file
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'coingecko_api_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("CoinGecko API module initialized")

from api.api_types import WalletTokenHolding

from langchain_core.tools import tool

class CandleInterval(StrEnum):
    """
    One of 1d, 1h, 1w
    """

    DAY = "1d"
    HOUR = "1h"
    WEEK = "1w"

# Initialize CoinGecko API key
api_key = os.environ.get("COINGECKO_API_KEY", "")

# Log API key info (partially obscured for security)
if api_key:
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "[TOO SHORT]"
    logger.info(f"Using CoinGecko API key: {masked_key}")
else:
    logger.error("COINGECKO_API_KEY environment variable is not set or empty!")

# Set up headers for direct API calls (always use Pro API)
COINGECKO_HEADERS = {
    "accept": "application/json",
    "x-cg-pro-api-key": api_key
}

# Base URL for CoinGecko API (Pro endpoint only)
COINGECKO_BASE_URL = "https://pro-api.coingecko.com/api/v3"

# Map for converting CandleInterval to CoinGecko's 'days' parameter
INTERVAL_TO_DAYS = {
    CandleInterval.HOUR: 1,   # 1 hour intervals, fetch 1 day of data
    CandleInterval.DAY: 30,   # 1 day intervals, fetch 30 days of data
    CandleInterval.WEEK: 180, # 1 week intervals, fetch 180 days of data
}

# Create a cache for symbol to ID mappings
_SYMBOL_TO_ID_CACHE = {}

# Create a cache for price data to reduce API calls
_PRICE_DATA_CACHE = {}

# Simple function to create a TTL cache
def ttl_cache(maxsize=128, ttl=300):
    """A decorator that creates a cache with a time-to-live (TTL)."""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = datetime.now()
            
            # Check if key exists and if it's still valid
            if key in cache:
                creation_time, result = cache[key]
                if (current_time - creation_time).total_seconds() < ttl:
                    return result
            
            # If not in cache or expired, call the function
            result = func(*args, **kwargs)
            
            # Store in cache with current time
            cache[key] = (current_time, result)
            
            # Clean old items (optional, can be removed for better performance)
            for k in list(cache.keys()):
                if (current_time - cache[k][0]).total_seconds() >= ttl:
                    del cache[k]
                    
            return result
        return wrapper
    return decorator

def make_coingecko_request(url, params=None, max_retries=3, backoff_factor=0.5):
    """Helper function to make CoinGecko Pro API requests with retries and backoff."""
    # Generate request ID for tracking
    request_id = f"req_{datetime.now().strftime('%H%M%S')}_{random.randint(1000, 9999)}"
    
    # Validate API key
    if not api_key:
        logger.error(f"[{request_id}] API key is missing or empty!")
    else:
        # Log headers with masked API key
        debug_headers = COINGECKO_HEADERS.copy()
        if 'x-cg-pro-api-key' in debug_headers:
            key = debug_headers['x-cg-pro-api-key']
            debug_headers['x-cg-pro-api-key'] = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "[TOO SHORT]"
        
        logger.debug(f"[{request_id}] Request headers: {debug_headers}")
    
    # Log request details
    logger.debug(f"[{request_id}] CoinGecko Pro API Request: {url}, Params: {params}")
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Add a small random delay to avoid rate limits
            sleep_time = random.uniform(0.1, 0.5)
            logger.debug(f"[{request_id}] Sleeping for {sleep_time:.2f}s before request")
            sleep(sleep_time)
            
            # Make request with timeout
            start_time = time.time()
            response = requests.get(url, params=params, headers=COINGECKO_HEADERS, timeout=10)
            elapsed_time = time.time() - start_time
            
            # Log response details
            logger.debug(f"[{request_id}] Response received in {elapsed_time:.2f}s: Status {response.status_code}")
            
            # Check for rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"[{request_id}] Rate limited. Retrying after {retry_after} seconds...")
                sleep(retry_after)
                retry_count += 1
                continue
            
            # Check for authentication issues
            if response.status_code == 401:
                logger.error(f"[{request_id}] Authentication error: {response.text}")
                # Let's check if API key is present in environment at runtime
                runtime_key = os.environ.get("COINGECKO_API_KEY", "")
                if not runtime_key:
                    logger.error(f"[{request_id}] API key missing from environment at runtime!")
                else:
                    masked_key = f"{runtime_key[:4]}...{runtime_key[-4:]}" if len(runtime_key) > 8 else "[TOO SHORT]"
                    logger.error(f"[{request_id}] API key in environment at runtime: {masked_key}")
            
            # Check for other error codes
            if response.status_code != 200:
                logger.error(f"[{request_id}] API error: {response.status_code} - {response.text}")
                response.raise_for_status()
                
            # Parse response
            data = response.json()
            
            # Log truncated response for debugging
            data_str = str(data)
            logged_data = data_str[:500] + "..." if len(data_str) > 500 else data_str
            logger.debug(f"[{request_id}] Response data: {logged_data}")
            
            return data
        
        except requests.exceptions.RequestException as e:
            retry_count += 1
            wait_time = backoff_factor * (2 ** (retry_count - 1))
            logger.warning(f"[{request_id}] Request failed: {e}. Retrying in {wait_time:.2f}s... (Attempt {retry_count}/{max_retries})")
            sleep(wait_time)
            
    # If we get here, all retries failed
    error_msg = f"[{request_id}] Failed to make request to {url} after {max_retries} attempts"
    logger.error(error_msg)
    raise Exception(error_msg)

@ttl_cache(maxsize=100, ttl=3600)  # Cache ID lookups for 1 hour
def get_coingecko_id(token_symbol: str) -> str:
    """
    Convert a token symbol to CoinGecko ID using the CSV data file.
    
    CoinGecko IDs are used in API calls and differ from token symbols.
    
    Args:
        token_symbol: Token symbol to convert (e.g., "BTC")
        
    Returns:
        CoinGecko ID for the token (e.g., "bitcoin")
    """
    # Check cache first
    token_symbol_lower = token_symbol.lower()
    if token_symbol_lower in _SYMBOL_TO_ID_CACHE:
        return _SYMBOL_TO_ID_CACHE[token_symbol_lower]
    
    # Hardcoded common tokens to ensure tests work even without CSV
    common_tokens = {
        "btc": "bitcoin",
        "eth": "ethereum",
        "sol": "solana",
        "ada": "cardano",
        "dot": "polkadot",
        "doge": "dogecoin",
        "xrp": "ripple",
        "link": "chainlink",
        "uni": "uniswap",
        "bnb": "binancecoin",
        "usdt": "tether",
        "usdc": "usd-coin",
        "dai": "dai",
        "matic": "matic-network",
        "avax": "avalanche-2",
        "atom": "cosmos",
        "shib": "shiba-inu",
        "ltc": "litecoin"
    }
    
    if token_symbol_lower in common_tokens:
        logger.info(f"Using hardcoded mapping for {token_symbol}: {common_tokens[token_symbol_lower]}")
        _SYMBOL_TO_ID_CACHE[token_symbol_lower] = common_tokens[token_symbol_lower]
        return common_tokens[token_symbol_lower]
    
    # Path to coingecko_ids.csv
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                           'static', 'coingecko_ids.csv')
    
    try:
        # Try to find in CSV file
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # Skip the first row which contains the note
                next(reader, None)
                
                # Find all matches by symbol (case-insensitive)
                matches = []
                for row in reader:
                    if row.get('Symbol', '').lower() == token_symbol_lower:
                        matches.append(row.get('Id (API id)', ''))
                
                # If we found matches, use the first one
                if matches:
                    _SYMBOL_TO_ID_CACHE[token_symbol_lower] = matches[0]
                    return matches[0]
    except Exception as e:
        logger.error(f"Error reading coingecko_ids.csv: {e}")
    
    # If we couldn't find it in the CSV, use the lowercase symbol as fallback
    logger.warning(f"Could not find CoinGecko ID for {token_symbol} in CSV, using symbol as fallback")
    _SYMBOL_TO_ID_CACHE[token_symbol_lower] = token_symbol_lower
    return token_symbol_lower


@tool()
@ttl_cache(maxsize=100, ttl=600)  # Cache for 10 minutes
def get_coingecko_price_history(
    token_symbol: str, candle_interval: CandleInterval, num_candles: int
) -> Dict[str, Any]:
    """
    Retrieves historical price data for a token using CoinGecko API's OHLC range endpoint.
    
    Args:
        token_symbol: The token symbol (e.g., "BTC")
        candle_interval: Time interval for candles (1h, 1d, 1w)
        num_candles: Number of historical candles to retrieve
        
    Returns:
        Dictionary with price history data or error information
    """
    # Generate a request ID for tracking
    request_id = f"{token_symbol}_{datetime.now().strftime('%H%M%S')}"
    logger.info(f"[{request_id}] Getting price history for {token_symbol}, interval: {candle_interval}, candles: {num_candles}")
    
    # Min value of 2 ensures we have at least two data points for calculating trends
    num_candles = min(max(2, int(num_candles)), 1000)
    
    # Get CoinGecko ID for the token
    try:
        token_id = get_coingecko_id(token_symbol)
        logger.info(f"[{request_id}] Mapped {token_symbol} to CoinGecko ID: {token_id}")
    except Exception as e:
        error_msg = f"Failed to map {token_symbol} to CoinGecko ID: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")
        return {"error": error_msg, "token_symbol": token_symbol, "source": "coingecko"}
    
    # Set currency to USD
    vs_currency = "usd"
    
    # Calculate from and to timestamps based on candle_interval and num_candles
    now = int(datetime.now().timestamp())  # Current time in seconds
    
    # Determine interval parameter based on candle_interval
    if candle_interval == CandleInterval.HOUR:
        interval = "hourly"
        # For hourly data, go back num_candles hours
        # Maximum 31 days (744 hourly candles) for hourly interval
        seconds_per_candle = 60 * 60  # 1 hour in seconds
        max_candles = 744  # 31 days * 24 hours
    else:  # DAY or WEEK (use daily as default)
        interval = "daily"
        # For daily data, go back num_candles days
        # Maximum 180 days for daily interval
        seconds_per_candle = 60 * 60 * 24  # 1 day in seconds
        max_candles = 180  # 180 days
    
    # Limit num_candles to API capabilities
    num_candles = min(num_candles, max_candles)
    
    # Calculate from_timestamp
    time_range_seconds = num_candles * seconds_per_candle
    from_timestamp = int((datetime.now() - timedelta(seconds=time_range_seconds)).timestamp())  # From time in seconds
    
    # Ensure from_timestamp is not before the minimum timestamp supported by CoinGecko (2018-01-01)
    min_timestamp = 1514764800  # January 1, 2018 in Unix timestamp (seconds)
    if from_timestamp < min_timestamp:
        logger.warning(f"[{request_id}] Adjusted from_timestamp to minimum supported by CoinGecko: {min_timestamp}")
        from_timestamp = min_timestamp
    
    # Create a cache key
    cache_key = f"{token_id}_{vs_currency}_{from_timestamp}_{now}_{interval}"
    
    # Check if we have cached data
    if cache_key in _PRICE_DATA_CACHE:
        logger.info(f"[{request_id}] Using cached data for {token_symbol}")
        cached_data = _PRICE_DATA_CACHE[cache_key]
        return {
            "token_symbol": token_symbol,
            "token_id": token_id,
            "candle_interval": candle_interval,
            "num_candles": len(cached_data),
            "data": cached_data,
            "columns": [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
            "source": "coingecko",
            "cached": True
        }
    
    try:
        # Construct the OHLC range endpoint URL
        # Reference: https://docs.coingecko.com/reference/coins-id-ohlc-range
        ohlc_url = f"{COINGECKO_BASE_URL}/coins/{token_id}/ohlc/range"
        
        params = {
            "vs_currency": vs_currency,
            "from": from_timestamp,
            "to": now,
            "interval": interval
        }
        
        logger.info(f"[{request_id}] Requesting OHLC data from CoinGecko Pro API: {ohlc_url} with {interval} interval")
        
        # Make the request with proper error handling and retries
        ohlc_data = make_coingecko_request(ohlc_url, params=params)
        
        # Check if the API returned an error message
        if isinstance(ohlc_data, dict) and "error" in ohlc_data:
            error_msg = f"CoinGecko API returned an error: {ohlc_data['error']}"
            logger.error(f"[{request_id}] {error_msg}")
            return {"error": error_msg, "token_symbol": token_symbol, "source": "coingecko"}
            
        # Check if we got valid data
        if not isinstance(ohlc_data, list) or len(ohlc_data) == 0:
            error_msg = f"CoinGecko API returned invalid or empty data: {str(ohlc_data)[:200]}"
            logger.error(f"[{request_id}] {error_msg}")
            return {"error": error_msg, "token_symbol": token_symbol, "source": "coingecko"}
        
        logger.info(f"[{request_id}] Received {len(ohlc_data)} OHLC entries from CoinGecko Pro API")
        
        # Format data to match expected structure
        # CoinGecko OHLC format: [timestamp, open, high, low, close]
        # We need to add volume data (setting to 0 as CoinGecko OHLC doesn't provide it directly)
        formatted_data = []
        for candle in ohlc_data:
            if len(candle) >= 5:  # Ensure we have all required fields
                timestamp, open_price, high, low, close = candle
                # Add a placeholder volume value (0)
                formatted_data.append([timestamp, open_price, high, low, close, 0])
        
        logger.info(f"[{request_id}] Formatted {len(formatted_data)} valid candles")
        
        # Cache the data
        _PRICE_DATA_CACHE[cache_key] = formatted_data
        
        # Limit to requested number of candles
        formatted_data = formatted_data[-num_candles:]
        
        return {
            "token_symbol": token_symbol,
            "token_id": token_id,
            "candle_interval": candle_interval,
            "num_candles": len(formatted_data),
            "data": formatted_data,
            "columns": [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
            "source": "coingecko"
        }

    except Exception as e:
        # Get traceback for better debugging
        error_traceback = traceback.format_exc()
        error_msg = f"Error fetching CoinGecko data for {token_symbol}: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}\n{error_traceback}")
        
        return {
            "error": error_msg,
            "token_symbol": token_symbol,
            "source": "coingecko"
        }


# Alias to maintain backward compatibility with existing code
get_binance_price_history = get_coingecko_price_history


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
        price_data = get_coingecko_price_history.invoke(
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
        # Note: For CoinGecko, we don't have volume data directly from OHLC
        # We'll use zeros or try to fetch volume separately if needed
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

        # 5. Volume Analysis
        volume_analysis = {
            "trend": "Neutral",
            "avg_volume": None,
            "current_vs_avg": None,
        }
        if len(volumes) >= 7:
            avg_volume = sum(volumes[-7:]) / 7
            current_volume = volumes[-1]

            volume_trend = "Increasing" if current_volume > avg_volume else "Decreasing"
            # Check if volume confirms price trend
            price_up = close_prices[-1] > open_prices[-1]
            volume_confirms = (price_up and volume_trend == "Increasing") or (
                not price_up and volume_trend == "Decreasing"
            )

            # Calculate percent change safely
            current_vs_avg = 0
            if avg_volume > 0:  # Avoid division by zero
                current_vs_avg = (current_volume / avg_volume - 1) * 100

            volume_analysis = {
                "trend": volume_trend,
                "avg_volume": round(avg_volume, 2),
                "current_volume": round(current_volume, 2),
                "current_vs_avg": round(current_vs_avg, 2),
                "confirms_price": volume_confirms,
            }

        # 9. Token-specific metrics
        token_metrics = {}
        if len(close_prices) > 0 and len(volumes) > 0:
            current_price = close_prices[-1]
            avg_daily_volume = sum(volumes[-min(7, len(volumes)) :]) / min(
                7, len(volumes)
            )

            # Calculate volatility safely
            volatility = None
            if len(close_prices) >= 7:
                max_price = max(close_prices[-7:])
                min_price = min(close_prices[-7:])
                if min_price > 0:  # Avoid division by zero
                    volatility = round(((max_price / min_price - 1) * 100), 2)

            token_metrics = {
                "price": round(current_price, 4),
                "avg_daily_volume_usd": round(avg_daily_volume * current_price, 2),
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
                "volume_analysis": volume_analysis,
            },
            "token_metrics": token_metrics,
            "analysis_summary": get_analysis_summary(
                sma7, sma20, sma50, sma200, bollinger_bands, volume_analysis
            ),
            "source": "coingecko",
        }
        
        return result
        
    except Exception as e:
        return {
            "error": f"Error analyzing price trend for {token_symbol}: {e}",
            "source": "coingecko"
        }


def get_analysis_summary(sma7, sma20, sma50, sma200, bollinger_bands, volume_analysis):
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
            vol_desc = (
                "Volume does not confirm price movement, suggesting potential reversal."
            )
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
    # Generate a request ID for this comparison operation
    comparison_id = f"compare_{datetime.now().strftime('%H%M%S')}"
    logger.info(f"[{comparison_id}] Starting comparison of {len(token_symbols)} assets: {token_symbols}")
    
    results = {}
    detailed_results = {}
    error_count = 0
    successful_tokens = []

    # Step 1: Collect individual asset data
    for idx, token_symbol in enumerate(token_symbols):
        token_id = f"{comparison_id}_{token_symbol}"
        logger.info(f"[{token_id}] Processing token {idx+1}/{len(token_symbols)}: {token_symbol}")
        
        try:
            # Get price data first to check for errors
            logger.info(f"[{token_id}] Fetching price data")
            price_data = get_coingecko_price_history.invoke(
                {
                    "token_symbol": token_symbol,
                    "candle_interval": candle_interval,
                    "num_candles": num_candles,
                }
            )
            
            # Check for errors in price data
            if "error" in price_data:
                error_msg = price_data["error"]
                logger.error(f"[{token_id}] Error fetching price data: {error_msg}")
                results[token_symbol] = {"error": error_msg}
                error_count += 1
                continue
            
            logger.info(f"[{token_id}] Successfully fetched price data with {len(price_data.get('data', []))} candles")
                
            # Now get the analysis
            logger.info(f"[{token_id}] Analyzing price trend")
            analysis = analyze_price_trend.invoke(
                {
                    "token_symbol": token_symbol,
                    "candle_interval": candle_interval,
                    "num_candles": num_candles,
                }
            )

            # Skip if there was an error
            if "error" in analysis:
                error_msg = analysis["error"]
                logger.error(f"[{token_id}] Error analyzing price trend: {error_msg}")
                results[token_symbol] = {"error": error_msg}
                error_count += 1
                continue
            
            logger.info(f"[{token_id}] Successfully analyzed price trend")

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
                    "long_term": analysis.get("moving_averages", {}).get("long_term"),
                },
                "volatility": analysis.get("token_metrics", {}).get("volatility"),
                "volume_trend": analysis.get("technical_indicators", {})
                .get("volume_analysis", {})
                .get("trend"),
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
            logger.info(f"[{token_id}] Successfully processed")

            # Add a small delay between tokens to avoid rate limiting
            time.sleep(1.0)  # Increased delay to be more conservative

        except Exception as e:
            error_traceback = traceback.format_exc()
            error_msg = f"Error analyzing {token_symbol}: {str(e)}"
            logger.error(f"[{token_id}] {error_msg}\n{error_traceback}")
            results[token_symbol] = {
                "error": error_msg
            }
            error_count += 1

    # If all tokens had errors, return a general error
    if error_count == len(token_symbols):
        logger.error(f"[{comparison_id}] All tokens failed. Rate limiting may be the cause.")
        return {
            "error": "Could not analyze any of the requested tokens. The CoinGecko API may be rate-limited or temporarily unavailable. Please try again later.",
            "source": "coingecko"
        }
    
    logger.info(f"[{comparison_id}] Successfully processed {len(successful_tokens)}/{len(token_symbols)} tokens")

    # Step 2: Calculate investor-friendly comparative metrics
    valid_tokens = {
        symbol: data
        for symbol, data in results.items()
        if "error" not in data and data.get("current_price") is not None
    }
    
    logger.info(f"[{comparison_id}] Found {len(valid_tokens)} valid tokens for comparison")

    comparative_analysis = {}

    if valid_tokens:
        logger.info(f"[{comparison_id}] Generating comparative analysis")
        
        # Performance ranking
        performance_ranking = sorted(
            valid_tokens.items(),
            key=lambda x: x[1].get("price_change_pct", 0),
            reverse=True,
        )

        # Volatility ranking (lower is better for risk-averse investors)
        volatility_ranking = sorted(
            [(symbol, data) for symbol, data in valid_tokens.items() if data.get("volatility", 0) is not None],
            key=lambda x: x[1].get("volatility", float("inf"))
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
                "value": round(risk_adjusted_ranking[0][1], 2) if risk_adjusted_ranking[0][1] != float("inf") else "High",
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
        logger.info(f"[{comparison_id}] Generating investment insights")

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
            
        logger.info(f"[{comparison_id}] Generated {len(investment_insights)} investment insights")

        # Format period in user-friendly terms
        period_text = f"{num_candles} "
        if candle_interval == "1d":
            period_text += "days"
        elif candle_interval == "1h":
            period_text += "hours"
        elif candle_interval == "1w":
            period_text += "weeks"
            
        logger.info(f"[{comparison_id}] Comparison complete. Period: {period_text}")
            
        # Return the final results
        return {
            "individual_tokens": results,
            "comparative_analysis": comparative_analysis,
            "investment_insights": investment_insights,
            "period": period_text,
            "source": "coingecko",
            "error_count": error_count,
            "total_tokens": len(token_symbols),
            "successful_tokens": len(successful_tokens)
        }
    else:
        logger.error(f"[{comparison_id}] No valid tokens for comparison after processing")
        return {
            "error": "Could not generate comparative analysis due to insufficient valid data.",
            "individual_tokens": results,
            "source": "coingecko",
            "error_count": error_count,
            "total_tokens": len(token_symbols)
        }


@tool()
def max_drawdown_for_token(
    token_symbol: str,
    candle_interval: CandleInterval = CandleInterval.DAY,
    num_candles: int = 90,
) -> Dict[str, Any]:
    """
    Calculates the maximum drawdown for a cryptocurrency using CoinGecko price data

    Args:
        token_symbol: Token symbol (e.g., "BTC")
        candle_interval: Candlestick interval (e.g., "1d", "4h", "1h", "15m")
        num_candles: Number of candlesticks to retrieve (max 1000)

    Returns:
        Dictionary containing the maximum drawdown value
    """
    try:
        # Get price data from CoinGecko
        price_data = get_coingecko_price_history.invoke(
            {
                "token_symbol": token_symbol,
                "candle_interval": candle_interval,
                "num_candles": num_candles,
            }
        )

        if "error" in price_data:
            return {
                "error": f"Failed to fetch price data for {token_symbol}: {price_data['error']}",
                "source": "coingecko"
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
            "source": "coingecko"
        }
    except Exception as e:
        return {
            "error": f"Error calculating maximum drawdown: {str(e)}",
            "traceback": traceback.format_exc(),
            "source": "coingecko"
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
            price_data = get_coingecko_price_history.invoke(
                {
                    "token_symbol": token.symbol,
                    "candle_interval": candle_interval,
                    "num_candles": num_candles,
                }
            )

            if "error" in price_data:
                error_symbols.append(token.symbol)
                continue  # Skip this token but continue with others

            # Extract closing prices
            close_prices = [float(candle[4]) for candle in price_data["data"]]
            all_price_data.append(close_prices)
            valid_symbols.append(token.symbol)
            valid_quantities.append(token.amount)

        if not all_price_data:
            return {
                "error": "Could not fetch price data for any of the tokens in your wallet. Please try with custom symbols that are available on CoinGecko.",
                "source": "coingecko"
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
        if candle_interval == "1d":
            period_text = f"{num_candles} days"
        elif candle_interval == "1h":
            period_text = f"{num_candles} hours"
        elif candle_interval == "1w":
            period_text = f"{num_candles} weeks"

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
                        if candle_interval == "1d"
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
            "source": "coingecko"
        }
    except Exception as e:
        return {
            "error": f"Error analyzing portfolio: {str(e)}",
            "traceback": traceback.format_exc(),
            "source": "coingecko"
        }


@tool()
def portfolio_value(
    token_symbols: List[str],
    token_quantities: List[float],
    candle_interval: CandleInterval = CandleInterval.DAY,
    num_candles: int = 90,
) -> Dict[str, Any]:
    """
    Creates the time series of portfolio total value using CoinGecko price data over the specified time period.
    """
    try:
        if len(token_symbols) != len(token_quantities):
            return {"error": "Number of symbols must match number of holdings"}

        # Fetch price data for each asset
        all_price_data = []

        for token_symbol in token_symbols:
            price_data = get_coingecko_price_history.invoke(
                {
                    "token_symbol": token_symbol,
                    "candle_interval": candle_interval,
                    "num_candles": num_candles,
                }
            )

            if "error" in price_data:
                return {
                    "error": f"Failed to fetch price data for {token_symbol}: {price_data['error']}",
                    "source": "coingecko"
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
            "source": "coingecko"
        }
    except Exception as e:
        return {
            "error": f"Error calculating portfolio value: {str(e)}",
            "traceback": traceback.format_exc(),
            "source": "coingecko"
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
            price_data = get_coingecko_price_history.invoke(
                {
                    "token_symbol": token_symbol,
                    "candle_interval": candle_interval,
                    "num_candles": num_candles,
                }
            )

            if "error" in price_data:
                return {
                    "error": f"Failed to fetch price data for {token_symbol}: {price_data['error']}",
                    "source": "coingecko"
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
            "source": "coingecko"
        }
    except Exception as e:
        return {
            "error": f"Error calculating portfolio volatility: {str(e)}",
            "traceback": traceback.format_exc(),
            "source": "coingecko"
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
    token_symbol: str,
    candle_interval: CandleInterval = CandleInterval.DAY,
    num_candles: int = 90,
) -> Dict[str, Any]:
    """
    Analyzes the trend in volatility for a cryptocurrency over the specified time period.
    """
    try:
        # Get price data from CoinGecko
        price_data = get_coingecko_price_history.invoke(
            {
                "token_symbol": token_symbol,
                "candle_interval": candle_interval,
                "num_candles": num_candles,
            }
        )

        if "error" in price_data:
            return {
                "error": f"Failed to fetch price data for {token_symbol}: {price_data['error']}",
                "source": "coingecko"
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
            "source": "coingecko"
        }
    except Exception as e:
        return {
            "error": f"Error analyzing volatility trend: {str(e)}",
            "traceback": traceback.format_exc(),
            "source": "coingecko"
        }
