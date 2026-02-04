from typing import List
import os
import requests
from cachetools import cached, TTLCache
import logging
from typing import Tuple, Optional
import math

from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from agent.telemetry import track_tool_usage

from api.api_types import TokenMetadata

TRENDING_POOLS_URL = "https://pro-api.coingecko.com/api/v3/onchain/networks/%s/trending_pools?include=base_token"
TOKEN_INFO_URL = (
    "https://pro-api.coingecko.com/api/v3/onchain/networks/%s/tokens/%s/info"
)
TOKEN_HOLDERS_URL = (
    "https://pro-api.coingecko.com/api/v3/onchain/networks/%s/tokens/%s/top_holders"
)

CHAIN_REMAPPINGS = {
    "sui": "sui-network",
    "ethereum": "eth",
    "ethereum-network": "eth",
    "polygon": "polygon_pos",
    "avalanche": "avax",
    "bnb": "bsc",
    "dogecoin": "dogechain",
}


@tool
@track_tool_usage("get_top_token_holders")
def get_top_token_holders(
    token_id: str,
    config: RunnableConfig = None,
) -> List[TokenMetadata]:
    """Get the top holders of a token on the given chain."""
    if ":" not in token_id:
        return "ERROR: Token ID must be in the format <chain>:<address>"

    chain, address = token_id.split(":", 1)
    chain = chain.lower()

    holders, error = get_top_token_holders_from_coingecko(address, chain)
    if error:
        return error

    return f"""Top holders of {address} on {chain}: {holders}."""


@cached(cache=TTLCache(maxsize=10_000, ttl=60 * 10))
def get_top_token_holders_from_coingecko(
    token_address: str, chain: str
) -> Tuple[List, Optional[str]]:
    """Get the top holders of a token on the given chain."""
    headers = {
        "accept": "application/json",
        "x-cg-pro-api-key": os.environ.get("COINGECKO_API_KEY"),
    }

    chain = chain.lower()
    if chain in CHAIN_REMAPPINGS:
        coingecko_chain = CHAIN_REMAPPINGS[chain]
    else:
        coingecko_chain = chain

    response = requests.get(
        TOKEN_HOLDERS_URL % (coingecko_chain, token_address), headers=headers
    )
    if response.status_code == 404:
        logging.warning(f"Token top holders not found: {token_address} on {chain}")
        return [], "Top holders for this token are not available."
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch token holders: {response.status_code} {response.text}"
        )

    data = response.json()["data"]
    holders = data["attributes"]["holders"]

    # Format each holder's information
    formatted_holders = []
    for holder in holders:
        holder_info = {
            "address": f"```address:{chain}:{holder['address']}```",
            "account_label": holder["label"] or "None",
            "percentage": holder["percentage"],
            "value_usd": holder["value"],
        }
        formatted_holders.append(holder_info)

    return formatted_holders, None


@tool
@track_tool_usage("get_trending_tokens")
def get_trending_tokens(
    chain: str = "solana",
    config: RunnableConfig = None,
) -> str:
    """Retrieve the latest trending tokens on the given chain from DEX data."""
    chain = chain.lower()
    trending_tokens = get_trending_tokens_from_coingecko(chain)[:9]
    return f"""Trending tokens: {trending_tokens}. In your answer, include the ID of each token you mention in the following format: ```token:<insert token_id>```, and the name and symbol too."""


@tool
@track_tool_usage("evaluate_token_risk")
def evaluate_token_risk(
    token_id: str,
    config: RunnableConfig = None,
) -> dict:
    """Evaluate the risk of a token on the given chain, especially for memecoins. Token ID is in the format <chain>:<address>."""
    if ":" not in token_id:
        return "ERROR: Token ID must be in the format <chain>:<address>"

    chain, address = token_id.split(":", 1)
    chain = chain.lower()

    token_info, error = get_token_info_from_coingecko(address, chain)
    if error:
        return error

    attributes = token_info["attributes"]
    if not attributes:
        return "Token info not available."

    risk_analysis = {
        "trust_score": {
            "overall_score": attributes.get("gt_score", 0),
            "category_scores (out of 100)": {
                "pool_quality_score (honeypot risk, buy/sell tax, proxy contract, liquidity amount)": attributes.get(
                    "gt_score_details", {}
                ).get("pool", 0),
                "token_age_score": attributes.get("gt_score_details", {}).get(
                    "creation", 0
                ),
                "info_completeness_score": attributes.get("gt_score_details", {}).get(
                    "info", 0
                ),
                "transaction_volume_score": attributes.get("gt_score_details", {}).get(
                    "transaction", 0
                ),
                "holders_distribution_score": attributes.get(
                    "gt_score_details", {}
                ).get("holders", 0),
            },
        },
        "holder_distribution": {
            "total_holders": attributes.get("holders", {}).get("count", 0),
            "distribution": {
                "top_10": attributes.get("holders", {})
                .get("distribution_percentage", {})
                .get("top_10", "unknown"),
            },
            "concentration_risk": (
                "High"
                if float(
                    attributes.get("holders", {})
                    .get("distribution_percentage", {})
                    .get("top_10", "0")
                )
                > 30
                else "Moderate"
            ),
        },
        "social_presence": {
            "twitter": attributes.get("twitter_handle"),
            "discord": attributes.get("discord_url"),
            "telegram": attributes.get("telegram_handle"),
            "website": attributes.get("websites"),
        },
    }

    return risk_analysis


@cached(cache=TTLCache(maxsize=100_000, ttl=60 * 20))
def get_token_info_from_coingecko(
    token_address: str, chain: str
) -> Tuple[TokenMetadata, Optional[str]]:
    """Get token info from CoinGecko's token info endpoint for the chain."""
    headers = {
        "accept": "application/json",
        "x-cg-pro-api-key": os.environ.get("COINGECKO_API_KEY"),
    }

    if chain in CHAIN_REMAPPINGS:
        coingecko_chain = CHAIN_REMAPPINGS[chain]
    else:
        coingecko_chain = chain

    response = requests.get(
        TOKEN_INFO_URL % (coingecko_chain, token_address), headers=headers
    )
    if response.status_code == 404:
        logging.warning(f"Token info not found: {token_address} on {chain}")
        return None, "Token metadata not available."
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch token info: {response.status_code} {response.text}"
        )

    data = response.json()
    return data["data"], None


@cached(cache=TTLCache(maxsize=100_000, ttl=60 * 20))
def get_trending_tokens_from_coingecko(chain: str) -> List[TokenMetadata]:
    headers = {
        "accept": "application/json",
        "x-cg-pro-api-key": os.environ.get("COINGECKO_API_KEY"),
    }

    if chain in CHAIN_REMAPPINGS:
        coingecko_chain = CHAIN_REMAPPINGS[chain]
    else:
        coingecko_chain = chain

    response = requests.get(TRENDING_POOLS_URL % coingecko_chain, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch trending tokens: {response.status_code} {response.text}"
        )

    data = response.json()
    trending_tokens = []

    token_metadata = {
        token["id"]: token["attributes"] for token in data.get("included", [])
    }

    # The response has a data array containing pool information
    for pool in data.get("data", []):
        attributes = pool["attributes"]
        relationships = pool["relationships"]

        # eg solana_BQQzEvYT4knThhkSPBvSKBLg1LEczisWLhx5ydJipump
        token_id = relationships["base_token"]["data"]["id"]
        token = token_metadata[token_id]

        rounded_price = _round_to_significant_digits(
            float(attributes["base_token_price_usd"]), 5
        )

        token = TokenMetadata(
            address=token["address"],
            chain=chain,
            name=token["name"],
            symbol=token["symbol"],
            dex_pool_address=attributes["address"],
            price_usd=rounded_price,
            market_cap_usd=attributes.get("market_cap_usd"),
        )
        trending_tokens.append(token)

    return trending_tokens


def _round_to_significant_digits(value: float, digits: int = 5) -> float:
    """
    Round a number to a specified number of significant digits.
    For numbers > 1, keep all integer digits and only round the fractional part.

    Examples:
        - 530.24 -> 530.24 (keep integer part, round fractional part)
        - 0.000555 -> 0.000555 (5 significant digits)
        - 0.000123456 -> 0.00012346 (5 significant digits)
        - 1234567.89 -> 1234567.9 (keep all integer digits, round fractional part)
    """
    if value == 0:
        return 0.0

    # For numbers >= 1, keep all integer digits and only round fractional part
    if abs(value) >= 1:
        # Count integer digits
        integer_part = int(abs(value))
        integer_digits = len(str(integer_part))

        # Calculate remaining digits for fractional part
        remaining_digits = max(0, digits - integer_digits)

        # Round to the calculated decimal places
        rounded = round(value, remaining_digits)

        # Remove trailing zeros for cleaner display
        if rounded == int(rounded):
            return float(int(rounded))

        return rounded
    else:
        # For numbers < 1, use the original significant digits logic
        magnitude = math.floor(math.log10(abs(value)))
        decimal_places = digits - 1 - magnitude
        rounded = round(value, decimal_places)

        # Remove trailing zeros for cleaner display
        if rounded == int(rounded):
            return float(int(rounded))

        return rounded
