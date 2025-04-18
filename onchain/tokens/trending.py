from typing import List, Dict
import os
import requests
from cachetools import cached, TTLCache
import logging
from typing import Tuple, Optional

from langchain_core.tools import BaseTool, tool
from langgraph.graph.graph import RunnableConfig
from agent.telemetry import track_tool_usage

from api.api_types import TokenMetadata

TRENDING_POOLS_URL = (
    "https://pro-api.coingecko.com/api/v3/onchain/networks/%s/trending_pools"
)
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

    try:
        holders, error = get_top_token_holders_from_coingecko(address, chain)
        if error:
            return error

        return f"""Top holders of {address} on {chain}: {holders}."""
    except Exception as e:
        logging.error(f"Error in get_top_token_holders with input {token_id}: {e}")
        return f"ERROR: Failed to get top holders for {token_id}: {e}"


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

    try:
        trending_tokens = get_trending_tokens_from_coingecko(chain)[:8]
        return f"""Latest trending tokens: {trending_tokens}. In your answer, include the ID of each token you mention in the following format: ```token:<insert token_id>```, and also the name and symbol of each token."""
    except Exception as e:
        logging.error(f"Error in get_trending_tokens with input {chain}: {e}")
        return f"ERROR: Failed to get trending tokens for {chain}: {e}"


@tool
@track_tool_usage("evaluate_token_risk")
def evaluate_token_risk(
    token_id: str,
    config: RunnableConfig = None,
) -> dict:
    """Evaluate the risk of a token on the given chain, especially for memecoins. Token ID is in the format <chain>:<address>."""
    if ":" not in token_id:
        raise ValueError("Token ID must be in the format <chain>:<address>")

    chain, address = token_id.split(":", 1)
    chain = chain.lower()

    token_info, error = get_token_info_from_coingecko(address, chain)
    if error:
        return error

    attributes = token_info["attributes"]
    risk_analysis = {
        "trust_score": {
            "overall_score": attributes.get("gt_score", 0),
            "category_scores (out of 100)": {
                "pool_quality_score (honeypot risk, buy/sell tax, proxy contract, liquidity amount)": attributes.get(
                    "gt_score_details", {}
                ).get(
                    "pool", 0
                ),
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


@cached(cache=TTLCache(maxsize=1000, ttl=60 * 10))
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


@cached(cache=TTLCache(maxsize=100, ttl=60 * 10))
def get_trending_tokens_from_coingecko(chain: str) -> List[TokenMetadata]:
    """Get trending tokens from CoinGecko's trending pools endpoint for the chain."""
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

    # The response has a data array containing pool information
    for pool in data.get("data", []):
        attributes = pool["attributes"]
        relationships = pool["relationships"]

        # eg solana_BQQzEvYT4knThhkSPBvSKBLg1LEczisWLhx5ydJipump
        token_address = relationships["base_token"]["data"]["id"].split("_")[1]

        # eg "Ghibli / SOL"
        token_name = attributes["name"].split("/")[0].strip()

        token = TokenMetadata(
            address=token_address,
            chain=chain,
            name=token_name,
            symbol=token_name,
            dex_pool_address=attributes["address"],
            price_usd=attributes["base_token_price_usd"],
            market_cap_usd=attributes.get("market_cap_usd"),
        )
        trending_tokens.append(token)

    return trending_tokens
