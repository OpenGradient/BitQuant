from typing import List, Dict
import os
import requests
from cachetools import cached, TTLCache

from langchain_core.tools import BaseTool, tool
from langgraph.graph.graph import RunnableConfig

from api.api_types import TokenMetadata

TRENDING_POOLS_URL = (
    "https://pro-api.coingecko.com/api/v3/onchain/networks/%s/trending_pools"
)
TOKEN_INFO_URL = (
    "https://pro-api.coingecko.com/api/v3/onchain/networks/%s/tokens/%s/info"
)

CHAIN_REMAPPINGS = {
    "sui": "sui-network",
    "ethereum": "eth",
    "ethereum-network": "eth",
}


@tool
def get_trending_tokens(
    chain: str = "solana",
    config: RunnableConfig = None,
) -> List[TokenMetadata]:
    """Retrieve the latest trending tokens on the given chain from DEX data."""
    chain = chain.lower()
    trending_tokens = get_trending_tokens_from_coingecko(chain)[:8]

    return f"""Latest trending tokens: {trending_tokens}. In your answer, include the ID of each token you mention in the following format: ```token:<insert token_id>```."""


@tool
def evaluate_token_risk(
    token_address: str,
    chain: str = "solana",
    config: RunnableConfig = None,
) -> dict:
    """Evaluate the risk of a token on the given chain."""
    chain = chain.lower()
    token_info = get_token_info_from_coingecko(token_address, chain)
    attributes = token_info["attributes"]
    
    risk_analysis = {
        "trust_score": {
            "overall": attributes.get("gt_score", 0),
            "breakdown": {
                "pool": attributes.get("gt_score_details", {}).get("pool", 0),
                "creation": attributes.get("gt_score_details", {}).get("creation", 0),
                "info": attributes.get("gt_score_details", {}).get("info", 0),
                "transaction": attributes.get("gt_score_details", {}).get("transaction", 0),
                "holders": attributes.get("gt_score_details", {}).get("holders", 0)
            }
        },
        "holder_distribution": {
            "total_holders": attributes.get("holders", {}).get("count", 0),
            "distribution": {
                "top_10": attributes.get("holders", {}).get("distribution_percentage", {}).get("top_10", "0"),
                "11_30": attributes.get("holders", {}).get("distribution_percentage", {}).get("11_30", "0"),
                "31_50": attributes.get("holders", {}).get("distribution_percentage", {}).get("31_50", "0"),
                "rest": attributes.get("holders", {}).get("distribution_percentage", {}).get("rest", "0")
            },
            "concentration_risk": "High" if float(attributes.get("holders", {}).get("distribution_percentage", {}).get("top_10", "0")) > 30 else "Moderate"
        },
        "social_presence": {
            "twitter": attributes.get("twitter_handle"),
            "discord": attributes.get("discord_url"),
            "telegram": attributes.get("telegram_handle"),
            "website": attributes.get("websites")
        },
    }

    return risk_analysis


@cached(cache=TTLCache(maxsize=1000, ttl=60 * 10))
def get_token_info_from_coingecko(token_address: str, chain: str) -> TokenMetadata:
    """Get token info from CoinGecko's token info endpoint for the chain."""
    headers = {
        "accept": "application/json",
        "x-cg-pro-api-key": os.environ.get("COINGECKO_API_KEY"),
    }

    if chain in CHAIN_REMAPPINGS:
        coingecko_chain = CHAIN_REMAPPINGS[chain]
    else:
        coingecko_chain = chain

    response = requests.get(TOKEN_INFO_URL % (coingecko_chain, token_address), headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch token info: {response.status_code} {response.text}"
        )

    data = response.json()
    return data["data"]


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
