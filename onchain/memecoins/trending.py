from typing import List
import os
import requests
from cachetools import cached, TTLCache

from langchain_core.tools import BaseTool, tool
from langgraph.graph.graph import RunnableConfig

from api.api_types import TokenMetadata

TRENDING_POOLS_URL = (
    "https://pro-api.coingecko.com/api/v3/onchain/networks/%s/trending_pools"
)


@tool
def get_trending_tokens_on_solana(
    config: RunnableConfig = None,
) -> List[TokenMetadata]:
    """Retrieve the latest trending tokens on Solana from DEX data."""
    return f"""Latest trending tokens: {get_trending_tokens_from_coingecko()[:8]}. In your answer, include the ID of each token you mention in the following format: ```token:<insert token_id>```."""


@cached(cache=TTLCache(maxsize=100, ttl=60 * 10))
def get_trending_tokens_from_coingecko(chain: str = "solana") -> List[TokenMetadata]:
    """Get trending tokens from CoinGecko's trending pools endpoint for the chain."""
    headers = {
        "accept": "application/json",
        "x-cg-pro-api-key": os.environ.get("COINGECKO_API_KEY"),
    }

    response = requests.get(TRENDING_POOLS_URL % chain, headers=headers)
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
        token_id = relationships["base_token"]["data"]["id"].replace("solana_", "")

        # eg "Ghibli / SOL"
        token_name = attributes["name"].split("/")[0].strip()

        token = TokenMetadata(
            address=token_id,
            name=token_name,
            symbol=token_name,
            dex_pool_address=attributes["address"],
            price_usd=attributes["base_token_price_usd"],
            market_cap_usd=attributes.get("market_cap_usd"),
        )
        trending_tokens.append(token)

    return trending_tokens
