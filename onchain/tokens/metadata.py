from dataclasses import dataclass
import botocore.exceptions
import requests
from typing import Optional, List
import logging
import botocore
from cachetools import TTLCache, LRUCache, cached
from ratelimit import limits, sleep_and_retry
from ratelimit.exception import RateLimitException
import time

@dataclass
class TokenMetadata:
    timestamp: int
    chain: Optional[str]
    address: str
    name: str
    symbol: str
    image_url: Optional[str]
    price: Optional[float]
    dex_pool_address: Optional[str]
    market_cap_usd: Optional[float]


class TokenMetadataRepo:
    DEXSCREENER_API_URL = "https://api.dexscreener.com/tokens/v1/%s/%s"
    DEXSCREENER_SEARCH_API_URL = "https://api.dexscreener.com/latest/dex/search"

    NOT_FOUND_CACHE_TTL = 3600 * 24  # 24 hours in seconds

    METADATA_CACHE_SIZE = 50_000  # Maximum number of metadata entries to cache
    METADATA_CACHE_TTL = 15 * 60  # 15 minutes in seconds

    DEXSCREENER_CALLS_PER_MINUTE = 200
    DEXSCREENER_SEARCH_CALLS_PER_MINUTE = 60
    DEXSCREENER_PERIOD = 60

    def __init__(self, tokens_table):
        self._tokens_table = tokens_table
        self._not_found_cache = TTLCache(maxsize=self.METADATA_CACHE_SIZE, ttl=self.NOT_FOUND_CACHE_TTL)
        self._metadata_cache = LRUCache(maxsize=self.METADATA_CACHE_SIZE)

    ##
    ## Search token
    ##

    @cached(cache=TTLCache(maxsize=100_000, ttl=60 * 60))
    def search_token(self, token: str, chain: Optional[str] = None) -> Optional[TokenMetadata]:
        """Search for a token by name or symbol."""
        # Check if token is a valid address
        token_metadata = self.get_token_metadata(token, chain)
        if token_metadata:
            return token_metadata

        # If not, search by name or symbol
        return self.search_token_on_dexscreener(token, chain)

    @sleep_and_retry
    @limits(calls=DEXSCREENER_SEARCH_CALLS_PER_MINUTE, period=DEXSCREENER_PERIOD)
    def search_token_on_dexscreener(
        self, token: str, chain: Optional[str]
    ) -> Optional[TokenMetadata]:
        """Search for a token by name or symbol on DexScreener."""
        response = requests.get(self.DEXSCREENER_SEARCH_API_URL, params={"q": token})
        if response.status_code != 200:
            if response.status_code == 429:
                raise RateLimitException(
                    f"Rate limit exceeded: {response.status_code} {response.text}", 60
                )
            else:
                raise Exception(
                    f"Failed to search token on dexscreener: {response.status_code}: {response.text}"
                )

        pairs = response.json()["pairs"]

        # Filter by chain if specified
        if chain:
            pairs = [pair for pair in pairs if pair["chainId"] == chain]
        if len(pairs) == 0:
            # Maybe return error message
            return None

        token_address = pairs[0]["baseToken"]["address"]
        token_chain = pairs[0]["chainId"]
        return self.get_token_metadata(token_address, token_chain)

    ##
    ## Get token metadata
    ##

    def get_token_metadata(
        self, token_address: str, chain: str = "solana"
    ) -> Optional[TokenMetadata]:
        # Check local not found cache first
        if token_address in self._not_found_cache:
            return None

        # Check metadata cache
        if token_address in self._metadata_cache:
            metadata = self._metadata_cache[token_address]
        else:
            metadata = self._get_from_dynamodb(chain, token_address)

        if (
            metadata is not None
            and metadata.timestamp >= time.time() - self.METADATA_CACHE_TTL
        ):
            self._metadata_cache[token_address] = metadata
            return metadata

        # If not in DynamoDB or has expired, fetch from DexScreener
        metadata = self.fetch_metadata_from_dexscreener(chain, token_address)
        if metadata:
            self._store_metadata(metadata)
            self._metadata_cache[token_address] = metadata
        else:
            self._store_not_found(token_address)
            self._not_found_cache[token_address] = True

        return metadata

    def _get_from_dynamodb(
        self, chain: str, token_address: str
    ) -> Optional[TokenMetadata]:
        """Retrieve token metadata from DynamoDB."""
        try:
            # TODO: Add chain to the key
            response = self._tokens_table.get_item(Key={"address": token_address})

            if "Item" not in response:
                return None

            item = response["Item"]

            # Check if this is a "not found" marker
            if item.get("not_found", False):
                self._not_found_cache[token_address] = True
                return None

            metadata = TokenMetadata(
                address=item["address"],
                chain=item.get("chain"),
                name=item["name"],
                symbol=item["symbol"],
                timestamp=item["timestamp"],
                image_url=item.get("image_url"),
                price=item.get("price"),
                dex_pool_address=item.get("dex_pool_address"),
                market_cap_usd=item.get("market_cap_usd"),
            )

            return metadata
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "ResourceNotFoundException":
                return None
            logging.error(f"Error retrieving token metadata from DynamoDB: {error}")
            raise error

    def _store_metadata(self, metadata: TokenMetadata) -> None:
        """Store token metadata in DynamoDB."""
        item = {
            "address": metadata.address,
            "name": metadata.name,
            "symbol": metadata.symbol,
            "timestamp": metadata.timestamp,
            "not_found": False,
        }

        if metadata.chain:
            item["chain"] = metadata.chain
        if metadata.price:
            item["price"] = metadata.price
        if metadata.image_url:
            item["image_url"] = metadata.image_url
        if metadata.dex_pool_address:
            item["dex_pool_address"] = metadata.dex_pool_address
        if metadata.market_cap_usd:
            item["market_cap_usd"] = metadata.market_cap_usd

        self._tokens_table.put_item(Item=item)
        logging.info(f"Stored metadata for token: {metadata.address}")

    def _store_not_found(self, token_address: str) -> None:
        """Store a marker indicating that token metadata was not found."""
        item = {
            "chain": "solana",
            "address": token_address,
            "not_found": True,
            "timestamp": int(time.time()),
        }
        self._tokens_table.put_item(Item=item)

    @sleep_and_retry
    @limits(calls=DEXSCREENER_CALLS_PER_MINUTE, period=DEXSCREENER_PERIOD)
    def fetch_metadata_from_dexscreener(
        self, chain: str, token_address: str
    ) -> Optional[TokenMetadata]:
        """Fetch token metadata from DexScreener API with rate limiting."""
        response = requests.get(self.DEXSCREENER_API_URL % (chain, token_address))
        if response.status_code != 200:
            if response.status_code == 429:
                raise RateLimitException(
                    f"Rate limit exceeded: {response.status_code} {response.text}", 60
                )
            else:
                logging.error(
                    f"Failed to fetch metadata from dexscreener: {response.status_code} {response.text}"
                )
                return None

        metadata = response.json()
        if len(metadata) == 0:
            return None

        metadata = metadata[0]
        return TokenMetadata(
            chain=chain,
            address=metadata["baseToken"]["address"],
            name=metadata["baseToken"]["name"],
            symbol=metadata["baseToken"]["symbol"],
            image_url=metadata["info"]["imageUrl"] if "info" in metadata else None,
            price=metadata["priceUsd"],
            dex_pool_address=metadata.get("pairAddress"),
            market_cap_usd=metadata.get("marketCap"),
            timestamp=int(time.time()),
        )
