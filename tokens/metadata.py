from dataclasses import dataclass
import botocore.exceptions
import requests
from typing import Optional
import logging
import botocore
from cachetools import TTLCache, LRUCache
from ratelimit import limits, sleep_and_retry
from ratelimit.exception import RateLimitException


@dataclass
class TokenMetadata:
    address: str
    name: str
    symbol: str
    image_url: Optional[str]
    price: Optional[float]


class TokenMetadataRepo:
    DEXSCREENER_API_URL = "https://api.dexscreener.com/tokens/v1/solana/%s"

    NOT_FOUND_CACHE_TTL = 3600 * 24  # 24 hours in seconds
    METADATA_CACHE_SIZE = 10000  # Maximum number of metadata entries to cache

    DEXSCREENER_CALLS_PER_MINUTE = 200
    DEXSCREENER_PERIOD = 60

    def __init__(self, tokens_table):
        self._tokens_table = tokens_table
        self._not_found_cache = TTLCache(maxsize=100_000, ttl=self.NOT_FOUND_CACHE_TTL)
        self._metadata_cache = LRUCache(maxsize=self.METADATA_CACHE_SIZE)

    def get_token_metadata(self, token_address: str) -> Optional[TokenMetadata]:
        # Check local not found cache first
        if token_address in self._not_found_cache:
            return None

        # Check metadata cache
        if token_address in self._metadata_cache:
            return self._metadata_cache[token_address]

        # Try to get from DynamoDB first
        metadata = self._get_from_dynamodb(token_address)
        if metadata is not None:
            self._metadata_cache[token_address] = metadata
            return metadata

        # If not in DynamoDB, fetch from DexScreener
        metadata = self.fetch_metadata_from_dexscreener(token_address)
        if metadata:
            self._store_metadata(metadata)
            self._metadata_cache[token_address] = metadata
        else:
            self._store_not_found(token_address)
            self._not_found_cache[token_address] = True

        return metadata

    def _get_from_dynamodb(self, token_address: str) -> Optional[TokenMetadata]:
        """Retrieve token metadata from DynamoDB."""
        try:
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
                name=item["name"],
                symbol=item["symbol"],
                image_url=item.get("image_url"),
                price=item.get("price"),
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
            "not_found": False,
        }

        if metadata.price:
            item["price"] = metadata.price
        if metadata.image_url:
            item["image_url"] = metadata.image_url

        self._tokens_table.put_item(Item=item)

    def _store_not_found(self, token_address: str) -> None:
        """Store a marker indicating that token metadata was not found."""
        item = {"address": token_address, "not_found": True}
        self._tokens_table.put_item(Item=item)

    @sleep_and_retry
    @limits(calls=DEXSCREENER_CALLS_PER_MINUTE, period=DEXSCREENER_PERIOD)
    def fetch_metadata_from_dexscreener(
        self, token_address: str
    ) -> Optional[TokenMetadata]:
        """Fetch token metadata from DexScreener API with rate limiting."""
        response = requests.get(self.DEXSCREENER_API_URL % token_address)
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
            address=metadata["baseToken"]["address"],
            name=metadata["baseToken"]["name"],
            symbol=metadata["baseToken"]["symbol"],
            image_url=metadata["info"]["imageUrl"] if "info" in metadata else None,
            price=metadata["priceUsd"],
        )
