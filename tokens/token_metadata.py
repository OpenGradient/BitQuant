from dataclasses import dataclass
import requests
from cachetools import TTLCache
from typing import Optional

class TokenMetadata:

    DEXSCREENER_API_URL = "https://api.dexscreener.com/tokens/v1/solana/%s"
    CACHE_TTL = 3600  # 1 hour in seconds

    def __init__(self):
        self._cache = TTLCache(maxsize=100_000, ttl=self.CACHE_TTL)

    @dataclass
    class TokenMetadata:
        address: str
        name: str
        symbol: str
        image_url: str    
        price_usd: float

    def get_token_metadata(self, token_address: str) -> Optional[TokenMetadata]:
        # Check cache first
        if token_address in self._cache:
            return self._cache[token_address]
        
        metadata = self.fetch_metadata_from_dexscreener(token_address)
        self._cache[token_address] = metadata

        return metadata

    def fetch_metadata_from_dexscreener(self, token_address: str) -> Optional[TokenMetadata]:
        response = requests.get(self.DEXSCREENER_API_URL % token_address)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch metadata from dexscreener: {response.status_code} {response.text}")

        metadata = response.json()
        if len(metadata) == 0:
            return None

        metadata = metadata[0]
        return TokenMetadata(
            address=metadata["baseToken"]["address"],
            name=metadata["baseToken"]["name"],
            symbol=metadata["baseToken"]["symbol"],
            image_url=metadata["info"]["imageUrl"],
            price_usd=metadata["priceUsd"],
        )

