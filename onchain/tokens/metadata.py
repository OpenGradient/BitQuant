from dataclasses import dataclass
import aiohttp
from typing import Optional, Callable, Awaitable
import logging
from cachetools import TTLCache, LRUCache
from aiolimiter import AsyncLimiter
import time
from async_lru import alru_cache
from boto3.dynamodb.table import TableResource

SUPPORTED_CHAINS = [
    "solana",
    "ethereum",
    "polygon",
    "bnb",
    "avax",
    "sui",
    "arbitrum",
    "base",
    "optimism",
    "celo",
    "fantom",
    "gnosis",
    "avalanche",
    "bsc",
]


@dataclass
class TokenMetadata:
    timestamp: int
    chain: str
    address: str
    name: str
    symbol: str
    image_url: Optional[str]
    price: Optional[float]
    dex_pool_address: Optional[str]
    market_cap_usd: Optional[int]


USDC_TOKEN = TokenMetadata(
    chain="solana",
    address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    name="USD Coin",
    symbol="USDC",
    image_url="https://statics.solscan.io/cdn/imgs/s60?ref=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f736f6c616e612d6c6162732f746f6b656e2d6c6973742f6d61696e2f6173736574732f6d61696e6e65742f45506a465764643541756671535371654d32714e31787a7962617043384734774547476b5a777954447431762f6c6f676f2e706e67",
    price=1.0,
    dex_pool_address=None,
    market_cap_usd=None,
    timestamp=int(time.time()),
)

USDT_TOKEN = TokenMetadata(
    chain="solana",
    address="Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    name="Tether",
    symbol="USDT",
    image_url="https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB/logo.svg",
    price=1.0,
    dex_pool_address=None,
    market_cap_usd=None,
    timestamp=int(time.time()),
)

SOL_TOKEN = TokenMetadata(
    chain="solana",
    address="So11111111111111111111111111111111111111112",
    name="Solana",
    symbol="SOL",
    image_url="https://solana.com/src/img/branding/solanaLogoMark.png",
    price=1.0,
    dex_pool_address=None,
    market_cap_usd=None,
    timestamp=int(time.time()),
)

# Ethereum hardcoded tokens — price=None for ETH (non-stablecoin).
# search_token() fetches the live price from DexScreener for ETH;
# stablecoins use price=1.0 which is accurate.
ETH_TOKEN = TokenMetadata(
    chain="ethereum",
    address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    name="Ethereum",
    symbol="ETH",
    image_url="https://assets.coingecko.com/coins/images/279/small/ethereum.png",
    price=None,
    dex_pool_address=None,
    market_cap_usd=None,
    timestamp=int(time.time()),
)

USDC_ETH_TOKEN = TokenMetadata(
    chain="ethereum",
    address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    name="USD Coin",
    symbol="USDC",
    image_url="https://assets.coingecko.com/coins/images/6319/small/usdc.png",
    price=1.0,
    dex_pool_address=None,
    market_cap_usd=None,
    timestamp=int(time.time()),
)

USDT_ETH_TOKEN = TokenMetadata(
    chain="ethereum",
    address="0xdAC17F958D2ee523a2206206994597C13D831ec7",
    name="Tether",
    symbol="USDT",
    image_url="https://assets.coingecko.com/coins/images/325/small/Tether.png",
    price=1.0,
    dex_pool_address=None,
    market_cap_usd=None,
    timestamp=int(time.time()),
)

DAI_ETH_TOKEN = TokenMetadata(
    chain="ethereum",
    address="0x6B175474E89094C44Da98b954EedeAC495271d0F",
    name="Dai",
    symbol="DAI",
    image_url="https://assets.coingecko.com/coins/images/9956/small/Badge_Dai.png",
    price=1.0,
    dex_pool_address=None,
    market_cap_usd=None,
    timestamp=int(time.time()),
)

WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"


class TokenMetadataRepo:
    DEXSCREENER_API_URL = "https://api.dexscreener.com/tokens/v1/%s/%s"
    DEXSCREENER_SEARCH_API_URL = "https://api.dexscreener.com/latest/dex/search"

    NOT_FOUND_CACHE_TTL = 10_000 * 24  # 24 hours in seconds

    METADATA_CACHE_SIZE = 1_000_000  # Maximum number of metadata entries to cache
    METADATA_CACHE_TTL = 60 * 60  # 1 hour in seconds

    DEXSCREENER_CALLS_PER_MINUTE = 200
    DEXSCREENER_SEARCH_CALLS_PER_MINUTE = 30
    DEXSCREENER_PERIOD = 60

    def __init__(self, get_table: Callable[[], Awaitable[TableResource]]):
        self.get_table = get_table
        self._not_found_cache = TTLCache(
            maxsize=self.METADATA_CACHE_SIZE, ttl=self.NOT_FOUND_CACHE_TTL
        )
        self._metadata_cache = LRUCache(maxsize=self.METADATA_CACHE_SIZE)
        self._session = None
        self._search_rate_limiter = AsyncLimiter(
            self.DEXSCREENER_SEARCH_CALLS_PER_MINUTE, self.DEXSCREENER_PERIOD
        )
        self._metadata_rate_limiter = AsyncLimiter(
            self.DEXSCREENER_CALLS_PER_MINUTE, self.DEXSCREENER_PERIOD
        )

    @property
    async def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),  # Add timeout
                connector=aiohttp.TCPConnector(
                    limit=100, limit_per_host=30
                ),  # Connection pooling
            )
        return self._session

    @alru_cache(maxsize=100_000, ttl=60 * 60)
    async def search_token(
        self, token: str, chain: Optional[str] = None
    ) -> Optional[TokenMetadata]:
        """Search for a token by name or symbol."""
        if chain:
            chain = chain.lower()
        if chain not in SUPPORTED_CHAINS:
            return None

        # Hardcoded Solana tokens
        if chain == "solana":
            if token == "usdc" or token == "USDC" or token == USDC_TOKEN.address:
                return USDC_TOKEN
            if token == "usdt" or token == "USDT" or token == USDT_TOKEN.address:
                return USDT_TOKEN
            if (
                token == "sol"
                or token == "SOL"
                or token == "solana"
                or token == "Solana"
                or token == SOL_TOKEN.address
            ):
                return SOL_TOKEN

        # Hardcoded Ethereum tokens
        if chain == "ethereum":
            token_lower = token.lower()
            if token_lower in ("eth", "ethereum", "weth") or token_lower == ETH_TOKEN.address.lower():
                # Fetch live price from DexScreener; fall back to static token if unavailable
                eth_meta = await self.get_token_metadata(WETH_ADDRESS, "ethereum")
                return eth_meta if eth_meta else ETH_TOKEN
            if token_lower in ("usdc", "usd coin") or token_lower == USDC_ETH_TOKEN.address.lower():
                return USDC_ETH_TOKEN
            if token_lower in ("usdt", "tether") or token_lower == USDT_ETH_TOKEN.address.lower():
                return USDT_ETH_TOKEN
            if token_lower in ("dai",) or token_lower == DAI_ETH_TOKEN.address.lower():
                return DAI_ETH_TOKEN

        if chain is not None:
            # Check if token is a valid address
            token_metadata = await self.get_token_metadata(token, chain)
            if token_metadata:
                return token_metadata

        if not token:
            return None

        # If not, search by name or symbol
        return await self._search_token_on_dexscreener(token, chain)

    async def _search_token_on_dexscreener(
        self, token: str, chain: Optional[str]
    ) -> Optional[TokenMetadata]:
        """Search for a token by name or symbol on DexScreener."""
        async with self._search_rate_limiter:
            session = await self.session
            async with session.get(
                self.DEXSCREENER_SEARCH_API_URL, params={"q": token}
            ) as response:
                if response.status != 200:
                    if response.status == 429:
                        raise Exception(
                            f"Rate limit exceeded: {response.status} {await response.text()}"
                        )
                    else:
                        raise Exception(
                            f"Failed to search token on dexscreener: {response.status}: {await response.text()}"
                        )

                data = await response.json()
                pairs = data["pairs"]

                # Filter by chain if specified
                if chain:
                    pairs = [pair for pair in pairs if pair["chainId"] == chain]
                if len(pairs) == 0:
                    return None

                token_address = pairs[0]["baseToken"]["address"]
                token_chain = pairs[0]["chainId"]
                return await self.get_token_metadata(token_address, token_chain)

    async def get_token_metadata(
        self, token_address: str, chain: str
    ) -> Optional[TokenMetadata]:
        # Check local not found cache first
        if (chain, token_address) in self._not_found_cache:
            return None

        if chain not in SUPPORTED_CHAINS:
            return None

        # Check metadata cache
        if (chain, token_address) in self._metadata_cache:
            metadata = self._metadata_cache[(chain, token_address)]
        else:
            metadata = await self._get_from_dynamodb(chain, token_address)

        if (
            metadata is not None
            and metadata.timestamp >= time.time() - self.METADATA_CACHE_TTL
        ):
            self._metadata_cache[(chain, token_address)] = metadata
            return metadata

        # If not in DynamoDB or has expired, fetch from DexScreener
        metadata = await self._fetch_metadata_from_dexscreener(chain, token_address)
        if metadata:
            await self._store_metadata(metadata)
            self._metadata_cache[(chain, token_address)] = metadata
        else:
            await self._store_not_found(chain, token_address)
            self._not_found_cache[(chain, token_address)] = True

        return metadata

    async def _get_from_dynamodb(
        self, chain: str, token_address: str
    ) -> Optional[TokenMetadata]:
        """Retrieve token metadata from DynamoDB - MOCKED to always return None."""
        # Mock: Always return None to force fetching from DexScreener
        return None

    async def _store_metadata(self, metadata: TokenMetadata) -> None:
        """Store token metadata in DynamoDB."""
        item = {
            "id": f"{metadata.chain}:{metadata.address}",
            "address": metadata.address,
            "chain": metadata.chain,
            "name": metadata.name,
            "symbol": metadata.symbol,
            "timestamp": metadata.timestamp,
            "not_found": False,
        }

        if metadata.price:
            item["price"] = metadata.price
        if metadata.image_url:
            item["image_url"] = metadata.image_url
        if metadata.dex_pool_address:
            item["dex_pool_address"] = metadata.dex_pool_address
        if metadata.market_cap_usd:
            item["market_cap_usd"] = int(metadata.market_cap_usd)

        async with self.get_table() as table:
            await table.put_item(Item=item)
            logging.info(
                f"Stored metadata for token: {metadata.address} on chain: {metadata.chain}"
            )

    async def _store_not_found(self, chain: str, token_address: str) -> None:
        """Store a marker indicating that token metadata was not found."""
        item = {
            "id": f"{chain}:{token_address}",
            "address": token_address,
            "chain": chain,
            "not_found": True,
            "timestamp": int(time.time()),
        }
        async with self.get_table() as table:
            await table.put_item(Item=item)

    async def _fetch_metadata_from_dexscreener(
        self, chain: str, token_address: str
    ) -> Optional[TokenMetadata]:
        """Fetch token metadata from DexScreener API with rate limiting."""
        async with self._metadata_rate_limiter:
            session = await self.session
            async with session.get(
                self.DEXSCREENER_API_URL % (chain, token_address)
            ) as response:
                if response.status != 200:
                    if response.status == 429:
                        raise Exception(
                            f"Rate limit exceeded: {response.status} {await response.text()}"
                        )
                    else:
                        logging.error(
                            f"Failed to fetch metadata from dexscreener: {response.status} {await response.text()}"
                        )
                        return None

                metadata = await response.json()
                if len(metadata) == 0:
                    return None

                metadata = metadata[0]
                return TokenMetadata(
                    chain=chain,
                    address=metadata["baseToken"]["address"],
                    name=metadata["baseToken"]["name"],
                    symbol=metadata["baseToken"]["symbol"],
                    image_url=(
                        metadata["info"]["imageUrl"] if "info" in metadata else None
                    ),
                    price=metadata["priceUsd"],
                    dex_pool_address=metadata.get("pairAddress"),
                    market_cap_usd=metadata.get("marketCap"),
                    timestamp=int(time.time()),
                )

    async def close(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None
