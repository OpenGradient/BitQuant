"""Aave V3 protocol implementation for Ethereum lending pool discovery."""

from typing import List, Optional, Dict, Any
import logging
import os

import aiohttp

from api.api_types import Pool, Token, Chain, PoolType
from onchain.pools.protocol import Protocol
from onchain.tokens.metadata import TokenMetadataRepo

logger = logging.getLogger(__name__)

# Aave V3 Ethereum subgraph — configurable via env var.
AAVE_V3_SUBGRAPH_URL = os.environ.get(
    "AAVE_V3_SUBGRAPH_URL",
    "https://gateway.thegraph.com/api/subgraphs/id/Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g",
)

# Stablecoins for classification
STABLECOIN_SYMBOLS = {"USDC", "USDT", "DAI", "BUSD", "TUSD", "FRAX", "LUSD", "GUSD", "sUSD"}


class AaveProtocol(Protocol):
    """
    Aave V3 protocol — fetches Ethereum lending reserves.

    Returns each reserve as a lending Pool with supply APR.
    Uses The Graph subgraph for on-chain data.
    """

    PROTOCOL_NAME = "aave-v3"

    _session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return self.PROTOCOL_NAME

    @property
    async def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def get_pools(self, token_metadata_repo: TokenMetadataRepo) -> List[Pool]:
        """Fetch Aave V3 reserves from the subgraph."""
        query = """
        {
            reserves(
                first: 50,
                where: { isActive: true }
            ) {
                id
                symbol
                name
                decimals
                underlyingAsset
                liquidityRate
                totalATokenSupply
                totalCurrentVariableDebt
                availableLiquidity
                price {
                    priceInEth
                }
            }
        }
        """

        try:
            session = await self.session
            async with session.post(
                AAVE_V3_SUBGRAPH_URL,
                json={"query": query},
            ) as response:
                if response.status in (401, 403):
                    logger.error(
                        f"Aave V3 subgraph auth failed ({response.status}). "
                        f"Set AAVE_V3_SUBGRAPH_URL with a valid Graph API key."
                    )
                    return []
                if response.status != 200:
                    logger.error(f"Aave V3 subgraph returned {response.status}")
                    return []
                data = await response.json()
        except Exception as e:
            logger.error(f"Error fetching Aave V3 reserves: {e}")
            return []

        reserves = data.get("data", {}).get("reserves", [])

        # Resolve ETH/USD price for TVL conversion.
        # Use the WETH token metadata from DexScreener (already cached).
        eth_usd_price = 0.0
        eth_meta = await token_metadata_repo.get_token_metadata(
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "ethereum"
        )
        if eth_meta and eth_meta.price:
            eth_usd_price = float(eth_meta.price)

        return self._convert_to_pools(reserves, eth_usd_price)

    def _convert_to_pools(
        self, reserves: List[Dict[str, Any]], eth_usd_price: float
    ) -> List[Pool]:
        result: List[Pool] = []

        for reserve in reserves:
            symbol = reserve.get("symbol", "")
            name = reserve.get("name", "")
            address = reserve.get("underlyingAsset", "")
            decimals = int(reserve.get("decimals", 18))

            tokens = [
                Token(address=address, name=name, symbol=symbol),
            ]

            # Aave liquidityRate is in RAY units (1e27), convert to APR percentage
            liquidity_rate = int(reserve.get("liquidityRate", 0))
            supply_apr = (liquidity_rate / 1e27) * 100

            # Calculate TVL in USD:
            #   totalATokenSupply is in token-native units (needs /10^decimals)
            #   priceInEth is the token price denominated in ETH (wei-scaled, 1e18)
            #   Multiply by eth_usd_price to get USD
            total_supply_raw = int(reserve.get("totalATokenSupply", 0))
            total_supply = total_supply_raw / (10**decimals)

            price_in_eth_raw = reserve.get("price", {}).get("priceInEth", "0")
            price_in_eth = int(price_in_eth_raw) / 1e18

            if eth_usd_price > 0 and price_in_eth > 0:
                tvl_usd = str(round(total_supply * price_in_eth * eth_usd_price, 2))
            else:
                tvl_usd = str(round(total_supply, 2))

            is_stablecoin = symbol.upper() in STABLECOIN_SYMBOLS

            pool = Pool(
                id=reserve.get("id", ""),
                chain=Chain.ETHEREUM,
                protocol="Aave V3",
                tokens=tokens,
                type=PoolType.LENDING,
                TVL=tvl_usd,
                APRLastDay=round(supply_apr, 2),
                APRLastWeek=round(supply_apr, 2),
                APRLastMonth=round(supply_apr, 2),
                isStableCoin=is_stablecoin,
                impermanentLossRisk=False,  # Lending has no IL
            )
            result.append(pool)

        return result
