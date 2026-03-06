"""Uniswap V3 protocol implementation for Ethereum pool discovery."""

from typing import List, Optional, Dict, Any
import logging
import os

import aiohttp

from api.api_types import Pool, Token, Chain, PoolType
from onchain.pools.protocol import Protocol
from onchain.tokens.metadata import TokenMetadataRepo

logger = logging.getLogger(__name__)

# Uniswap V3 subgraph endpoint — configurable via env var.
# The Graph gateway requires an API key appended as a query param or header
# for production use. Set UNISWAP_V3_SUBGRAPH_URL to override.
UNISWAP_V3_SUBGRAPH_URL = os.environ.get(
    "UNISWAP_V3_SUBGRAPH_URL",
    "https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV",
)

# Fee tier mapping (hundredths of a bip -> percentage)
FEE_TIER_MAP = {
    100: 0.01,
    500: 0.05,
    3000: 0.30,
    10000: 1.00,
}

# Stablecoin symbols for pool classification
STABLECOIN_SYMBOLS = {"USDC", "USDT", "DAI", "BUSD", "TUSD", "FRAX", "LUSD", "GUSD", "sUSD"}


class UniswapV3Protocol(Protocol):
    """
    Uniswap V3 protocol — fetches top Ethereum liquidity pools by TVL.

    Uses The Graph's hosted subgraph for pool data. Calculates APR
    from historical fee revenue relative to pool liquidity.
    """

    PROTOCOL_NAME = "uniswap-v3"
    MAX_POOLS = 100

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
        """Fetch top Uniswap V3 pools from the subgraph."""
        query = """
        {
            pools(
                first: %d,
                orderBy: totalValueLockedUSD,
                orderDirection: desc,
                where: { totalValueLockedUSD_gt: "100000" }
            ) {
                id
                token0 { id symbol name }
                token1 { id symbol name }
                feeTier
                totalValueLockedUSD
                poolDayData(first: 30, orderBy: date, orderDirection: desc) {
                    date
                    feesUSD
                    tvlUSD
                }
            }
        }
        """ % self.MAX_POOLS

        try:
            session = await self.session
            async with session.post(
                UNISWAP_V3_SUBGRAPH_URL,
                json={"query": query},
            ) as response:
                if response.status in (401, 403):
                    logger.error(
                        f"Uniswap V3 subgraph auth failed ({response.status}). "
                        f"Set UNISWAP_V3_SUBGRAPH_URL with a valid Graph API key."
                    )
                    return []
                if response.status != 200:
                    logger.error(f"Uniswap V3 subgraph returned {response.status}")
                    return []
                data = await response.json()
        except Exception as e:
            logger.error(f"Error fetching Uniswap V3 pools: {e}")
            return []

        pools_data = data.get("data", {}).get("pools", [])
        return self._convert_to_pools(pools_data)

    def _convert_to_pools(self, subgraph_pools: List[Dict[str, Any]]) -> List[Pool]:
        result: List[Pool] = []

        for pool_data in subgraph_pools:
            token0 = pool_data.get("token0", {})
            token1 = pool_data.get("token1", {})

            tokens = [
                Token(
                    address=token0.get("id", ""),
                    name=token0.get("name", ""),
                    symbol=token0.get("symbol", ""),
                ),
                Token(
                    address=token1.get("id", ""),
                    name=token1.get("name", ""),
                    symbol=token1.get("symbol", ""),
                ),
            ]

            tvl_usd = pool_data.get("totalValueLockedUSD", "0")

            # Calculate APR from fee revenue
            day_data = pool_data.get("poolDayData", [])
            apr_1d = self._calculate_apr(day_data, days=1)
            apr_7d = self._calculate_apr(day_data, days=7)
            apr_30d = self._calculate_apr(day_data, days=30)

            is_stablecoin = self._is_stablecoin_pool(tokens)

            fee_tier = int(pool_data.get("feeTier", 0))
            fee_pct = FEE_TIER_MAP.get(fee_tier, 0)
            protocol_label = f"Uniswap V3 ({fee_pct}%)"

            pool = Pool(
                id=pool_data.get("id", ""),
                chain=Chain.ETHEREUM,
                protocol=protocol_label,
                tokens=tokens,
                type=PoolType.AMM,
                TVL=tvl_usd,
                APRLastDay=apr_1d,
                APRLastWeek=apr_7d,
                APRLastMonth=apr_30d,
                isStableCoin=is_stablecoin,
                impermanentLossRisk=not is_stablecoin,
            )
            result.append(pool)

        return result

    def _calculate_apr(
        self, day_data: List[Dict[str, Any]], days: int = 1
    ) -> float:
        """Calculate annualized APR from poolDayData fee revenue."""
        try:
            if not day_data or len(day_data) == 0:
                return 0.0

            # Use up to `days` entries (already sorted desc by date)
            entries = day_data[:days]
            if not entries:
                return 0.0

            total_fees = sum(float(d.get("feesUSD", 0)) for d in entries)
            avg_tvl = sum(float(d.get("tvlUSD", 0)) for d in entries) / len(entries)

            if avg_tvl <= 0:
                return 0.0

            daily_yield = total_fees / len(entries) / avg_tvl
            apr = daily_yield * 365 * 100
            return round(apr, 2)
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0

    def _is_stablecoin_pool(self, tokens: List[Token]) -> bool:
        if not tokens:
            return False
        return all(t.symbol.upper() in STABLECOIN_SYMBOLS for t in tokens)
