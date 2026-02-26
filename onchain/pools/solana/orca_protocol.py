from typing import List, Optional, Dict, Any
from enum import IntEnum

from pydantic import BaseModel
import aiohttp

from api.api_types import Pool, Token, Chain, PoolType
from onchain.pools.protocol import Protocol
from onchain.tokens.metadata import TokenMetadataRepo


class OrcaProtocol(Protocol):
    """
    Implementation of Orca Protocol API
    API docs: https://api.orca.so/docs
    """

    PROTOCOL_NAME = "orca"
    BASE_URL = "https://api.orca.so/v2"

    chain_id: str
    _session: Optional[aiohttp.ClientSession] = None

    def __init__(self, chain_id: str = "solana"):
        """
        Initialize the OrcaProtocol client

        Args:
            chain_id: The chain ID to use (default: "solana")
        """
        self.chain_id = chain_id

    @property
    def name(self) -> str:
        return self.PROTOCOL_NAME

    @property
    async def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the protocol's session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_pools(self, token_metadata_repo: TokenMetadataRepo) -> List[Pool]:
        """
        Fetch pools from Orca API and convert to the internal Pool model
        """
        # Make API request
        url = f"{self.BASE_URL}/{self.chain_id}/pools"
        session = await self.session
        async with session.get(url) as response:
            response.raise_for_status()  # Raise exception for non-200 responses
            data = await response.json()

        # Parse response
        return self._convert_to_pools(data["data"])

    def _convert_to_pools(self, orca_pools: List[Dict[str, Any]]) -> List[Pool]:
        result: List[Pool] = []

        for orca_pool in orca_pools:
            # Convert tokens
            tokens = []

            # Add token A
            token_a_data = orca_pool.get("tokenA", {})
            if token_a_data:
                tokens.append(
                    Token(
                        address=token_a_data.get("address", ""),
                        name=token_a_data.get("name", ""),
                        symbol=token_a_data.get("symbol", ""),
                    )
                )

            # Add token B
            token_b_data = orca_pool.get("tokenB", {})
            if token_b_data:
                tokens.append(
                    Token(
                        address=token_b_data.get("address", ""),
                        name=token_b_data.get("name", ""),
                        symbol=token_b_data.get("symbol", ""),
                    )
                )

            # Extract stats for APR calculations
            stats = orca_pool.get("stats", {})

            # Get TVL in USDC
            tvl_usdc = orca_pool.get("tvlUsdc", "0")

            # Calculate APR based on fees and rewards
            apr_last_day = self._calculate_apr(stats.get("24h", {}), tvl_usdc)
            apr_last_week = self._calculate_apr(stats.get("7d", {}), tvl_usdc, days=7)
            apr_last_month = self._calculate_apr(
                stats.get("30d", {}), tvl_usdc, days=30
            )

            # Determine if pool is stablecoin
            is_stablecoin = self._is_stablecoin_pool(tokens)

            # Create Pool object
            pool = Pool(
                id=orca_pool.get("address", ""),
                chain=Chain.SOLANA,
                protocol="Orca",
                tokens=tokens,
                type=PoolType.AMM,
                TVL=tvl_usdc,
                APRLastDay=apr_last_day,
                APRLastWeek=apr_last_week,
                APRLastMonth=apr_last_month,
                isStableCoin=is_stablecoin,
                impermanentLossRisk=not is_stablecoin,
            )

            result.append(pool)

        return result

    def _calculate_apr(
        self, period_stats: Dict[str, Any], tvl_usdc: str, days: int = 1
    ) -> float:
        """
        Calculate APR based on fees and rewards for a specific period

        Args:
            period_stats: Stats for a specific period (24h, 7d, 30d)
            tvl_usdc: TVL in USDC
            days: Number of days in the period

        Returns:
            APR as a float
        """
        try:
            # Extract fees and rewards
            fees = float(period_stats.get("fees", "0"))
            rewards = float(period_stats.get("rewards", "0"))
            tvl = float(tvl_usdc)

            if tvl <= 0:
                return 0.0

            # Calculate daily yield
            daily_yield = (fees + rewards) / tvl

            # Annualize based on period length
            apr = (daily_yield / days) * 365 * 100

            return round(apr, 2)
        except (ValueError, TypeError):
            return 0.0

    def _is_stablecoin_pool(self, tokens: List[Token]) -> bool:
        """
        Determine if a pool is a stablecoin pool based on token symbols

        Args:
            tokens: List of tokens in the pool

        Returns:
            True if all tokens are stablecoins, False otherwise
        """
        if not tokens:
            return False

        # Common stablecoin symbols
        stablecoin_symbols = {
            "USDC",
            "USDT",
            "DAI",
            "BUSD",
            "TUSD",
            "USDH",
            "USDR",
            "CUSD",
        }

        # Check if all tokens are stablecoins
        for token in tokens:
            if token.symbol not in stablecoin_symbols:
                return False

        return True
