from typing import List, Optional, Dict, Any
from enum import IntEnum

from pydantic import BaseModel
import requests

from api.api_types import Pool, Token, Chain
from defi.pools.protocol import Protocol


class OrcaProtocol(Protocol):
    """
    Implementation of Orca Protocol API
    API docs: https://api.orca.so/docs
    """

    BASE_URL = "https://api.orca.so/v2"

    def __init__(self, chain_id: str = "solana"):
        """
        Initialize the OrcaProtocol client

        Args:
            chain_id: The chain ID to use (default: "solana")
        """
        self.chain_id = chain_id

    def get_pools(
        self,
        addresses: Optional[List[str]] = None,
        token: Optional[str] = None,
        tokens_both_of: Optional[List[str]] = None,
        limit: int = 100,
        sort_by: str = "volume",
        sort_direction: str = "desc",
    ) -> List[Pool]:
        """
        Fetch pools from Orca API and convert to the internal Pool model

        Args:
            addresses: Optional list of pool addresses to filter by
            token: Optional token mint address to find pools containing this token
            tokens_both_of: Optional list of at least two token mint addresses
            limit: Maximum number of items to return (default: 100, max: 1000)
            sort_by: Field by which to sort (volume, rewards, tvl, fees_earned)
            sort_direction: Direction to sort (asc, desc)

        Returns:
            List of Pool objects
        """
        # Build query parameters
        params = {"limit": limit, "sortBy": sort_by, "sortDirection": sort_direction}

        if addresses:
            params["addresses"] = ",".join(addresses)

        if token:
            params["token"] = token

        if tokens_both_of:
            params["tokensBothOf"] = ",".join(tokens_both_of)

        # Make API request
        url = f"{self.BASE_URL}/{self.chain_id}/pools"
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for non-200 responses

        # Parse response
        data = response.json()
        return self._convert_to_pools(data["data"])

    def _convert_to_pools(self, orca_pools: List[Dict[str, Any]]) -> List[Pool]:
        """
        Convert Orca API pool format to internal Pool model

        Args:
            orca_pools: List of pools from Orca API

        Returns:
            List of Pool objects in the internal format
        """
        result = []

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
                type="AMM" if orca_pool.get("poolType") == "whirlpool" else "Unknown",
                TVL=tvl_usdc,
                APRLastDay=apr_last_day,
                APRLastWeek=apr_last_week,
                APRLastMonth=apr_last_month,
                isStableCoin=is_stablecoin,
                impermanentLossRisk=not is_stablecoin,  # Higher risk for non-stablecoin pools
                risk="Low" if is_stablecoin else "Medium",
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
