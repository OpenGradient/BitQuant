from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import aiohttp
from datetime import datetime, timedelta, UTC
import statistics

from api.api_types import Pool, Token, Chain, PoolType
from onchain.pools.protocol import Protocol
from onchain.pools.solana.constants import stablecoin_symbols
from onchain.tokens.metadata import TokenMetadataRepo


class KaminoProtocol(Protocol):
    """
    Implementation of Kamino Protocol API for lending pools
    API docs: https://api.kamino.finance
    """

    PROTOCOL_NAME = "kamino"
    BASE_URL = "https://api.kamino.finance"
    _session: Optional[aiohttp.ClientSession] = None

    def __init__(
        self,
        cluster: str = "mainnet-beta",
        program_id: str = "KLend2g3cP87fffoy8q1mQqGKjrxjC8boSyAYavgmjD",
        market_pubkey: str = "7u3HeHxYDLhnCoErrtycNokbQYbWGzLs6JSDqGAv5PfF",
    ):
        """
        Initialize the KaminoProtocol client

        Args:
            cluster: The Solana cluster to use (default: "mainnet-beta")
            program_id: The Kamino lending program ID (default: KLend2g3cP87fffoy8q1mQqGKjrxjC8boSyAYavgmjD)
        """
        self.cluster = cluster
        self.program_id = program_id
        self.market_pubkey = market_pubkey

    @property
    def name(self) -> str:
        return self.PROTOCOL_NAME

    async def _get_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def get_pools(self, token_metadata_repo: TokenMetadataRepo) -> List[Pool]:
        """
        Fetch pools from Kamino API and convert to the internal Pool model
        """
        session = await self._get_session()
        url = f"{self.BASE_URL}/v1/markets/{self.market_pubkey}/reserves"
        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.json()

        reserves = data.get("reserves", [])
        kamino_pools = []

        for r in reserves:
            id = r.get("reserve")

            # No names available from data
            token = Token(
                address=r.get("liquidityTokenMint", ""),
                name="",
                symbol=r.get("liquidityToken", ""),
            )

            # Calculate TVL in USD
            tvl_usd = float(r.get("totalSupplyUsd"))

            # Calculate APR based on historical data
            apr_dict = await self._calculate_apr_metrics(reserve_pubkey=id)

            pool = Pool(
                id=id,
                chain=Chain.SOLANA,
                protocol=self.name,
                tokens=[token],
                type=PoolType.LENDING,
                TVL=str(tvl_usd),
                APRLastDay=apr_dict.get("APRLastDay"),
                APRLastWeek=apr_dict.get("APRLastWeek", None),
                APRLastMonth=apr_dict.get("APRLastMonth", None),
                isStableCoin=token.name in stablecoin_symbols,
                impermanentLossRisk=False,
            )

            kamino_pools.append(pool)

        return kamino_pools

    async def _fetch_metrics(
        self, reserve_pubkey: str, start_date: datetime, end_date: datetime, frequency: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch metrics data for a given reserve pubkey and time range
        """
        session = await self._get_session()
        url = f"{self.BASE_URL}/v1/reserves/{reserve_pubkey}/metrics"
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "frequency": frequency,
        }
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    def _calculate_apr_from_data(
        self, metrics_data: list, start_timestamp: datetime, end_timestamp: datetime
    ) -> Optional[float]:
        """
        Calculate APR from metrics data for a specific time range.

        Args:
            metrics_data: The metrics data returned from API
            start_timestamp: Start timestamp for filtering
            end_timestamp: End timestamp for filtering

        Returns:
            Median APR value for the specified time period
        """
        if not metrics_data:
            return None

        histories = metrics_data.get("history", [])
        if not histories:
            return None

        apr_values = []

        for history in histories:
            timestamp_str = history.get("timestamp")
            if not timestamp_str:
                continue

            # Parse timestamp (format: "2024-03-13T00:00:00.000Z")
            try:
                timestamp = datetime.strptime(
                    timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ"
                ).replace(tzinfo=UTC)
            except ValueError:
                # Try alternative format if needed
                try:
                    timestamp = datetime.strptime(
                        timestamp_str, "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=UTC)
                except ValueError:
                    continue

            # Check if timestamp is in the desired range
            if start_timestamp <= timestamp <= end_timestamp:
                metrics_history = history.get("metrics", {})
                apr = metrics_history.get("supplyInterestAPY")
                if apr is not None:
                    apr_values.append(apr)

        if not apr_values:
            return None

        # Return average APR value (x100 to convert to percentage)
        return statistics.mean(apr_values) * 100

    async def _calculate_apr_metrics(self, reserve_pubkey: str) -> Dict[str, Optional[float]]:
        """
        Calculate APR metrics for a given reserve pubkey for the last day, week, and month
        using a single API call.

        Args:
            reserve_pubkey: The reserve pubkey to fetch metrics for

        Returns:
            Dictionary containing APR metrics for different time periods:
            - APRLastDay: Median APR for the last day
            - APRLastWeek: Median APR for the last week
            - APRLastMonth: Median APR for the last month
        """
        # Calculate date ranges
        now = datetime.now(UTC)
        one_day_ago = now - timedelta(days=1)
        one_week_ago = now - timedelta(days=7)
        one_month_ago = now - timedelta(days=30)

        # Format dates for API - get data from now until one month ago
        end_date = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        month_start_date = one_month_ago.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # Fetch a month's worth of data with hourly frequency
        metrics_data = await self._fetch_metrics(
            reserve_pubkey, one_month_ago, now, "hour"
        )

        # Calculate APR for last day, week, and month using the same dataset
        results = {}
        results["APRLastDay"] = self._calculate_apr_from_data(
            metrics_data, one_day_ago, now
        )
        results["APRLastWeek"] = self._calculate_apr_from_data(
            metrics_data, one_week_ago, now
        )
        results["APRLastMonth"] = self._calculate_apr_from_data(
            metrics_data, one_month_ago, now
        )

        return results
