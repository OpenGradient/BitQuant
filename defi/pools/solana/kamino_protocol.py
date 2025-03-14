from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import requests
from datetime import datetime, timedelta, UTC
import statistics

from api.api_types import Pool, Token, Chain, PoolType
from defi.pools.protocol import Protocol
from defi.pools.solana.constants import stablecoin_symbols


class KaminoProtocol(Protocol):
    """
    Implementation of Kamino Protocol API for lending pools
    API docs: https://api.kamino.finance
    """

    PROTOCOL_NAME = "kamino"
    BASE_URL = "https://api.kamino.finance"

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

    def get_pools(self) -> List[Pool]:
        """
        Fetch lending pools from Kamino API and convert to the internal Pool model
        """
        markets = ["7u3HeHxYDLhnCoErrtycNokbQYbWGzLs6JSDqGAv5PfF"]
        kamino_pools = []

        for market in markets:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; OpenGradient/1.0)",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.5",
            }
            response = requests.get(
                f"https://api.kamino.finance/kamino-market/{market}/reserves/metrics?env=mainnet-beta",
                headers=headers,
            )

            try:
                reserves = response.json()
            except Exception as e:
                print(
                    f"Could not return pool response as JSON: {e} for response: {response}"
                )
                continue

            for r in reserves:
                id = r.get("reserve")

                # No names available from data
                token = Token(
                    address=r.get("liquidityTokenMint", ""),
                    name="",
                    symbol=r.get("liquidityToken", ""),
                )

                # Calculate TVL in USD
                # TODO (Kyle): Confirm if this is the right calculation (maybe just use totalSupplyUsd?)
                tvl_usd = float(r.get("totalSupplyUsd"))

                # Calculate APR based on historical data
                apr_dict = self._calculate_apr_metrics(reserve_pubkey=id)

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

    def _fetch_metrics(
        self,
        reserve_pubkey: str,
        start_date: str,
        end_date: str,
        frequency: str = "hour",
    ) -> list:
        """Function to fetch metrics data for a specific time range."""
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; OpenGradient/1.0)",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.5",
        }
        url = (
            f"https://api.kamino.finance/kamino-market/{self.market_pubkey}/reserves/{reserve_pubkey}/metrics/history"
            f"?env={self.cluster}&start={start_date}&end={end_date}&frequency={frequency}"
        )

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            return []

        try:
            return response.json()
        except Exception as e:
            print(f"Error returning response in JSON format: {e}")
            return []

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

    def _calculate_apr_metrics(self, reserve_pubkey: str) -> Dict[str, Optional[float]]:
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
        metrics_data = self._fetch_metrics(
            reserve_pubkey, month_start_date, end_date, "hour"
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
