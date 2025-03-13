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

        # import requests
        # import utils  # Assuming you have a similar utils module in Python

        markets = ["7u3HeHxYDLhnCoErrtycNokbQYbWGzLs6JSDqGAv5PfF"]
        kamino_pools = []

        for market in markets:
            response = requests.get(
                f"https://api.kamino.finance/kamino-market/{market}/reserves/metrics?env=mainnet-beta"
            )

            try:
                reserves = response.json()
            except Exception as e:
                print(f"Could not return pool response as JSON: {e} for response: {response}")
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
                tvl_usd = float(r.get("totalSupplyUsd")) - float(
                    r.get("totalBorrowUsd")
                )

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
    
    def _fetch_metrics(self, reserve_pubkey: str, start_date: str, end_date: str, frequency: str = "hour") -> list:
        """Function to fetch metrics data for a specific time range."""
        url = (
            f"https://api.kamino.finance/kamino-market/{self.market_pubkey}/reserves/{reserve_pubkey}/metrics/history"
            f"?env={self.cluster}&start={start_date}&end={end_date}&frequency={frequency}"
        )
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            return []
        
        try:
            return response.json()
        except Exception as e:
            print(f"Error returning response in JSON format: {e}")
            return []
    
    def _calculate_apr(self, metrics_data: list) -> Optional[float]:
        """Function to calculate APR from metrics data."""
        if not metrics_data:
            return None
        
        histories = metrics_data.get("history")
        apr_values = []
        for history in histories:
            metrics_history = history.get("metrics")
            apr_values.append(metrics_history.get("supplyInterestAPY"))

        if not apr_values:
            return None
        
        return statistics.mean(apr_values) * 100

    def _calculate_apr_metrics(self, reserve_pubkey: str) -> Dict[str, Optional[float]]:
        """
        Calculate APR metrics for a given reserve pubkey for the last day, week, and month.
        
        Args:
            reserve_pubkey: The reserve pubkey to fetch metrics for
            
        Returns:
            Dictionary containing APR metrics for different time periods:
            - APRLastDay: APR for the last day
            - APRLastWeek: APR for the last week (Can be None if data not available)
            - APRLastMonth: APR for the last month (Can be None if data not available)
        """        
        # Calculate date ranges
        now = datetime.now(UTC)
        one_day_ago = now - timedelta(days=1)
        one_week_ago = now - timedelta(days=7)
        one_month_ago = now - timedelta(days=30)
        
        # Format dates for API
        end_date = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        day_start_date = one_day_ago.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        week_start_date = one_week_ago.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        month_start_date = one_month_ago.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        # Initialize results
        results = {}
        
        # Calculate APR for last day (required)
        day_metrics = self._fetch_metrics(reserve_pubkey, day_start_date, end_date)
        results["APRLastDay"] = self._calculate_apr(day_metrics)
        
        # Calculate APR for last week (if available) with frequency set to day
        week_metrics = self._fetch_metrics(reserve_pubkey, week_start_date, end_date, "day")
        results["APRLastWeek"] = self._calculate_apr(week_metrics)
        
        # Calculate APR for last month (if available) wtih frequency set to day
        month_metrics = self._fetch_metrics(reserve_pubkey, month_start_date, end_date, "day")
        results["APRLastMonth"] = self._calculate_apr(month_metrics)
        
        return results
