from typing import List, Dict, Any
import json
import os

import requests

from api.api_types import Pool, Token, Chain, PoolType
from defi.pools.protocol import Protocol


class SaveProtocol(Protocol):

    PROTOCOL_NAME = "save"
    BASE_URL = "https://api.solend.fi/v1/"
    MAIN_MARKET_ADDRESS = "4UpD2fh7xH3VP9QQaXtsS1YY3bxzWhtfpks7FatyKvdY"

    token_list: Dict

    def __init__(self, chain_id: str = "solana"):
        """
        Initialize the SaveProtocol client

        Args:
            chain_id: The chain ID to use (default: "solana")
        """
        self.chain_id = chain_id

        # Load token list from static file
        token_list_path = os.path.join("static", "tokenlist.json")
        with open(token_list_path, "r") as f:
            self.token_list = json.load(f)

    @property
    def name(self) -> str:
        return self.PROTOCOL_NAME

    def get_pools(self) -> List[Pool]:
        url = f"{self.BASE_URL}reserves?ids={self.MAIN_MARKET_ADDRESS}&scope=all"
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for non-200 responses

        data = response.json()
        pools = self._convert_to_pools(data["results"])

        return sorted(pools, key = lambda p: int(p.TVL), reverse=True)

    def _convert_to_pools(self, save_pools: List[Dict[str, Any]]) -> List[Pool]:
        result = []

        for pool_data in save_pools:
            # Extract basic pool info
            reserve = pool_data.get("reserve", {})

            # Skip if essential data is missing
            if not reserve:
                continue

            pool_id = reserve.get("pubkey", "")

            is_stale = reserve.get("lastUpdate").get("stale") == 1
            if is_stale:
                continue

            # Get liquidity info
            liquidity = reserve.get("liquidity", {})
            token_address = liquidity.get("mintPubkey", "")
            token_decimals = liquidity.get("mintDecimals", 0)

            # Get Rates
            rates = pool_data.get("rates", {})
            supply_interest = rates.get("supplyInterest", "0")

            # Parse APR - handle values with commas
            try:
                supply_apr = float(supply_interest.replace(",", ""))
            except (ValueError, AttributeError):
                continue

            # Calculate TVL
            available_amount = int(liquidity.get("availableAmount", "0"))
            borrowed_amount_wads = int(liquidity.get("borrowedAmountWads", "0")) // (
                10**18
            )  # Convert from wads
            tvl_tokens = available_amount + borrowed_amount_wads

            # Convert to USD using market price
            market_price = float(liquidity.get("marketPrice", "0")) / (
                10**18
            )  # Assume price in wads format
            tvl_usd = (tvl_tokens / (10**token_decimals)) * market_price

            # Skip tokens that are not in the list
            if token_address not in self.token_list:
                continue

            # Get token information from the token list
            token_info = self.token_list.get(token_address)
            token_name = token_info.get("name")
            token_symbol = token_info.get("symbol")

            # Create token object
            token = Token(address=token_address, name=token_name, symbol=token_symbol)

            # Check if token is a stablecoin (based on token list symbols or known addresses)
            stable_symbols = ["USDC", "USDT", "DAI", "USDH", "USDS", "AUSD"]
            is_stablecoin = token_symbol in stable_symbols

            # Create Pool object
            pool = Pool(
                id=pool_id,
                chain=Chain.SOLANA,
                protocol=self.PROTOCOL_NAME,
                tokens=[token],
                type=PoolType.LENDING,
                TVL=str(int(tvl_usd)),
                APRLastDay=supply_apr,
                APRLastWeek=None,
                APRLastMonth=None,
                isStableCoin=is_stablecoin,
                impermanentLossRisk=False,  # Lending pools don't have IL risk
            )

            # Filter out unusable pools
            if available_amount <= 1000:
                continue
            if int(pool.TVL) <= 100_000:
                continue

            result.append(pool)

        return result
