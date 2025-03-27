from typing import List, Dict, Any

import requests

from api.api_types import Pool, Token, Chain, PoolType
from defi.pools.protocol import Protocol
from tokens.metadata import TokenMetadataRepo


class SaveProtocol(Protocol):
    PROTOCOL_NAME = "save"
    BASE_URL = "https://api.solend.fi/v1/"
    MAIN_MARKET_ADDRESS = "4UpD2fh7xH3VP9QQaXtsS1YY3bxzWhtfpks7FatyKvdY"

    def __init__(self, chain_id: str = "solana"):
        """
        Initialize the SaveProtocol client

        Args:
            chain_id: The chain ID to use (default: "solana")
        """
        self.chain_id = chain_id

    @property
    def name(self) -> str:
        return self.PROTOCOL_NAME

    def get_pools(self, token_metadata_repo: TokenMetadataRepo) -> List[Pool]:
        url = f"{self.BASE_URL}reserves?ids={self.MAIN_MARKET_ADDRESS}&scope=all"
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for non-200 responses

        data = response.json()
        pools = self._convert_to_pools(data["results"], token_metadata_repo)

        return sorted(pools, key=lambda p: int(p.TVL), reverse=True)

    def _convert_to_pools(
        self, save_pools: List[Dict[str, Any]], token_metadata_repo: TokenMetadataRepo
    ) -> List[Pool]:
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

            # Get token information from the token list
            token_info = token_metadata_repo.get_token_metadata(token_address)
            if token_info is None:
                continue

            # Create token object
            token = Token(
                address=token_address, name=token_info.name, symbol=token_info.symbol
            )

            # Check if token is a stablecoin (based on token list symbols or known addresses)
            stable_symbols = ["USDC", "USDT", "DAI", "USDH", "USDS", "AUSD"]
            is_stablecoin = token_info.symbol in stable_symbols

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
            if available_amount <= 1000 or int(pool.TVL) <= 100_000:
                continue

            result.append(pool)

        return result
