from typing import List, Dict
import requests

from plugins.plugin import Plugin
from plugins.types import Pool, Token

POOL_FETCH_ENDPOINT = "https://open-api.naviprotocol.io/api/navi/pools"
POOL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

TOKEN_LIST_ENDPOINT = (
    "https://aggregator-api.naviprotocol.io/coins/support-token-list?page=1&pageSize=50"
)
TOKEN_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9,hu;q=0.8",
    "origin": "https://www.navi.ag",
    "priority": "u=1, i",
    "referer": "https://www.navi.ag/",
    "sec-ch-ua": '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
}


class NaviPlugin(Plugin):

    tokens: Dict

    def initialize(self):
        response = requests.get(url=TOKEN_LIST_ENDPOINT, headers=TOKEN_HEADERS)
        response.raise_for_status()

        tokens = response.json()["data"]["list"]
        tokens_by_address: Dict = {
            t["address"]: {"address": t["address"], "symbol": t["symbol"]}
            for t in tokens
        }

        self.tokens = tokens_by_address

    def fetch_pools(self) -> List[Pool]:
        response = requests.get(url=POOL_FETCH_ENDPOINT, headers=POOL_HEADERS)
        response.raise_for_status()

        raw_pools = response.json()["data"]
        pools = [self.convert_to_pool(p) for p in raw_pools]

        return pools

    def convert_to_pool(self, pool: Dict) -> Pool:
        coin_type = f"0x{pool["coinType"]}"
        token_symbol = (
            self.tokens[coin_type]["symbol"] if coin_type in self.tokens else coin_type
        )

        return Pool(
            id=token_symbol,
            tokens=[
                Token(
                    symbol=token_symbol,
                    price=NaviPlugin.calc_token_price(
                        oracle_price=pool["oracle"]["price"],
                        oracle_decimals=pool["oracle"]["decimal"],
                    ),
                )
            ],
            TVL=NaviPlugin.format_usd(
                NaviPlugin.calc_dollar_amount(
                    amount=pool["totalSupplyAmount"],
                    oracle_price=pool["oracle"]["price"],
                    oracle_decimals=pool["oracle"]["decimal"],
                )
            ),
            APRLastDay=pool["supplyIncentiveApyInfo"]["apy"],
            APRLastWeek=None,
            APRLastMonth=None,
            protocol="Navi",
        )

    @staticmethod
    def calc_token_price(oracle_price: str, oracle_decimals: int) -> float:
        """Calculates the dollar value of a token"""
        return (1 / 10 ** float(oracle_decimals)) * float(oracle_price)

    @staticmethod
    def calc_dollar_amount(
        amount: str, oracle_price: str, oracle_decimals: int
    ) -> float:
        """Calculate dollar value of an amount using oracle price and decimals"""
        return (float(amount) / 10 ** float(oracle_decimals)) * float(oracle_price)

    @staticmethod
    def format_usd(amount: float) -> str:
        millions = amount / 1_000_000
        if millions >= 1:
            return f"${millions:.1f}M"
        else:
            # For amounts less than 1M, show full number with commas
            return "${:,.0f}".format(amount)
