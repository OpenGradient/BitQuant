from typing import List, Dict
import requests

from plugins.types import Pool

POOL_FETCH_ENDPOINT = "https://open-api.naviprotocol.io/api/navi/pools"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


def fetch_pools() -> List[Pool]:
    response = requests.get(url=POOL_FETCH_ENDPOINT, headers=HEADERS)
    response.raise_for_status()

    raw_pools = response.json()["data"]
    pools = [convert_to_pool(p) for p in raw_pools]

    return pools


def convert_to_pool(pool: Dict) -> Pool:
    print(pool)

    return Pool(
        name="",
        TVL=format_usd(
            calc_dollar_amount(
                amount=pool["totalSupplyAmount"],
                oracle_price=pool["oracle"]["price"],
                oracle_decimals=pool["oracle"]["decimal"],
            )
        ),
        APRLastDay=pool["supplyIncentiveApyInfo"]["apy"],
        APRLastWeek=0.0,
        APRLastMonth=0.0,
    )


def calc_dollar_amount(amount: str, oracle_price: str, oracle_decimals: int) -> float:
    """Calculate dollar value of an amount using oracle price and decimals"""
    return (float(amount) / 10 ** float(oracle_decimals)) * float(oracle_price)


def format_usd(amount: float) -> str:
    formatted = "${:,}".format(amount)

    return formatted
