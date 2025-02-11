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
    print(pools)

    return []


def convert_to_pool(pool: Dict) -> Pool:
    return Pool(name="", TVL="", APRLastDay=0, APRLastWeek=0, APRLastMonth=0)
