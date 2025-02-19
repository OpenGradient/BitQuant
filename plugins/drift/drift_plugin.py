from typing import List, Dict
import requests

from plugins.plugin import Plugin
from plugins.types import Pool, Token

from driftpy.drift_client import DriftClient


class DriftPlugin(Plugin):

    drift_client: DriftClient

    def initialize(self):
        self.drift_client = DriftClient(
            connection,
            wallet,
            "mainnet",
            perp_market_indexes=perp_markets,
            spot_market_indexes=spot_market_indexes,
            oracle_infos=oracle_infos,
            account_subscription=AccountSubscriptionConfig("demo"),
        )

    def fetch_pools(self) -> List[Pool]:
        return []
