import unittest

from defi.pools.defillama_source import DefiLlamaProtocols
from api.api_types import Chain, Pool, PoolQuery


class TestDefiLlamaSource(unittest.TestCase):
    def test_defillama(self):
        metrics = DefiLlamaProtocols()
        metrics.refresh_metrics()

        sol_pools = metrics.get_pools(
            PoolQuery(
                chain=Chain.SOLANA,
                protocols=["save"],
            )
        )

        print(sol_pools)
