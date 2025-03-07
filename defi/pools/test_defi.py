import unittest

from defi.pools.defillama_metrics import DefiLlamaMetrics
from api.types import Chain, Pool, PoolQuery


class TestPlugins(unittest.TestCase):

    def test_defillama(self):
        metrics = DefiLlamaMetrics()
        metrics.refresh_metrics()

        sol_pools = metrics.get_pools(
            PoolQuery(
                chain=Chain.SOLANA,
                protocols=["save"],
            )
        )

        print(sol_pools)
