import unittest

from defi.stats import DefiMetrics
from api.types import Chain, Pool, PoolQuery


class TestPlugins(unittest.TestCase):

    def test_defillama(self):
        metrics = DefiMetrics()
        metrics.refresh_metrics()

        sol_pools = metrics.get_pools(
            PoolQuery(
                chain=Chain.SOLANA,
                protocols=["save"],
            )
        )

        print(sol_pools)
