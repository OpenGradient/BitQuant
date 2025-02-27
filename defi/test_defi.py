import unittest

from defi.stats import DefiMetrics
from defi.types import Chain, Pool, PoolQuery


class TestPlugins(unittest.TestCase):

    def test_defillama(self):
        metrics = DefiMetrics()
        metrics.refresh_metrics()

        sol_pools = metrics.get_pools(
            PoolQuery(
                chain=Chain.SOLANA,
                isStableCoin=True,
                protocols=["kamino-liquidity", "kamino-lend", "save"],
            )
        )

        print(sol_pools)
