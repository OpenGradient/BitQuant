import unittest

from plugins.navi.navi_plugin import NaviPlugin
from plugins.defi_metrics import DeFiMetrics
from plugins.types import Chain


class TestPlugins(unittest.TestCase):

    def test_plugin(self):
        plugin = NaviPlugin()
        plugin.initialize()

        pools = plugin.fetch_pools()
        self.assertTrue(len(pools) > 0, "Pools are empty")

        print(pools)

    def test_defillama(self):
        metrics = DeFiMetrics()
        pools = metrics.refresh_metrics()

        sol = [p for p in pools if p.chain == Chain.SOLANA]
        print(sol)
