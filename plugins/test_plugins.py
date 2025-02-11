import unittest

from plugins.navi.navi_plugin import NaviPlugin


class TestPlugins(unittest.TestCase):

    def test_plugin(self):
        plugin = NaviPlugin()

        plugin.initialize()
        pools = plugin.fetch_pools()

        self.assertTrue(len(pools) > 0, "Pools are empty")
        print(pools)
