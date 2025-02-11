import unittest

from plugins.navi.fetcher import fetch_pools


class TestPlugins(unittest.TestCase):

    def test_navi(self):
        pools = fetch_pools()

        print(pools)
        self.assertTrue(len(pools) > 0, "Pools are empty")
