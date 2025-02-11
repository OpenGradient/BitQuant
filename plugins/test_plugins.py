import unittest

from plugins.navi.fetcher import fetch_pools, fetch_tokens


class TestPlugins(unittest.TestCase):

    def test_api(self):
        tokens = fetch_tokens()
        pools = fetch_pools()

        self.assertTrue(len(pools) > 0, "Pools are empty")
        self.assertTrue(len(tokens) > 0, "Tokens are empty")
