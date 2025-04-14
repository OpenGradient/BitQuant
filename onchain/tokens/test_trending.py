import unittest

from onchain.tokens.trending import get_trending_tokens_from_coingecko


class TestTrending(unittest.TestCase):
    def test_get_trending_tokens(self):
        tokens = get_trending_tokens_from_coingecko()

        self.assertGreater(len(tokens), 0)
        print(tokens)
