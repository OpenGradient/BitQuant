import unittest

from tokens.portfolio import PortfolioFetcher

class TestPortfolio(unittest.TestCase):
    def test_get_portfolio(self):
        portfolio = PortfolioFetcher()
        holdings = portfolio.get_portfolio("")
        print(holdings)

        self.assertGreater(len(holdings), 0)
