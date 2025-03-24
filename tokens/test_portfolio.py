import unittest

from tokens.portfolio import PortfolioFetcher

class TestPortfolio(unittest.TestCase):
    def test_get_portfolio(self):
        portfolio = PortfolioFetcher()
        # Binance wallet
        holdings = portfolio.get_portfolio("9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM")
        print(holdings)

        self.assertGreater(len(holdings), 0)
