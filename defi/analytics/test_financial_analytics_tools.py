import unittest
from langgraph.graph.graph import RunnableConfig

from defi.analytics.financial_analytics_tools import (
    get_binance_price_history,
    portfolio_value,
    portfolio_volatility,
    portfolio_summary,
    analyze_volatility_trend,
    analyze_wallet_portfolio,
    max_drawdown_for_token,
)


class TestFinancialAnalyticsTools(unittest.TestCase):

    def test_get_binance_price_history(self):
        response = get_binance_price_history.invoke(
            {"token_symbol": "BTC", "candle_interval": "1d", "num_candles": 30}
        )

        self.assertNotIn("error", response)
        print(response)

    def test_portfolio_value(self):
        response = portfolio_value.invoke(
            {
                "token_symbols": ["BTC", "ETH"],
                "token_quantities": [1, 2],
                "candle_interval": "1d",
                "num_candles": 30,
            }
        )

        self.assertNotIn("error", response)
        print(response)

    def test_portfolio_volatility(self):
        response = portfolio_volatility.invoke(
            {
                "token_symbols": ["BTC", "ETH", "USDT"],
                "token_quantities": [1, 2, 3],
                "candle_interval": "1d",
                "num_candles": 30,
            }
        )

        self.assertNotIn("error", response)
        print(response)

    def test_portfolio_summary(self):
        response = portfolio_summary.invoke(
            {
                "token_symbols": ["BTC", "ETH"],
                "token_quantities": [1, 2],
            }
        )

        self.assertNotIn("error", response)
        print(response)

    def test_analyze_volatility_trend(self):
        response = analyze_volatility_trend.invoke(
            {"token_symbol": "BTC", "candle_interval": "1d", "num_candles": 30}
        )

        self.assertNotIn("error", response)
        print(response)

    def test_analyze_wallet_portfolio(self):
        response = analyze_wallet_portfolio.invoke(
            input={},
            config=RunnableConfig(
                configurable={
                    "tokens": [
                        {
                            "address": "So11111111111111111111111111111111111111112",
                            "amount": 10,
                        },
                        {
                            "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                            "amount": 2,
                        },
                    ]
                }
            ),
        )

        self.assertNotIn("error", response)
        print(response)

    def test_max_drawdown_for_token(self):
        response = max_drawdown_for_token.invoke(
            {"token_symbol": "BTC", "candle_interval": "1d", "num_candles": 30}
        )

        self.assertNotIn("error", response)
        print(response)


if __name__ == "__main__":
    unittest.main()
