import unittest
from langchain_core.runnables.config import RunnableConfig

from onchain.analytics.analytics_tools import (
    get_coingecko_price_data,
    portfolio_volatility,
    analyze_wallet_portfolio,
    max_drawdown_for_token,
    compare_assets,
    analyze_price_trend,
    CandleInterval,
)


class TestFinancialAnalyticsTools(unittest.TestCase):
    def test_get_coingecko_price_history(self):
        response = get_coingecko_price_data(
            token_symbol="BTC",
            candle_interval=CandleInterval.DAY,
            num_candles=30,
        )

        self.assertNotIn("error", response)

    def test_analyze_price_trend(self):
        response = analyze_price_trend(
            token_symbol="BTC",
            candle_interval=CandleInterval.DAY,
            num_candles=90,
        )

        self.assertNotIn("error", response)
        print(response)

    def test_portfolio_volatility(self):
        response = portfolio_volatility(
            token_symbols=["BTC", "ETH"],
            token_quantities=[1, 2],
            candle_interval=CandleInterval.DAY,
            num_candles=30,
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
                            "symbol": "SOL",
                        },
                        {
                            "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                            "amount": 2,
                            "symbol": "USDC",
                        },
                    ]
                }
            ),
        )

        self.assertNotIn("error", response)
        print(response)

    def test_max_drawdown_for_token(self):
        response = max_drawdown_for_token.invoke(
            {
                "token_symbol": "BTC",
                "candle_interval": CandleInterval.DAY,
                "num_candles": 30,
            }
        )

        self.assertNotIn("error", response)
        print(response)

    def test_btc_price_history_basic(self):
        """Test fetching Bitcoin price history with default parameters"""
        btc_history = get_coingecko_price_data(
            token_symbol="BTC",
            candle_interval=CandleInterval.DAY,
            num_candles=30,
        )

        # Verify response structure
        self.assertIsNotNone(btc_history)
        self.assertIn("data", btc_history)
        self.assertIn("columns", btc_history)
        self.assertIn("token_symbol", btc_history)
        self.assertEqual(btc_history["token_symbol"], "BTC")

        # Verify data content
        data = btc_history.get("data", [])
        self.assertTrue(isinstance(data, list))
        self.assertTrue(len(data) > 0)
        self.assertEqual(len(data), 30)

    def test_eth_custom_interval(self):
        """Test fetching Ethereum price history with custom time interval"""
        eth_history = get_coingecko_price_data(
            token_symbol="ETH",
            candle_interval=CandleInterval.HOUR,
            num_candles=30,
        )

        self.assertIsNotNone(eth_history)
        self.assertIn("data", eth_history)

        data = eth_history.get("data", [])
        self.assertEqual(len(data), 30, "Should respect the limit parameter")

    def test_invalid_token_symbol(self):
        """Test error handling for invalid token symbol"""
        response = get_coingecko_price_data(
            token_symbol="INVALIDTOKEN",
            candle_interval=CandleInterval.DAY,
            num_candles=30,
        )
        self.assertIn("error", response)

    def test_small_data_limit(self):
        """Test fetching price history with a small number of candles"""
        small_limit = 5
        small_history = get_coingecko_price_data(
            token_symbol="BTC",
            num_candles=small_limit,
            candle_interval=CandleInterval.DAY,
        )

        self.assertIn("data", small_history)
        data = small_history.get("data", [])
        actual_points = len(data)
        self.assertEqual(
            actual_points,
            small_limit,
            f"Should return {small_limit} data points, got {actual_points}",
        )

    def test_candlestick_data_structure(self):
        """Test the structure and content of individual candlesticks"""
        btc_history = get_coingecko_price_data(
            token_symbol="BTC",
            candle_interval=CandleInterval.DAY,
            num_candles=1,
        )

        data = btc_history.get("data", [])
        columns = btc_history.get("columns", [])

        self.assertTrue(len(data) > 0)
        sample = data[0]

        # Verify all expected columns are present
        expected_columns = {"open", "high", "low", "close", "open_time"}
        self.assertTrue(all(col in columns for col in expected_columns))

        # Verify data types
        price_indices = [columns.index(col) for col in ["open", "high", "low", "close"]]
        for idx in price_indices:
            self.assertTrue(float(sample[idx]) > 0, "Price values should be positive")

        time_indices = [columns.index(col) for col in ["open_time"]]
        for idx in time_indices:
            self.assertTrue(
                isinstance(sample[idx], (int, float)), "Time values should be numeric"
            )

    def test_compare_assets(self):
        """Test basic functionality of compare_assets"""
        result = compare_assets.invoke(
            {
                "token_symbols": ["BTC", "ETH", "SOL"],
                "candle_interval": CandleInterval.DAY,
                "num_candles": 90,
            }
        )

        # Verify basic structure
        self.assertIsNotNone(result)
        self.assertIn("comparative_analysis", result)
        self.assertIn("investment_insights", result)

        print(result)


if __name__ == "__main__":
    unittest.main()
