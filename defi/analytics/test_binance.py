import unittest
from defi.analytics.binance_tools import get_binance_price_history


class TestBinanceAPI(unittest.TestCase):
    """Test suite for Binance API functionality"""

    def test_btc_price_history_basic(self):
        """Test fetching Bitcoin price history with default parameters"""
        btc_history = get_binance_price_history.invoke(
            {"token_symbol": "BTC", "candle_interval": "1d", "num_candles": 30}
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
        eth_history = get_binance_price_history.invoke(
            {"token_symbol": "ETH", "candle_interval": "4h", "num_candles": 30}
        )

        self.assertIsNotNone(eth_history)
        self.assertIn("data", eth_history)

        data = eth_history.get("data", [])
        self.assertEqual(len(data), 30, "Should respect the limit parameter")

    def test_invalid_token_symbol(self):
        """Test error handling for invalid token symbol"""
        response = get_binance_price_history.invoke(
            {"token_symbol": "INVALIDTOKEN", "candle_interval": "1d", "num_candles": 30}
        )
        self.assertTrue(response.startswith("Error fetching Binance data"))

    def test_small_data_limit(self):
        """Test fetching price history with a small number of candles"""
        small_limit = 5
        small_history = get_binance_price_history.invoke(
            {"token_symbol": "BTC", "num_candles": small_limit}
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
        btc_history = get_binance_price_history.invoke(
            {"token_symbol": "BTC", "candle_interval": "1d", "num_candles": 1}
        )

        data = btc_history.get("data", [])
        columns = btc_history.get("columns", [])

        self.assertTrue(len(data) > 0)
        sample = data[0]

        # Verify all expected columns are present
        expected_columns = {"open", "high", "low", "close", "open_time", "close_time"}
        self.assertTrue(all(col in columns for col in expected_columns))

        # Verify data types
        price_indices = [columns.index(col) for col in ["open", "high", "low", "close"]]
        for idx in price_indices:
            self.assertTrue(float(sample[idx]) > 0, "Price values should be positive")

        time_indices = [columns.index(col) for col in ["open_time", "close_time"]]
        for idx in time_indices:
            self.assertTrue(
                isinstance(sample[idx], (int, float)), "Time values should be numeric"
            )


if __name__ == "__main__":
    unittest.main()
