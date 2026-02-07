import unittest
from langchain_core.runnables.config import RunnableConfig

from onchain.analytics.analytics_tools import (
    get_coingecko_price_data,
    portfolio_volatility,
    analyze_wallet_portfolio,
    max_drawdown_for_token,
    compare_assets,
    analyze_price_trend,
    get_token_market_info,
    get_top_coins_by_market_cap,
    get_global_market_overview,
    compare_tokens,
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


    # --- Tests for new market data tools ---

    def test_get_token_market_info(self):
        """Test fetching comprehensive market data for a token"""
        result = get_token_market_info.invoke({"token_symbol": "BTC"})

        self.assertNotIn("error", result)
        self.assertEqual(result["token_id"], "bitcoin")
        self.assertIn("market_data", result)
        self.assertIn("current_price_usd", result["market_data"])
        self.assertIn("market_cap_usd", result["market_data"])
        self.assertIn("price_change_percentage", result["market_data"])
        self.assertIn("ath", result["market_data"])
        self.assertIn("atl", result["market_data"])
        self.assertIn("supply", result["market_data"])
        self.assertIsNotNone(result["market_data"]["current_price_usd"])
        self.assertIsNotNone(result.get("market_cap_rank"))

    def test_get_token_market_info_invalid(self):
        """Test error handling for invalid token in market info"""
        result = get_token_market_info.invoke(
            {"token_symbol": "TOTALLYINVALIDTOKEN999"}
        )
        self.assertIn("error", result)

    def test_get_top_coins_by_market_cap(self):
        """Test fetching top coins by market cap"""
        result = get_top_coins_by_market_cap.invoke(
            {"vs_currency": "usd", "num_coins": 10}
        )

        self.assertNotIn("error", result)
        self.assertIn("coins", result)
        self.assertEqual(result["num_coins"], 10)
        self.assertEqual(len(result["coins"]), 10)

        # Verify first coin has expected fields
        first_coin = result["coins"][0]
        self.assertIn("rank", first_coin)
        self.assertIn("name", first_coin)
        self.assertIn("current_price", first_coin)
        self.assertIn("market_cap", first_coin)
        self.assertIn("price_change_percentage", first_coin)

    def test_get_top_coins_by_market_cap_with_category(self):
        """Test fetching top coins filtered by category"""
        result = get_top_coins_by_market_cap.invoke(
            {
                "vs_currency": "usd",
                "num_coins": 5,
                "category": "decentralized-finance-defi",
            }
        )

        self.assertNotIn("error", result)
        self.assertIn("coins", result)
        self.assertEqual(result["category"], "decentralized-finance-defi")
        self.assertTrue(len(result["coins"]) > 0)

    def test_get_global_market_overview(self):
        """Test fetching global market overview"""
        result = get_global_market_overview.invoke({})

        self.assertNotIn("error", result)
        self.assertIn("global_market", result)
        self.assertIn("defi_market", result)

        gm = result["global_market"]
        self.assertIn("total_market_cap_usd", gm)
        self.assertIn("btc_dominance", gm)
        self.assertIn("eth_dominance", gm)
        self.assertIn("active_cryptocurrencies", gm)
        self.assertIsNotNone(gm["total_market_cap_usd"])

        dm = result["defi_market"]
        self.assertIn("defi_market_cap", dm)
        self.assertIn("defi_to_eth_ratio", dm)

    def test_compare_tokens(self):
        """Test side-by-side fundamental comparison of tokens"""
        result = compare_tokens.invoke(
            {"token_symbols": ["BTC", "ETH", "SOL"]}
        )

        self.assertNotIn("error", result)
        self.assertIn("tokens", result)
        self.assertIn("relative_metrics", result)
        self.assertEqual(len(result["tokens"]), 3)

        # Verify relative metrics
        rm = result["relative_metrics"]
        self.assertIn("highest_market_cap", rm)
        self.assertIn("best_24h_performer", rm)
        self.assertIn("worst_24h_performer", rm)

        # Verify per-token data
        for token in result["tokens"]:
            self.assertIn("current_price", token)
            self.assertIn("market_cap", token)
            self.assertIn("price_change_percentage", token)

    def test_compare_tokens_too_few(self):
        """Test that compare_tokens rejects fewer than 2 tokens"""
        result = compare_tokens.invoke({"token_symbols": ["BTC"]})
        self.assertIn("error", result)
        self.assertIn("at least 2", result["error"])

    def test_compare_tokens_too_many(self):
        """Test that compare_tokens rejects more than 4 tokens"""
        result = compare_tokens.invoke(
            {"token_symbols": ["BTC", "ETH", "SOL", "ADA", "DOT"]}
        )
        self.assertIn("error", result)
        self.assertIn("at most 4", result["error"])


if __name__ == "__main__":
    unittest.main()
