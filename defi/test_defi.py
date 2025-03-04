import unittest

from defi.stats import DefiMetrics
from defi.types import Chain, Pool, PoolQuery
from agent.tools import show_binance_price_history


class TestPlugins(unittest.TestCase):

    def test_defillama(self):
        metrics = DefiMetrics()
        metrics.refresh_metrics()

        sol_pools = metrics.get_pools(
            PoolQuery(
                chain=Chain.SOLANA,
                protocols=["save"],
            )
        )

        print(sol_pools)

        # Test Binance price history tool
        try:
            print("\n========== Testing show_binance_price_history('BTCUSDT') ==========")
            btc_history = show_binance_price_history.invoke({"pair": "BTCUSDT"})
            self.assertIsNotNone(btc_history)
            self.assertIn('current_price', btc_history)
            
            print(f"Bitcoin Current Price: ${btc_history.get('current_price', 0):,.2f}")
            if 'percent_change_24h' in btc_history and btc_history['percent_change_24h'] is not None:
                print(f"24h Change: {btc_history.get('percent_change_24h', 0):.2f}%")
            if 'percent_change_week' in btc_history and btc_history['percent_change_week'] is not None:
                print(f"7-day Change: {btc_history.get('percent_change_week', 0):.2f}%")
            
            print("\nPrice Statistics:")
            for key, value in btc_history.get('stats', {}).items():
                if value is not None:
                    print(f"  {key.replace('_', ' ').title()}: ${value:,.2f}")
            
            print(f"\nData Points: {btc_history.get('data_points', 0)}")
            
            # Print a sample of the data (first and last 3 points)
            close_prices = btc_history.get('close_prices', [])
            dates = btc_history.get('dates', [])
            
            if close_prices and dates:
                if len(close_prices) > 6:
                    print("\nSample data (first 3 and last 3 points):")
                    for i in range(3):
                        print(f"  {dates[i]}: ${close_prices[i]:,.2f}")
                    print("  ...")
                    for i in range(-3, 0):
                        print(f"  {dates[i]}: ${close_prices[i]:,.2f}")
                else:
                    print("\nAll price data:")
                    for i in range(len(close_prices)):
                        print(f"  {dates[i]}: ${close_prices[i]:,.2f}")
        except Exception as e:
            print(f"Warning: Failed to get Binance price history: {e}")
