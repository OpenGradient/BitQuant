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

    def test_binance_price_history(self):
        """Test Binance price history functionality"""
        print("\n========== Testing Binance Price History ==========")
        
        try:
            print("\nRetrieving Bitcoin price history from Binance...")
            btc_history = show_binance_price_history.invoke({"pair": "BTCUSDT"})
            
            # Validate response structure
            self.assertIsNotNone(btc_history)
            self.assertIn('current_price', btc_history)
            
            # Print price information with clear formatting
            print("\nBitcoin Price Information:")
            print(f"Current Price: ${btc_history.get('current_price', 0):,.2f}")
            
            # Print percentage changes
            print("\nPrice Changes:")
            if 'percent_change_24h' in btc_history and btc_history['percent_change_24h'] is not None:
                print(f"24h Change: {btc_history.get('percent_change_24h', 0):.2f}%")
            if 'percent_change_week' in btc_history and btc_history['percent_change_week'] is not None:
                print(f"7-day Change: {btc_history.get('percent_change_week', 0):.2f}%")
            if 'percent_change_month' in btc_history and btc_history['percent_change_month'] is not None:
                print(f"30-day Change: {btc_history.get('percent_change_month', 0):.2f}%")
            
            # Print statistics
            print("\nPrice Statistics:")
            for key, value in btc_history.get('stats', {}).items():
                if value is not None:
                    if key == 'volatility':
                        # Volatility is typically a ratio, not a currency amount
                        print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: ${value:,.2f}")
            
            print(f"\nData Points: {btc_history.get('data_points', 0)}")
            
            # Print a sample of the data (first and last 3 points)
            close_prices = btc_history.get('close_prices', [])
            dates = btc_history.get('dates', [])
            
            if close_prices and dates:
                if len(close_prices) > 6:
                    print("\nSample data:")
                    print("First 3 data points:")
                    for i in range(min(3, len(close_prices))):
                        print(f"  {dates[i]}: ${close_prices[i]:,.2f}")
                    
                    print("\nLast 3 data points:")
                    for i in range(max(0, len(close_prices)-3), len(close_prices)):
                        print(f"  {dates[i]}: ${close_prices[i]:,.2f}")
                else:
                    print("\nAll price data:")
                    for i in range(len(close_prices)):
                        print(f"  {dates[i]}: ${close_prices[i]:,.2f}")
                    
            print("\nBinance price history test completed successfully.")
            
        except Exception as e:
            print(f"Error testing Binance price history: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Binance price history test failed: {e}")
