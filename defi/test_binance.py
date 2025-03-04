import unittest
from agent.tools import show_binance_price_history

class TestBinanceAPI(unittest.TestCase):
    """Test suite for Binance API functionality"""
    
    def test_show_binance_price_history(self):
        """Test fetching price history from Binance"""
        print("\n========== Testing Binance Price History ==========")
        
        # Test case 1: Standard case with Bitcoin
        try:
            btc_history = show_binance_price_history.invoke({"pair": "BTCUSDT"})
            self.assertIsNotNone(btc_history)
            self.assertIn('current_price', btc_history)
            
            print("\nBitcoin Price Information:")
            print(f"Current Price: ${btc_history.get('current_price', 0):,.2f}")
            
            print("\nPrice Changes:")
            if 'percent_change_24h' in btc_history and btc_history['percent_change_24h'] is not None:
                print(f"24h Change: {btc_history.get('percent_change_24h', 0):.2f}%")
            if 'percent_change_week' in btc_history and btc_history['percent_change_week'] is not None:
                print(f"7-day Change: {btc_history.get('percent_change_week', 0):.2f}%")
            
            print("\nPrice Statistics:")
            for key, value in btc_history.get('stats', {}).items():
                if value is not None:
                    if key == 'volatility':
                        print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: ${value:,.2f}")
        except Exception as e:
            print(f"Error testing BTC: {e}")
            raise
        
        # Test case 2: Different time interval and crypto
        try:
            eth_history = show_binance_price_history.invoke({"pair": "ETHUSDT", "interval": "4h", "limit": 30})
            self.assertIsNotNone(eth_history)
            self.assertIn('current_price', eth_history)
            data_points = eth_history.get('data_points', 0)
            print(f"\nEthereum tested with 4h interval - {data_points} data points")
            self.assertEqual(data_points, 30, "Should respect the limit parameter")
        except Exception as e:
            print(f"Error testing ETH with custom interval: {e}")
            raise
        
        # Test case 3: Error handling - invalid pair
        try:
            # This should raise a ValueError
            show_binance_price_history.invoke({"pair": "INVALIDPAIR"})
            print("Warning: Invalid pair did not trigger an exception")
            self.fail("Should have raised an exception for invalid pair")
        except ValueError as e:
            print(f"\nCorrectly detected invalid pair: {str(e)}")
        except Exception as e:
            print(f"Unexpected error type for invalid pair: {type(e).__name__}: {str(e)}")
            self.fail(f"Expected ValueError but got {type(e).__name__}")
        
        # Test case 4: Test with very small limit
        try:
            small_limit = 5
            small_history = show_binance_price_history.invoke({"pair": "BTCUSDT", "limit": small_limit})
            actual_points = small_history.get('data_points', 0)
            print(f"\nTested with limit={small_limit}: got {actual_points} data points")
            self.assertEqual(actual_points, small_limit, 
                            f"Should return {small_limit} data points, got {actual_points}")
        except Exception as e:
            print(f"Error testing with small limit: {e}")
            raise
            
    def test_detailed_output_format(self):
        """Test the detailed output formatting of price data"""
        print("\n========== Testing show_binance_price_history('BTCUSDT') ==========")
        
        btc_history = show_binance_price_history.invoke({"pair": "BTCUSDT"})
        
        # Basic validation
        self.assertIsNotNone(btc_history)
        self.assertIn('current_price', btc_history)
        self.assertIn('close_prices', btc_history)
        self.assertIn('dates', btc_history)
        
        # Print formatted output
        print(f"Bitcoin Current Price: ${btc_history.get('current_price', 0):,.2f}")
        
        if 'percent_change_24h' in btc_history and btc_history['percent_change_24h'] is not None:
            print(f"24h Change: {btc_history.get('percent_change_24h', 0):.2f}%")
        if 'percent_change_week' in btc_history and btc_history['percent_change_week'] is not None:
            print(f"7-day Change: {btc_history.get('percent_change_week', 0):.2f}%")
        
        print("\nPrice Statistics:")
        for key, value in btc_history.get('stats', {}).items():
            if value is not None:
                if key == 'volatility':
                    print(f"  {key.replace('_', ' ').title()}: ${value:.2f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: ${value:,.2f}")
        
        # Show data points count
        data_points = len(btc_history.get('close_prices', []))
        print(f"\nData Points: {data_points}")
        
        # Show sample data (first 3 and last 3 points)
        dates = btc_history.get('dates', [])
        prices = btc_history.get('close_prices', [])
        
        if dates and prices and len(dates) == len(prices):
            print("\nSample data (first 3 and last 3 points):")
            # Print first 3
            for i in range(min(3, len(dates))):
                print(f"  {dates[i]}: ${prices[i]:,.2f}")
            
            if len(dates) > 6:
                print("  ...")
                
            # Print last 3
            for i in range(max(0, len(dates)-3), len(dates)):
                print(f"  {dates[i]}: ${prices[i]:,.2f}")

if __name__ == '__main__':
    unittest.main()
