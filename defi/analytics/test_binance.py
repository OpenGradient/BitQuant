import unittest
from defi.analytics.binance_tools import get_binance_price_history

class TestBinanceAPI(unittest.TestCase):
    """Test suite for Binance API functionality"""
    
    def test_show_binance_price_history(self):
        """Test fetching raw price history from Binance"""
        print("\n========== Testing Binance Price History (Raw API Data) ==========")
        
        # Test case 1: Standard case with Bitcoin
        try:
            btc_history = get_binance_price_history(pair="BTCUSDT")
            self.assertIsNotNone(btc_history)
            
            # Verify the structure of the response
            self.assertIn('data', btc_history)
            self.assertIn('columns', btc_history)
            self.assertIn('pair', btc_history)
            self.assertEqual(btc_history['pair'], "BTCUSDT")
            
            # Verify the data is a list of candlesticks
            data = btc_history.get('data', [])
            self.assertTrue(isinstance(data, list))
            self.assertTrue(len(data) > 0)
            
            # Print some info about the data
            print(f"\nBitcoin Price Data Summary:")
            print(f"Trading Pair: {btc_history.get('pair')}")
            print(f"Interval: {btc_history.get('interval')}")
            print(f"Number of candlesticks: {len(data)}")
            
            # Display column information
            columns = btc_history.get('columns', [])
            print("\nData Structure:")
            print(f"Columns: {', '.join(columns)}")
            
            # Display a sample candlestick
            if data:
                print("\nSample Candlestick (First data point):")
                sample = data[0]
                for i, col in enumerate(columns):
                    if i < len(sample):
                        if col in ['open', 'high', 'low', 'close']:
                            print(f"  {col}: ${float(sample[i]):.2f}")
                        elif col in ['open_time', 'close_time']:
                            print(f"  {col}: {sample[i]} (ms)")
                        else:
                            print(f"  {col}: {sample[i]}")
                
                # Calculate and show current price from the last candlestick
                if len(data) > 0 and len(data[-1]) > 4:
                    current_price = float(data[-1][4])  # Close price
                    print(f"\nCurrent Price (from last candlestick): ${current_price:.2f}")
        
        except Exception as e:
            print(f"Error testing BTC: {e}")
            raise
        
        # Test case 2: Different time interval and crypto
        try:
            eth_history = get_binance_price_history(pair="ETHUSDT", interval="4h", limit=30)
            self.assertIsNotNone(eth_history)
            self.assertIn('data', eth_history)
            
            data = eth_history.get('data', [])
            print(f"\nEthereum tested with 4h interval - {len(data)} data points")
            self.assertEqual(len(data), 30, "Should respect the limit parameter")
        except Exception as e:
            print(f"Error testing ETH with custom interval: {e}")
            raise
        
        # Test case 3: Error handling - invalid pair
        try:
            # This should raise a ValueError
            get_binance_price_history(pair="INVALIDPAIR")
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
            small_history = get_binance_price_history(pair="BTCUSDT", limit=small_limit)
            self.assertIn('data', small_history)
            
            data = small_history.get('data', [])
            actual_points = len(data)
            print(f"\nTested with limit={small_limit}: got {actual_points} data points")
            self.assertEqual(actual_points, small_limit, 
                            f"Should return {small_limit} data points, got {actual_points}")
        except Exception as e:
            print(f"Error testing with small limit: {e}")
            raise

if __name__ == '__main__':
    unittest.main()