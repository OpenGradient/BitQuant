"""
Simple debug test for COW validator.
This test makes actual API calls to the COW protocol and prints results.
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.cow_validator import COWValidator


@pytest.mark.asyncio
async def test_cow_validator_debug():
    """Test the COW validator with real API calls."""

    # Create a mock token metadata repo (you can replace this with a real one)
    class MockTokenMetadataRepo:
        async def get_token_metadata(self, token_address, network):
            # Mock metadata for common tokens
            mock_metadata = {
                "0xA0b86a33E6441b8c4C8C0E1234567890abcdef123": {"price": 1.0},  # USDC
                "0x6810e776880c02933d47db1b9fc05908e5386b96": {
                    "price": 1.0
                },  # Example token
                "0x0000000000000000000000000000000000000000": {"price": 2500.0},  # ETH
            }

            if token_address in mock_metadata:

                class MockTokenMetadata:
                    def __init__(self, price):
                        self.price = price

                return MockTokenMetadata(mock_metadata[token_address]["price"])
            return None

        async def search_token(self, token_symbol, network):
            # Mock ETH search
            if token_symbol == "ETH" and network == "ethereum":

                class MockTokenMetadata:
                    def __init__(self, price):
                        self.price = price

                return MockTokenMetadata(4000.0)
            return None

    # Initialize the validator
    token_metadata_repo = MockTokenMetadataRepo()
    validator = COWValidator(token_metadata_repo)

    print("🔍 COW Validator Debug Test")
    print("=" * 50)

    # Test with a sample order UID (you can replace this with a real one)
    # Note: This is a placeholder - you'll need to use a real order UID from COW protocol
    # test_order_uid = "0xa6c1b45d6f11b799b81a458f09e5ac617f873b81a9eb802adcbec3decc6f508dba3cb449bd2b4adddbc894d8697f5170800eadecffffffff"
    # test_order_uid = "0xb8af40f5f102d2730a6f664c24d156c8f9881c53beeda9ac5d6f1a775b9d62f9d562e50db16e978593aa8282ef1a2ceedf5978b368e6a6f3"
    test_order_uid = "0xcc9370eeaa198990d1d917461438d943a9fd44b9f2471c23d64503d8fc951ec0ba3cb449bd2b4adddbc894d8697f5170800eadecffffffff"

    print(f"Testing order UID: {test_order_uid}")
    print()

    # Test on Base network (where this trade exists)
    print("🌐 Testing on Base network...")
    try:
        result_base = await validator.validate_swap_order(test_order_uid, "base")
        if result_base:
            print("✅ Base validation successful!")
            print(f"   Valid: {result_base['valid']}")
            print(f"   Is COW Order: {result_base['is_cow_order']}")
            print(f"   Sell Token: {result_base.get('sell_token', 'N/A')}")
            print(f"   Sell Token Price: {result_base.get('sell_token_price', 'N/A')}")
            print(f"   Buy Token: {result_base.get('buy_token', 'N/A')}")
            print(f"   Sell Amount: {result_base.get('sell_amount', 'N/A')}")
            print(
                f"   Executed Amount: {result_base.get('executed_sell_amount', 'N/A')}"
            )
            print(
                f"   Referral Reward (USDC): ${result_base.get('referral_reward_usdc', 0):.6f}"
            )
            print(f"   Points: {result_base.get('points_awarded', 0)}")
        else:
            print("❌ Base validation failed - order not found or invalid")
    except Exception as e:
        print(f"❌ Error testing Base: {e}")

    print()

    # Test points calculation
    print("🎯 Testing points calculation...")
    test_rewards = [0.001, 0.01, 0.1, 1.0, 10.0]
    for reward in test_rewards:
        points = validator.calculate_points_from_reward(reward)
        print(f"   ${reward:.3f} USDC -> {points} points")

    # Close the validator
    await validator.close()
    print()
    print("✅ Test completed!")
