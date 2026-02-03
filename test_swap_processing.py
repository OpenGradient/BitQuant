#!/usr/bin/env python3
"""
Test script for the JUP swap processing functionality.
This script demonstrates how to use the process_swap endpoint.
"""

import asyncio
import json
from server.swap_tracker import SwapTracker
from server.jup_validator import JUPValidator
from server.activity_tracker import ActivityTracker
from server.dynamodb_helpers import DatabaseManager


async def test_swap_processing():
    """
    Test the swap processing functionality.
    """
    print("Testing JUP Swap Processing System")
    print("=" * 50)

    # Initialize components (in a real scenario, these would be properly configured)
    database_manager = DatabaseManager()

    # Test data
    test_txid = "4yJx666ajeuHxE2VsBjitanVVbaiS5bmmQHxU188yeH2A2vgqPvzAefELtyQwgP9o9i44qxk2baJmZYCzQbh914Y"
    test_user_address = "test_user_123"
    test_input_amount = 4000000

    # Initialize services
    swap_tracker = SwapTracker(
        database_manager.table_context_factory("twoligma_processed_swaps")
    )
    jup_validator = JUPValidator()

    print(f"1. Testing swap validation for transaction: {test_txid}")

    # Test validation (this will now check actual on-chain data)
    validation_result = await jup_validator.validate_swap_transaction(test_txid)
    print(f"   Validation result: {validation_result}")

    if validation_result and validation_result.get("valid"):
        print(f"\n2. Testing referral reward from transaction")
        referral_reward = validation_result.get("referral_reward", 0.0)
        print(f"   Actual referral reward from transaction: {referral_reward}")

        # Test points calculation based on actual reward
        points = jup_validator.calculate_points_from_reward(referral_reward)
        print(f"   Points to award: {points}")
    else:
        print(f"\n2. Transaction validation failed - no referral reward to process")
        referral_reward = 0.0
        points = 0

    print(f"\n3. Testing duplicate prevention")

    # Test duplicate check
    is_processed = await swap_tracker.is_swap_processed(test_txid)
    print(f"   Is already processed: {is_processed}")

    if not is_processed:
        print(f"\n4. Simulating swap processing")

        # Mark as processed
        success = await swap_tracker.mark_swap_processed(
            test_txid, test_user_address, referral_reward, points
        )
        print(f"   Marked as processed: {success}")

        # Check again
        is_processed_after = await swap_tracker.is_swap_processed(test_txid)
        print(f"   Is now processed: {is_processed_after}")

    print(f"\n5. Testing API request format")

    # Example API request
    api_request = {
        "txid": test_txid,
        "swapResult": {
            "txid": test_txid,
            "inputAddress": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "outputAddress": "So11111111111111111111111111111111111111112",
            "inputAmount": test_input_amount,
            "outputAmount": 19818150,
        },
    }

    print(f"   Example request body:")
    print(json.dumps(api_request, indent=2))

    # Example API response
    api_response = {
        "success": True,
        "points_awarded": points,
        "referral_reward": referral_reward,
        "message": f"Successfully processed swap and awarded {points} points",
    }

    print(f"\n   Example response body:")
    print(json.dumps(api_response, indent=2))

    print(f"\n" + "=" * 50)
    print("Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_swap_processing())
