from datetime import datetime, timezone
from typing import Optional
from server.dynamodb_helpers import TableContext


class SwapTracker:
    """
    A class for tracking processed JUP swap transactions to prevent duplicates.
    """

    def __init__(self, get_table: callable):
        """
        Initialize the SwapTracker with a function that returns an async DynamoDB table.
        """
        self.get_table = get_table

    async def is_swap_processed(self, chain: str, txid: str) -> bool:
        """
        Check if a swap transaction has already been processed.

        Args:
            txid: The transaction ID to check

        Returns:
            True if the transaction has been processed, False otherwise
        """
        try:
            async with self.get_table() as table:
                response = await table.get_item(Key={"txid": f"{chain}:{txid}"})
                return "Item" in response
        except Exception as e:
            # If there's an error checking, assume it's not processed
            # This prevents blocking on DynamoDB errors
            return False

    async def mark_swap_processed(
        self,
        chain: str,
        txid: str,
        user_address: str,
        referral_reward: float,
        points_awarded: int,
    ) -> bool:
        """
        Mark a swap transaction as processed.

        Args:
            txid: The transaction ID
            user_address: The user's wallet address
            referral_reward: The calculated referral reward amount
            points_awarded: The points awarded to the user

        Returns:
            True if successfully marked as processed, False otherwise
        """
        try:
            async with self.get_table() as table:
                await table.put_item(
                    Item={
                        "txid": f"{chain}:{txid}",
                        "user_address": user_address,
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "referral_reward": referral_reward,
                        "points_awarded": points_awarded,
                    }
                )
                return True
        except Exception as e:
            return False

    async def get_swap_details(self, chain: str, txid: str) -> Optional[dict]:
        """
        Get details of a processed swap transaction.

        Args:
            txid: The transaction ID

        Returns:
            Dictionary with swap details if found, None otherwise
        """
        try:
            async with self.get_table() as table:
                response = await table.get_item(Key={"txid": f"{chain}:{txid}"})
                return response.get("Item")
        except Exception:
            return None
