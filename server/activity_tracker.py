from boto3.resources.base import ServiceResource
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config import MINER_TOKEN


@dataclass
class ActivityStats:
    """
    A class for tracking activity stats for users.
    """

    message_count: int
    successful_invites: int
    points: int
    daily_message_count: int
    daily_message_limit: int


class ActivityTracker:
    """
    A class for tracking points for users.
    """

    DAILY_MESSAGE_LIMIT = 40

    def __init__(self, table: ServiceResource):
        """
        Initialize the PointsTracker with a DynamoDB table.
        """
        self.table = table

    def increment_message_count(self, user_address: str, miner_token: str = None) -> bool:
        """
        Increment the message count for a user.
        Returns True if the message was counted, False if the daily limit was reached.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        try:
            response = self.table.get_item(
                Key={"user_address": user_address},
                ProjectionExpression="message_count, last_message_date, daily_message_count",
            )
            item = response.get("Item", {})

            last_message_date = item.get("last_message_date")
            daily_message_count = item.get("daily_message_count", 0)

            # Reset daily count if it's a new day
            if last_message_date != today:
                daily_message_count = 0

            # Check if daily limit reached, except for Subnet miner wallet
            if daily_message_count >= self.DAILY_MESSAGE_LIMIT and miner_token != MINER_TOKEN :
                return False

            # Update both total and daily message counts
            self.table.update_item(
                Key={"user_address": user_address},
                UpdateExpression="SET message_count = if_not_exists(message_count, :zero) + :inc, "
                "daily_message_count = :daily_count, "
                "last_message_date = :today",
                ExpressionAttributeValues={
                    ":inc": 1,
                    ":zero": 0,
                    ":daily_count": daily_message_count + 1,
                    ":today": today,
                },
            )
            return True
        except Exception:
            return False

    def increment_successful_invites(self, user_address: str):
        """
        Increment the successful invites count for a user.
        """
        self.table.update_item(
            Key={"user_address": user_address},
            UpdateExpression="ADD successful_invites :inc",
            ExpressionAttributeValues={":inc": 1},
        )

    def get_activity_stats(self, user_address: str) -> ActivityStats:
        """
        Get the message count and successful invites count for a user.
        Returns ActivityStats with 0 for both counts if the user doesn't exist.
        """
        try:
            response = self.table.get_item(
                Key={"user_address": user_address},
                ProjectionExpression="message_count, successful_invites, daily_message_count, last_message_date",
            )
            item = response.get("Item", {})

            message_count = item.get("message_count", 0)
            successful_invites = item.get("successful_invites", 0)
            daily_message_count = item.get("daily_message_count", 0)
            last_message_date = item.get("last_message_date")
            points = message_count + (successful_invites * 30)

            # Reset daily count if it's a new day
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if last_message_date != today:
                daily_message_count = 0

            return ActivityStats(
                message_count=message_count,
                successful_invites=successful_invites,
                points=points,
                daily_message_count=daily_message_count,
                daily_message_limit=self.DAILY_MESSAGE_LIMIT,
            )
        except Exception:
            return ActivityStats(
                message_count=0,
                successful_invites=0,
                points=0,
                daily_message_count=0,
                daily_message_limit=self.DAILY_MESSAGE_LIMIT,
            )
