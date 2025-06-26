from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict

from server.config import MINER_TOKEN, DAILY_LIMIT_BYPASS_WALLETS
from server.dynamodb_helpers import TableContext


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
    rank: int  # Global rank based on points


class PointsConfig:
    POINTS_PER_MESSAGE = 1
    POINTS_PER_SUCCESSFUL_INVITE = 80
    DAILY_MESSAGE_LIMIT = 10


class ActivityTracker:
    """
    A class for tracking points for users.
    """

    def __init__(self, get_table: Callable[[], TableContext]):
        """
        Initialize the PointsTracker with a function that returns an async DynamoDB table.
        """
        self.get_table = get_table
        self._blocked_cache: Dict[str, datetime] = {}  # address -> expiration time

    def _get_today_end(self) -> datetime:
        """
        Get the end of today (start of tomorrow) in UTC.
        """
        today = datetime.now(timezone.utc).date()
        tomorrow = today + timedelta(days=1)
        return datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)

    def _is_blocked_cached(self, user_address: str) -> bool:
        """
        Check if user is cached as blocked and not expired.
        """
        expiration = self._blocked_cache.get(user_address)
        if expiration and datetime.now(timezone.utc) < expiration:
            return True
        elif expiration:
            # Remove expired entry
            del self._blocked_cache[user_address]
        return False

    def _cache_blocked_address(self, user_address: str):
        """
        Cache an address as blocked until the end of today.
        """
        self._blocked_cache[user_address] = self._get_today_end()

    async def increment_message_count(self, user_address: str) -> bool:
        """
        Increment the message count for a user.
        Returns True if the message was counted, False if the daily limit was reached.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Check cache first
        if self._is_blocked_cached(user_address):
            return False

        try:
            async with self.get_table() as table:
                response = await table.get_item(
                    Key={"user_address": user_address},
                    ProjectionExpression="message_count, last_message_date, daily_message_count",
                )
                item = response.get("Item", {})

                last_message_date = item.get("last_message_date")
                daily_message_count = item.get("daily_message_count", 0)

                # Reset daily count if it's a new day
                if last_message_date != today:
                    daily_message_count = 0

                # Check if daily limit reached
                if daily_message_count >= PointsConfig.DAILY_MESSAGE_LIMIT:
                    # Cache this blocked address
                    self._cache_blocked_address(user_address)
                    return False

                # Update both total and daily message counts, and points
                await table.update_item(
                    Key={"user_address": user_address},
                    UpdateExpression="SET message_count = if_not_exists(message_count, :zero) + :inc, "
                    "daily_message_count = :daily_count, "
                    "last_message_date = :today "
                    "ADD points :points_inc",
                    ExpressionAttributeValues={
                        ":inc": 1,
                        ":zero": 0,
                        ":daily_count": daily_message_count + 1,
                        ":today": today,
                        ":points_inc": PointsConfig.POINTS_PER_MESSAGE,
                    },
                )
                return True
        except Exception:
            return False

    async def increment_successful_invites(self, user_address: str):
        """
        Increment the successful invites count for a user.
        """
        async with self.get_table() as table:
            await table.update_item(
                Key={"user_address": user_address},
                UpdateExpression="ADD successful_invites :inc, points :points_inc",
                ExpressionAttributeValues={
                    ":inc": 1,
                    ":points_inc": PointsConfig.POINTS_PER_SUCCESSFUL_INVITE,
                },
            )

    async def get_activity_stats(self, user_address: str) -> ActivityStats:
        """
        Get the message count and successful invites count for a user.
        Returns ActivityStats with 0 for both counts if the user doesn't exist.
        """
        try:
            async with self.get_table() as table:
                # First get the user's stats
                response = await table.get_item(
                    Key={"user_address": user_address},
                    ProjectionExpression="message_count, successful_invites, daily_message_count, last_message_date, points",
                )
                item = response.get("Item", {})

                message_count = item.get("message_count", 0)
                successful_invites = item.get("successful_invites", 0)
                daily_message_count = item.get("daily_message_count", 0)
                last_message_date = item.get("last_message_date")
                points = item.get("points", 0)

                # Reset daily count if it's a new day
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if last_message_date != today:
                    daily_message_count = 0

                stats = ActivityStats(
                    message_count=message_count,
                    successful_invites=successful_invites,
                    points=points,
                    daily_message_count=daily_message_count,
                    daily_message_limit=PointsConfig.DAILY_MESSAGE_LIMIT,
                    rank=-1,
                )

                return stats
        except Exception:
            return ActivityStats(
                message_count=0,
                successful_invites=0,
                points=0,
                daily_message_count=0,
                daily_message_limit=PointsConfig.DAILY_MESSAGE_LIMIT,
                rank=-1,  # Return -1 for rank if there's an error
            )
