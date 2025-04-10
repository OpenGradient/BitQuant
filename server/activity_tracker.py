from boto3.resources.base import ServiceResource
from dataclasses import dataclass


@dataclass
class ActivityStats:
    """
    A class for tracking activity stats for users.
    """

    message_count: int
    successful_invites: int
    points: int


class ActivityTracker:
    """
    A class for tracking points for users.
    """

    def __init__(self, table: ServiceResource):
        """
        Initialize the PointsTracker with a DynamoDB table.
        """
        self.table = table

    def increment_message_count(self, user_address: str):
        """
        Increment the message count for a user.
        """
        self.table.update_item(
            Key={"user_address": user_address},
            UpdateExpression="ADD message_count :inc",
            ExpressionAttributeValues={":inc": 1},
        )

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
                ProjectionExpression="message_count, successful_invites",
            )
            item = response.get("Item", {})

            message_count = item.get("message_count", 0)
            successful_invites = item.get("successful_invites", 0)
            points = message_count + (successful_invites * 30)

            return ActivityStats(
                message_count=message_count,
                successful_invites=successful_invites,
                points=points,
            )
        except Exception:
            return ActivityStats(message_count=0, successful_invites=0, points=0)
