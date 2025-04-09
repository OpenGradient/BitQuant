from boto3.resources.base import ServiceResource
from dataclasses import dataclass


@dataclass
class Points:
    """
    A class for tracking points for users.
    """
    message_count: int
    successful_invites: int

class PointsTracker:

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
            Key={'user_address': user_address},
            UpdateExpression='ADD message_count :inc',
            ExpressionAttributeValues={':inc': 1}
        )

    def increment_successful_invites(self, user_address: str):
        """
        Increment the successful invites count for a user.
        """
        self.table.update_item(
            Key={'user_address': user_address},
            UpdateExpression='ADD successful_invites :inc',
            ExpressionAttributeValues={':inc': 1}
        )

    def get_points(self, user_address: str) -> Points:
        """
        Get the message count and successful invites count for a user.
        Returns Points with 0 for both counts if the user doesn't exist.
        """
        try:
            response = self.table.get_item(
                Key={'user_address': user_address},
                ProjectionExpression='message_count, successful_invites'
            )
            item = response.get('Item', {})
            return Points(
                message_count=item.get('message_count', 0),
                successful_invites=item.get('successful_invites', 0)
            )
        except Exception:
            return Points(message_count=0, successful_invites=0)