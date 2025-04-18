import boto3
import os
from typing import Dict, Any

from activity_tracker import PointsConfig


def migrate_points():
    """
    Migrate existing user actions to points.
    This script will:
    1. Scan the activity table
    2. Calculate points for each user based on their historical actions
    3. Update their points in the table
    """
    # Initialize DynamoDB
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=os.environ.get("AWS_REGION"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    activity_table = dynamodb.Table("twoligma_activity")

    # Scan the table
    response = activity_table.scan()
    items = response.get("Items", [])

    # Process each user
    for item in items:
        user_address = item["user_address"]
        message_count = item.get("message_count", 0)
        successful_invites = item.get("successful_invites", 0)

        # Calculate points based on historical actions
        points = (
            message_count * PointsConfig.POINTS_PER_MESSAGE
            + successful_invites * PointsConfig.POINTS_PER_SUCCESSFUL_INVITE
        )

        # Update the user's points
        try:
            activity_table.update_item(
                Key={"user_address": user_address},
                UpdateExpression="SET points = :points",
                ExpressionAttributeValues={":points": points},
            )
            print(f"Updated points for {user_address}: {points}")
        except Exception as e:
            print(f"Error updating points for {user_address}: {e}")


if __name__ == "__main__":
    migrate_points()
