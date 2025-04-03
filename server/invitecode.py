from typing import Optional
import secrets
import logging
from datetime import datetime
from boto3.resources.base import ServiceResource

logger = logging.getLogger(__name__)


class InviteCodeManager:
    """Manages invite codes for whitelisting users."""

    MAX_UNUSED_CODES = 1000

    def __init__(self, table: ServiceResource):
        self.table = table

    def generate_invite_code(self, creator_address: str) -> Optional[str]:
        """Generate a new invite code for a whitelisted user."""
        try:
            # Check current number of unused codes
            response = self.table.query(
                IndexName="creator_address-index",
                KeyConditionExpression="creator_address = :addr",
                FilterExpression="#used = :used",
                ExpressionAttributeNames={"#used": "used"},
                ExpressionAttributeValues={":addr": creator_address, ":used": False},
            )

            unused_codes = response.get("Items", [])
            if len(unused_codes) >= self.MAX_UNUSED_CODES:
                logger.warning(
                    f"User {creator_address} has reached the limit of {self.MAX_UNUSED_CODES} unused invite codes"
                )
                return None

            # Generate a random 14-character code (10 bytes ≈ 14 characters in base64url)
            invite_code = secrets.token_urlsafe(10)

            # Save to DynamoDB
            self.table.put_item(
                Item={
                    "code": invite_code,
                    "creator_address": creator_address,
                    "created_at": int(datetime.now().timestamp()),
                    "used": False,
                    "used_by": None,
                    "used_at": None,
                }
            )

            return invite_code
        except Exception as e:
            logger.error(f"Error generating invite code: {e}")
            return None

    def use_invite_code(self, code: str, user_address: str) -> bool:
        """Use an invite code to whitelist a new user."""
        try:
            # Get the invite code
            response = self.table.get_item(Key={"code": code})
            if "Item" not in response:
                return False

            invite = response["Item"]

            # Check if code is already used
            if invite.get("used", False):
                return False

            # Update the invite code as used
            self.table.update_item(
                Key={"code": code},
                UpdateExpression="SET #used = :used, used_by = :used_by, used_at = :used_at",
                ExpressionAttributeNames={"#used": "used"},
                ExpressionAttributeValues={
                    ":used": True,
                    ":used_by": user_address,
                    ":used_at": int(datetime.now().timestamp()),
                },
            )

            return True
        except Exception as e:
            logger.error(f"Error using invite code: {e}")
            return False

    def get_invite_stats(self, address: str) -> dict:
        """Get invite statistics for a user."""
        try:
            # Get all codes created by the user
            response = self.table.query(
                IndexName="creator_address-index",
                KeyConditionExpression="creator_address = :addr",
                ExpressionAttributeValues={":addr": address},
            )

            codes = response.get("Items", [])

            # Count used and unused codes
            used_codes = [code for code in codes if code.get("used", False)]

            return {
                "total_codes": len(codes),
                "used_codes": len(used_codes),
                "unused_codes": len(codes) - len(used_codes),
            }
        except Exception as e:
            logger.error(f"Error getting invite stats: {e}")
            return {"total_codes": 0, "used_codes": 0, "unused_codes": 0}
