from typing import Optional, Callable, Awaitable
import secrets
import logging
from datetime import datetime
from datadog import statsd

from server.activity_tracker import ActivityTracker
from server.dynamodb_helpers import TableContext

logger = logging.getLogger(__name__)


class InviteCodeManager:
    """Manages invite codes for whitelisting users."""

    # How many unused codes a user can have
    MAX_UNUSED_CODES = 30

    def __init__(
        self,
        get_table: Callable[[], TableContext],
        activity_tracker: ActivityTracker,
    ):
        self.get_table = get_table
        self.activity_tracker = activity_tracker

    async def generate_invite_code(self, creator_address: str) -> Optional[str]:
        """Generate a new invite code for a whitelisted user."""
        try:
            async with self.get_table() as table:
                # Check current number of unused codes
                response = await table.query(
                    IndexName="creator_address-index",
                    KeyConditionExpression="creator_address = :addr",
                    FilterExpression="#used = :used",
                    ExpressionAttributeNames={"#used": "used"},
                    ExpressionAttributeValues={
                        ":addr": creator_address,
                        ":used": False,
                    },
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
                await table.put_item(
                    Item={
                        "code": invite_code,
                        "creator_address": creator_address,
                        "created_at": int(datetime.now().timestamp()),
                        "used": False,
                        "used_by": None,
                        "used_at": None,
                    }
                )

                statsd.increment("invitecode.generated")
                return invite_code
        except Exception as e:
            logger.error(f"Error generating invite code: {e}")
            return None

    async def use_invite_code(self, code: str, user_address: str) -> bool:
        """Use an invite code to whitelist a new user."""
        try:
            async with self.get_table() as table:
                # Get the invite code
                response = await table.get_item(Key={"code": code})
                if "Item" not in response:
                    return False

                invite = response["Item"]

                # Check if code is already used
                if invite.get("used", False):
                    return False

                # Update the invite code as used
                await table.update_item(
                    Key={"code": code},
                    UpdateExpression="SET #used = :used, used_by = :used_by, used_at = :used_at",
                    ExpressionAttributeNames={"#used": "used"},
                    ExpressionAttributeValues={
                        ":used": True,
                        ":used_by": user_address,
                        ":used_at": int(datetime.now().timestamp()),
                    },
                )

                # Increment creator's successful invites count
                await self.activity_tracker.increment_successful_invites(
                    invite["creator_address"]
                )

                statsd.increment("invitecode.activated")
                return True
        except Exception as e:
            logger.error(f"Error using invite code: {e}")
            return False

    async def get_invite_stats(self, address: str) -> dict:
        """Get invite statistics for a user."""
        try:
            # Get all codes created by the user
            response = await self.table.query(
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
