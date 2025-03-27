from typing import List, Set
import logging
from boto3.resources.base import ServiceResource

logger = logging.getLogger(__name__)


class TwoLigmaWhitelist:
    """Repository for managing wallet access permissions."""

    def __init__(self, table: ServiceResource):
        self.table = table
        self._allowed: Set[str] = set()
        self._load_whitelist()

    def _load_whitelist(self):
        """Load whitelist from DynamoDB into memory."""
        try:
            response = self.table.scan()
            self._allowed = {item["wallet"] for item in response.get("Items", [])}
        except Exception as e:
            logger.error(f"Error loading whitelist from DynamoDB: {e}")
            self._allowed = set()

    def is_allowed(self, address: str) -> bool:
        """Check if a wallet address is allowed to access the service."""
        return address in self._allowed

    def get_allowed(self) -> List[str]:
        """Get all wallet addresses that have access to the service."""
        return list(self._allowed)

    def add(self, address: str) -> bool:
        """Add a wallet address to the allowed list."""
        try:
            self.table.put_item(Item={"wallet": address})
            self._allowed.add(address)
            return True
        except Exception as e:
            logger.error(f"Error adding address to whitelist: {e}")
            return False

    def remove(self, address: str) -> bool:
        """Remove a wallet address from the allowed list."""
        try:
            self.table.delete_item(Key={"wallet": address})
            self._allowed.discard(address)
            return True
        except Exception as e:
            logger.error(f"Error removing address from whitelist: {e}")
            return False
