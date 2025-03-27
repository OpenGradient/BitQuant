from typing import List, Set
import logging
from cachetools import TTLCache
from boto3.resources.base import ServiceResource

logger = logging.getLogger(__name__)


class TwoLigmaWhitelist:
    """Repository for managing wallet access permissions."""

    def __init__(self, table: ServiceResource):
        self.table = table
        self._allowed: Set[str] = set()  # Permanent cache for allowed addresses
        self._not_allowed = TTLCache(
            maxsize=10000, ttl=300
        )  # 5 min TTL for not allowed addresses

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
        if address in self._allowed:
            return True

        # Check if we've recently verified this address is not allowed
        if address in self._not_allowed:
            return False

        # If not in either cache, check DynamoDB
        try:
            response = self.table.get_item(Key={"wallet": address})
            is_allowed = "Item" in response
            if not is_allowed:
                self._not_allowed[address] = True
            else:
                self._allowed.add(address)

            return is_allowed
        except Exception as e:
            logger.error(f"Error checking address in DynamoDB: {e}")
            return False

    def get_allowed(self) -> List[str]:
        """Get all wallet addresses that have access to the service."""
        return list(self._allowed)

    def add(self, address: str) -> bool:
        """Add a wallet address to the allowed list."""
        try:
            self.table.put_item(Item={"wallet": address})
            self._allowed.add(address)
            if address in self._not_allowed:
                del self._not_allowed[address]
            return True
        except Exception as e:
            logger.error(f"Error adding address to whitelist: {e}")
            return False
