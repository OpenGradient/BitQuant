from dataclasses import dataclass
import botocore.exceptions
import requests
from typing import Optional
import logging
import botocore


@dataclass
class TokenMetadata:
    address: str
    name: str
    symbol: str
    image_url: Optional[str]


class TokenMetadataRepo:
    DEXSCREENER_API_URL = "https://api.dexscreener.com/tokens/v1/solana/%s"

    def __init__(self, tokens_table):
        self._tokens_table = tokens_table
        self._not_found_cache = set()

    def get_token_metadata(self, token_address: str) -> Optional[TokenMetadata]:
        # Check local not found cache first
        if token_address in self._not_found_cache:
            return None

        # Try to get from DynamoDB first
        metadata = self._get_from_dynamodb(token_address)
        if metadata is not None:  # Explicitly check for None since metadata could be False
            return metadata

        # If not in DynamoDB, fetch from DexScreener
        metadata = self.fetch_metadata_from_dexscreener(token_address)
        if metadata:
            self._store_metadata(metadata)
        else:
            self._store_not_found(token_address)
            self._not_found_cache.add(token_address)

        return metadata

    def _get_from_dynamodb(self, token_address: str) -> Optional[TokenMetadata]:
        """Retrieve token metadata from DynamoDB."""
        try:
            response = self._tokens_table.get_item(
                Key={"address": token_address}
            )
            
            if "Item" not in response:
                return None

            item = response["Item"]
            
            # Check if this is a "not found" marker
            if item.get("not_found", False):
                self._not_found_cache.add(token_address)
                return None

            return TokenMetadata(
                address=item["address"],
                name=item["name"],
                symbol=item["symbol"],
                image_url=item.get("image_url"),
            )
        except botocore.exceptions.ClientError as error:
            if error.response['Error']['Code'] == 'ResourceNotFoundException':
                return None
            logging.error(f"Error retrieving token metadata from DynamoDB: {error}")
            raise error

    def _store_metadata(self, metadata: TokenMetadata) -> None:
        """Store token metadata in DynamoDB."""
        item = {
            "address": metadata.address,
            "name": metadata.name,
            "symbol": metadata.symbol,
            "not_found": False
        }
        
        if metadata.image_url:
            item["image_url"] = metadata.image_url

        self._tokens_table.put_item(Item=item)

    def _store_not_found(self, token_address: str) -> None:
        """Store a marker indicating that token metadata was not found."""
        item = {
            "address": token_address,
            "not_found": True
        }
        self._tokens_table.put_item(Item=item)

    def fetch_metadata_from_dexscreener(self, token_address: str) -> Optional[TokenMetadata]:
        """Fetch token metadata from DexScreener API."""
        try:
            response = requests.get(self.DEXSCREENER_API_URL % token_address)
            if response.status_code != 200:
                logging.error(f"Failed to fetch metadata from dexscreener: {response.status_code} {response.text}")
                return None

            metadata = response.json()
            if len(metadata) == 0:
                return None

            metadata = metadata[0]
            return TokenMetadata(
                address=metadata["baseToken"]["address"],
                name=metadata["baseToken"]["name"],
                symbol=metadata["baseToken"]["symbol"],
                image_url=metadata["info"]["imageUrl"] if "info" in metadata else None,
            )
        except Exception as e:
            logging.error(f"Error fetching metadata from DexScreener: {e}")
            return None
