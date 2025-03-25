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

    def get_token_metadata(self, token_address: str) -> Optional[TokenMetadata]:
        # Try to get from DynamoDB first
        metadata = self._get_from_dynamodb(token_address)
        if metadata:
            return metadata

        # If not in DynamoDB, fetch from DexScreener
        metadata = self.fetch_metadata_from_dexscreener(token_address)
        if metadata:
            self._store_in_dynamodb(metadata)

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

    def _store_in_dynamodb(self, metadata: TokenMetadata) -> None:
        """Store token metadata in DynamoDB."""
        item = {
            "address": metadata.address,
            "name": metadata.name,
            "symbol": metadata.symbol,
        }
        
        if metadata.image_url:
            item["image_url"] = metadata.image_url

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
