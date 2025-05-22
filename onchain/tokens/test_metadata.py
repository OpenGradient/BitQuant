import unittest
import os
import boto3
import dotenv
import asyncio

from onchain.tokens.metadata import TokenMetadataRepo

dotenv.load_dotenv()


class TestMetadata(unittest.TestCase):
    def setUp(self):
        self.dynamodb = boto3.resource(
            "dynamodb",
            region_name=os.environ.get("AWS_REGION"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        self.tokens_table = self.dynamodb.Table("token_metadata_v2")
        self.repo = TokenMetadataRepo(self.tokens_table)

    def test_search_token(self):
        tokens = asyncio.run(self.repo.search_token("fartcoin", None))
        print(f"Search results: {tokens}")

    def test_get_token_metadata(self):
        metadata = asyncio.run(
            self.repo.get_token_metadata(
                "9Rhbn9G5poLvgnFzuYBtJgbzmiipNra35QpnUek9virt", "solana"
            )
        )
        print(f"Token metadata: {metadata}")

    def test_search_token_on_dexscreener(self):
        tokens = asyncio.run(self.repo.search_token("Fartcoin", "Solana"))

        self.assertIsNotNone(tokens)
        print(f"Search results: {tokens}")
