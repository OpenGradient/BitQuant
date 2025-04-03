import unittest
import os
import boto3
import dotenv

from onchain.tokens.metadata import TokenMetadataRepo

dotenv.load_dotenv()


class TestMetadata(unittest.TestCase):
    def test_get_token_metadata(self):
        dynamodb = boto3.resource(
            "dynamodb",
            region_name=os.environ.get("AWS_REGION"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        tokens_table = dynamodb.Table("sol_token_metadata")

        repo = TokenMetadataRepo(tokens_table)
        metadata = repo.get_token_metadata(
            "9Rhbn9G5poLvgnFzuYBtJgbzmiipNra35QpnUek9virt"
        )

        print(metadata)
