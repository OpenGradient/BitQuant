import unittest
import json
import boto3
import os

from onchain.pools.solana.orca_protocol import OrcaProtocol
from onchain.pools.solana.save_protocol import SaveProtocol
from onchain.pools.solana.kamino_protocol import KaminoProtocol
from api.api_types import Chain, Pool, PoolQuery
from onchain.tokens.metadata import TokenMetadataRepo


class TestProtocols(unittest.TestCase):
    def setUp(self):
        dynamodb = boto3.resource(
            "dynamodb",
            region_name=os.environ.get("AWS_REGION"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        tokens_table = dynamodb.Table("token_metadata_v2")
        self.token_metadata_repo = TokenMetadataRepo(tokens_table)

    def test_save(self):
        save = SaveProtocol()
        pools = save.get_pools(self.token_metadata_repo)

        self.assertGreater(len(pools), 5)
        print(f"Save protocols: {len(pools)}")

    def test_orca(self):
        orca = OrcaProtocol()
        pools = orca.get_pools(self.token_metadata_repo)

        self.assertGreater(len(pools), 5)
        print(f"Orca protocols: {len(pools)}")

    def test_kamino(self):
        kamino = KaminoProtocol()
        pools = kamino.get_pools(self.token_metadata_repo)

        self.assertGreater(len(pools), 5)
        print(f"Kamino protocols: {len(pools)}")
