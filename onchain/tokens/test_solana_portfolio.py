import unittest
import os
import boto3
import asyncio

from onchain.portfolio.solana_portfolio import PortfolioFetcher
from onchain.tokens.metadata import TokenMetadataRepo
import dotenv

dotenv.load_dotenv()


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        dynamodb = boto3.resource(
            "dynamodb",
            region_name=os.environ.get("AWS_REGION"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

        tokens_table = dynamodb.Table("token_metadata_v2")
        self.token_metadata_repo = TokenMetadataRepo(tokens_table)
        self.portfolio = PortfolioFetcher(self.token_metadata_repo)

    async def asyncSetUp(self):
        await self.token_metadata_repo._get_session()

    async def asyncTearDown(self):
        await self.portfolio.close()

    def test_get_portfolio(self):
        # Binance wallet
        holdings = asyncio.run(self.portfolio.get_portfolio(
            "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"
        ))
        print(holdings)

        self.assertGreater(len(holdings.holdings), 0)
