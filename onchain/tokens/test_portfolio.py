import unittest
import os
import boto3

from onchain.tokens.portfolio import PortfolioFetcher
from onchain.tokens.metadata import TokenMetadataRepo
import dotenv

dotenv.load_dotenv()


class TestPortfolio(unittest.TestCase):
    def test_get_portfolio(self):
        dynamodb = boto3.resource(
            "dynamodb",
            region_name=os.environ.get("AWS_REGION"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

        tokens_table = dynamodb.Table("token_metadata_v2")
        token_metadata_repo = TokenMetadataRepo(tokens_table)
        portfolio = PortfolioFetcher(token_metadata_repo)

        # Binance wallet
        holdings = portfolio.get_portfolio(
            "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"
        )
        print(holdings)

        self.assertGreater(len(holdings), 0)
