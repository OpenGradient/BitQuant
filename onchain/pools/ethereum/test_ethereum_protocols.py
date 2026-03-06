import unittest
import os
import boto3
import asyncio

from onchain.pools.ethereum.uniswap_v3_protocol import UniswapV3Protocol
from onchain.pools.ethereum.aave_protocol import AaveProtocol
from onchain.portfolio.ethereum_portfolio import EthereumPortfolioFetcher
from onchain.tokens.metadata import TokenMetadataRepo
from api.api_types import Chain, PoolType
import dotenv

dotenv.load_dotenv()


class TestEthereumProtocols(unittest.TestCase):
    def setUp(self):
        dynamodb = boto3.resource(
            "dynamodb",
            region_name=os.environ.get("AWS_REGION"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        tokens_table = dynamodb.Table("token_metadata_v2")
        self.token_metadata_repo = TokenMetadataRepo(tokens_table)

    def test_uniswap_v3(self):
        uniswap = UniswapV3Protocol()
        pools = asyncio.run(uniswap.get_pools(self.token_metadata_repo))

        self.assertGreater(len(pools), 5)
        print(f"Uniswap V3 pools: {len(pools)}")

        # Verify pool structure
        for pool in pools[:3]:
            self.assertEqual(pool.chain, Chain.ETHEREUM)
            self.assertEqual(pool.type, PoolType.AMM)
            self.assertIn("Uniswap V3", pool.protocol)
            self.assertEqual(len(pool.tokens), 2)
            self.assertIsNotNone(pool.TVL)
            self.assertGreaterEqual(pool.APRLastDay, 0)

    def test_uniswap_v3_stablecoin_detection(self):
        uniswap = UniswapV3Protocol()
        pools = asyncio.run(uniswap.get_pools(self.token_metadata_repo))

        # Non-stablecoin pools should have IL risk
        for pool in pools:
            if not pool.isStableCoin:
                self.assertTrue(pool.impermanentLossRisk)
            else:
                self.assertFalse(pool.impermanentLossRisk)

    def test_aave_v3(self):
        aave = AaveProtocol()
        pools = asyncio.run(aave.get_pools(self.token_metadata_repo))

        self.assertGreater(len(pools), 5)
        print(f"Aave V3 reserves: {len(pools)}")

        # Verify pool structure
        for pool in pools[:3]:
            self.assertEqual(pool.chain, Chain.ETHEREUM)
            self.assertEqual(pool.type, PoolType.LENDING)
            self.assertEqual(pool.protocol, "Aave V3")
            self.assertEqual(len(pool.tokens), 1)
            self.assertFalse(pool.impermanentLossRisk)
            self.assertGreaterEqual(pool.APRLastDay, 0)

    def test_aave_v3_tvl_is_usd(self):
        """TVL should be in USD (not raw token units)."""
        aave = AaveProtocol()
        pools = asyncio.run(aave.get_pools(self.token_metadata_repo))

        for pool in pools:
            tvl = float(pool.TVL)
            # Any active Aave reserve should have meaningful TVL in USD
            if tvl > 0:
                self.assertGreater(tvl, 100, f"{pool.tokens[0].symbol} TVL too low: {tvl}")


class TestEthereumPortfolio(unittest.TestCase):
    def setUp(self):
        dynamodb = boto3.resource(
            "dynamodb",
            region_name=os.environ.get("AWS_REGION"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        tokens_table = dynamodb.Table("token_metadata_v2")
        self.token_metadata_repo = TokenMetadataRepo(tokens_table)
        self.portfolio_fetcher = EthereumPortfolioFetcher(self.token_metadata_repo)

    def tearDown(self):
        asyncio.run(self.portfolio_fetcher.close())

    def test_get_portfolio(self):
        # Vitalik's public wallet
        portfolio = asyncio.run(
            self.portfolio_fetcher.get_portfolio(
                "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
            )
        )
        print(portfolio)

        self.assertGreater(len(portfolio.holdings), 0)
        # Vitalik holds ETH
        eth_holdings = [h for h in portfolio.holdings if h.symbol == "ETH"]
        self.assertEqual(len(eth_holdings), 1)
        self.assertGreater(eth_holdings[0].amount, 0)

    def test_empty_wallet(self):
        # Zero address should have no holdings
        portfolio = asyncio.run(
            self.portfolio_fetcher.get_portfolio(
                "0x0000000000000000000000000000000000000001"
            )
        )
        self.assertEqual(len(portfolio.holdings), 0)

    def test_invalid_address(self):
        portfolio = asyncio.run(
            self.portfolio_fetcher.get_portfolio("not-a-valid-address")
        )
        self.assertEqual(len(portfolio.holdings), 0)
        self.assertEqual(portfolio.total_value_usd, 0)
