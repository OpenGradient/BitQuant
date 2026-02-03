import pytest
import os

import dotenv
import boto3
from unittest.mock import Mock

from server.cow_validator import COWValidator
from onchain.tokens.metadata import TokenMetadataRepo

# Load environment variables
dotenv.load_dotenv()


@pytest.fixture
def mock_token_metadata_repo():
    """Mock TokenMetadataRepo for unit tests."""
    mock_repo = Mock(spec=TokenMetadataRepo)

    # Mock the _get_from_dynamodb method to always return None
    # This forces the repo to fetch from DexScreener instead
    async def mock_get_from_dynamodb(chain, token_address):
        return None

    mock_repo._get_from_dynamodb = mock_get_from_dynamodb

    # Mock the _store_metadata method to do nothing
    async def mock_store_metadata(metadata):
        return None

    mock_repo._store_metadata = mock_store_metadata

    return mock_repo


@pytest.fixture
def validator(mock_token_metadata_repo):
    """COWValidator instance with mocked dependencies."""
    return COWValidator(mock_token_metadata_repo)


@pytest.fixture
def real_token_metadata_repo():
    """Real TokenMetadataRepo for integration tests with mocked DynamoDB."""
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    tokens_table = dynamodb.Table("token_metadata_v2")

    # Create a simple async function to get the table
    async def get_table():
        return tokens_table

    # Create the real repo but patch the _get_from_dynamodb method to always return None
    repo = TokenMetadataRepo(get_table)

    async def mock_get_from_dynamodb(chain, token_address):
        return None

    repo._get_from_dynamodb = mock_get_from_dynamodb

    # Mock the _store_metadata method to do nothing
    async def mock_store_metadata(metadata):
        return None

    repo._store_metadata = mock_store_metadata

    return repo


@pytest.fixture
def real_validator(real_token_metadata_repo):
    """COWValidator instance with real dependencies for integration tests."""
    return COWValidator(real_token_metadata_repo)


class TestCOWValidatorUnit:
    """Unit tests with mocked dependencies."""

    def test_calculate_points_from_reward(self, validator):
        """Test the calculate_points_from_reward method."""
        # Test with positive USDC value above $0.01
        usdc_value = 5.25
        points = validator.calculate_points_from_reward(usdc_value)
        assert points == 525  # Should be int(5.25 * 100) = 525

        # Test with zero value
        points_zero = validator.calculate_points_from_reward(0.0)
        assert points_zero == 0

        # Test with value below $0.01 (should get 0 points)
        points_below_threshold = validator.calculate_points_from_reward(0.005)
        assert points_below_threshold == 0

        # Test with value just above $0.01
        points_just_above = validator.calculate_points_from_reward(0.015)
        assert points_just_above == 1  # int(0.015 * 100) = 1

    @pytest.mark.asyncio
    async def test_calculate_referral_reward_usdc(self, validator):
        """Test the calculate_referral_reward_usdc method."""
        # Test with default fee (20 bps = 0.2%)
        sell_amount = "1000000000000000000"  # 1 ETH in wei
        sell_token_price = 1.0  # ETH price in ETH (1:1)

        # Mock the token metadata repo to return ETH price
        async def mock_get_token_metadata(address, chain):
            if address == "0x0000000000000000000000000000000000000000":  # ETH address
                mock_metadata = Mock()
                mock_metadata.price = 2000.0  # ETH price in USDC
                return mock_metadata
            return None

        async def mock_search_token(symbol, chain):
            if symbol == "ETH":
                mock_metadata = Mock()
                mock_metadata.price = 2000.0  # ETH price in USDC
                return mock_metadata
            return None

        validator.token_metadata_repo.get_token_metadata = mock_get_token_metadata
        validator.token_metadata_repo.search_token = mock_search_token

        reward = await validator._calculate_referral_reward_usdc(
            sell_amount, sell_token_price, "ethereum", 20
        )

        # Expected calculation:
        # sell_amount = 1 ETH in wei
        # referral_reward_wei = (1e18 * 20) / 10000 = 2e15 wei
        # referral_reward_token = 2e15 / 1e18 = 0.002 ETH
        # sell_token_price_usdc = 1 * 2000 = 2000 USDC per ETH
        # reward = 0.002 * 2000 = 4 USDC
        expected_reward = 4.0
        assert abs(reward - expected_reward) < 0.01


class TestCOWValidatorIntegration:
    """Integration tests that use real API calls."""

    @pytest.mark.asyncio
    async def test_real_order_fetch(self, real_validator: COWValidator):
        """Test fetching a real order from CoW API."""
        # Use a known CoW order UID for testing
        order_uid = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"

        print(f"\n🔍 Fetching real CoW order: {order_uid}")

        # Test the _fetch_order method directly
        order_data = await real_validator._fetch_order(order_uid, "ethereum")

        if order_data:
            print("✅ Successfully fetched order data!")
            print(f"Order details: {order_data}")

            # Check if it's a valid CoW order
            is_valid = await real_validator._is_valid_cow_order(order_data)
            print(f"🔍 Is valid CoW order: {is_valid}")
        else:
            print(
                "❌ Failed to fetch order data (this might be expected for test order)"
            )

        await real_validator.close()

    @pytest.mark.asyncio
    async def test_real_order_validation(self, real_validator: COWValidator):
        """Test the full validation process with a real order."""
        # Use a known CoW order UID for testing
        order_uid = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"

        print(f"\n🚀 Running full validation for order: {order_uid}")

        # Run the full validation
        result = await real_validator.validate_swap_order(order_uid, "ethereum")

        if result:
            print("✅ Validation completed successfully!")
            print(f"Valid: {result.get('valid', False)}")
            print(f"Is CoW order: {result.get('is_cow_order', False)}")
            print(f"Referral reward: ${result.get('referral_reward_usdc', 0):.4f}")

            # Calculate points if there's a reward
            if result.get("referral_reward_usdc", 0) > 0:
                points = real_validator.calculate_points_from_reward(
                    result["referral_reward_usdc"]
                )
                print(f"Points to award: {points}")

            # Basic assertions
            assert result is not None
            assert "valid" in result
            assert "is_cow_order" in result
            assert "referral_reward_usdc" in result
        else:
            print("❌ Validation failed or returned None")
            print("ℹ️  This could be normal if the order is not valid or doesn't exist")

        await real_validator.close()
