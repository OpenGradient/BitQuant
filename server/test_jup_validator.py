import pytest
import os

import dotenv
import boto3
from unittest.mock import Mock

from server.jup_validator import JUPValidator
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
    """JUPValidator instance with mocked dependencies."""
    return JUPValidator(mock_token_metadata_repo)


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
    """JUPValidator instance with real dependencies for integration tests."""
    return JUPValidator(real_token_metadata_repo)


class TestJUPValidatorUnit:
    """Unit tests with mocked dependencies."""

    def test_calculate_referral_reward(self, validator):
        """Test the calculate_referral_reward method."""
        # Test with default fee (50 bps = 0.5%)
        input_amount = 1000000  # 1M units
        reward = validator.calculate_referral_reward(input_amount)
        expected_reward = (1000000 * 50) / 10000  # 5000
        assert reward == expected_reward

        # Test with custom fee
        custom_fee_bps = 100  # 1%
        reward_custom = validator.calculate_referral_reward(
            input_amount, custom_fee_bps
        )
        expected_custom = (1000000 * 100) / 10000  # 10000
        assert reward_custom == expected_custom

    def test_calculate_points_from_reward(self, validator):
        """Test the calculate_points_from_reward method."""
        # Test with positive USDC value above $0.1
        usdc_value = 5.25
        points = validator.calculate_points_from_reward(usdc_value)
        assert points == 525  # Should be int(5.25 * 100) = 525

        # Test with zero value
        points_zero = validator.calculate_points_from_reward(0.0)
        assert points_zero == 0

        # Test with exactly $0.1 (should get 10 points)
        points_exact = validator.calculate_points_from_reward(0.1)
        assert points_exact == 10

        # Test with value just above $0.1
        points_just_above = validator.calculate_points_from_reward(0.15)
        assert points_just_above == 15  # int(0.15 * 100) = 15


class TestJUPValidatorIntegration:
    """Integration tests that use real RPC calls."""

    @pytest.mark.asyncio
    async def test_real_transaction_fetch(self, real_validator: JUPValidator):
        """Test fetching a real transaction from Solana RPC."""
        txid = "4yJx666ajeuHxE2VsBjitanVVbaiS5bmmQHxU188yeH2A2vgqPvzAefELtyQwgP9o9i44qxk2baJmZYCzQbh914Y"

        print(f"\n🔍 Fetching real transaction: {txid}")

        # Test the _fetch_transaction method directly
        transaction_data = await real_validator._fetch_transaction(txid)

        assert transaction_data is not None, "Failed to fetch the real transaction"

        # Check if it's a Jupiter swap
        is_jupiter = await real_validator._is_jupiter_swap(transaction_data)
        print(f"🔍 Is Jupiter swap: {is_jupiter}")

        if is_jupiter:
            print("✅ This is a Jupiter swap transaction!")

            # Check for referral rewards
            referral_reward = await real_validator._check_referral_reward(
                transaction_data
            )
            print(f"💰 Referral reward (USDC value): ${referral_reward:.4f}")

            if referral_reward > 0:
                points = real_validator.calculate_points_from_reward(referral_reward)
                print(f"🎯 Points to award: {points}")
            else:
                print("ℹ️  No referral rewards detected")
        else:
            print("❌ This is not a Jupiter swap transaction")

        await real_validator.close()

    @pytest.mark.asyncio
    async def test_real_transaction_validation(self, real_validator: JUPValidator):
        """Test the full validation process with a real transaction."""
        txid = "3wnSpXSThjiGGqsuRr6rbMEZ6vhcAbF98cxoqgYBYUauQoTYL8Gt8sagChii75gjLznooxh2xinycrCw3o9t6LJ5"

        print(f"\n🚀 Running full validation for transaction: {txid}")

        # Run the full validation
        result = await real_validator.validate_swap_transaction(txid)

        if result:
            print("✅ Validation completed successfully!")
            print(f"Valid: {result.get('valid', False)}")
            print(f"Is Jupiter swap: {result.get('is_jup_swap', False)}")
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
            assert "is_jup_swap" in result
            assert "referral_reward_usdc" in result
        else:
            print("❌ Validation failed or returned None")
            print(
                "ℹ️  This could be normal if the transaction is not a Jupiter swap or has no referral rewards"
            )

        await real_validator.close()
