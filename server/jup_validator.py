import logging
import asyncio
from typing import Optional, Dict, Any

from solana.rpc.async_api import AsyncClient
from solders.signature import Signature
from solders.pubkey import Pubkey
from datadog import statsd

import os

from onchain.tokens.metadata import TokenMetadataRepo


class JUPValidator:
    """
    A class for validating JUP swap transactions and calculating referral rewards.
    """

    def __init__(self, token_metadata_repo: TokenMetadataRepo):
        self.base_url = "https://api.jup.ag"
        self.referral_fee_bps = 50  # 0.5% referral fee in basis points
        self.referral_account = Pubkey.from_string(
            "59LkX71yLUJpbdcKRBGQt4sxLqj1uFKW1uK4jH8GY9b4"
        )
        # Use the same RPC URL as the existing PortfolioFetcher
        self.rpc_url = os.environ.get("SOLANA_RPC_URL")
        self.http_client = AsyncClient(self.rpc_url)
        self.token_metadata_repo = token_metadata_repo

    async def validate_swap_transaction(self, txid: str) -> Optional[Dict[str, Any]]:
        """
        Validate a JUP swap transaction by fetching its details from the blockchain
        and checking if the referral account received tokens.

        Args:
            txid: The transaction ID to validate

        Returns:
            Dictionary with transaction details if valid, None if invalid
        """
        try:
            # Fetch transaction details from Solana RPC
            for _ in range(4):
                transaction_data = await self._fetch_transaction(txid)
                if not transaction_data:
                    logging.error(f"Transaction data not found for {txid}, retrying...")
                    await asyncio.sleep(1)
                else:
                    break

            if not transaction_data:
                logging.error(f"Transaction data not found for {txid} after 3 retries")
                statsd.increment("jup_validator.transaction_not_found")
                return None

            # Check if this is a JUP swap by looking for Jupiter program calls
            is_jup_swap = await self._is_jupiter_swap(transaction_data)
            if not is_jup_swap:
                logging.error(f"Transaction is not a JUP swap for {txid}")
                statsd.increment("jup_validator.transaction_not_jup_swap")
                return None

            # Check if referral account received tokens
            referral_reward_usdc = await self._check_referral_reward(transaction_data)
            statsd.increment("jup_validator.success")

            return {
                "valid": True,
                "is_jup_swap": True,
                "referral_reward_usdc": referral_reward_usdc,
            }
        except Exception as e:
            logging.error(f"Error validating swap transaction {txid}: {e}")
            return None

    def calculate_referral_reward(
        self, input_amount: int, fee_bps: int = None
    ) -> float:
        """
        Calculate the referral reward based on input amount and fee basis points.

        Args:
            input_amount: The input amount in token units
            fee_bps: Fee basis points (defaults to class default)

        Returns:
            The calculated referral reward
        """
        if fee_bps is None:
            fee_bps = self.referral_fee_bps

        # Calculate referral fee: (input_amount * fee_bps) / 10000
        referral_reward = (input_amount * fee_bps) / 10000
        return referral_reward

    def calculate_points_from_reward(self, referral_reward_usdc: float) -> int:
        """
        Less than $0.1 is 0 points, above that, the number of points is the usdc value times 100.
        """
        if referral_reward_usdc < 0.01:
            return 0

        return round(referral_reward_usdc * 100)

    async def _fetch_transaction(self, txid: str) -> Optional[Dict[str, Any]]:
        """
        Fetch transaction details from Solana RPC using the existing client.

        Args:
            txid: The transaction ID

        Returns:
            Transaction details if found, None otherwise
        """
        # Use the existing Solana RPC client
        try:
            signature = Signature.from_string(txid)
            response = await self.http_client.get_transaction(
                signature, encoding="json", max_supported_transaction_version=0
            )
            return response.value
        except Exception as error:
            logging.warning(f"_fetch_transaction error for {txid}: {error}")
            return None

    async def _is_jupiter_swap(self, transaction_data: Any) -> bool:
        """
        Check if the transaction is a Jupiter swap by examining the logs.

        Args:
            transaction_data: Transaction data from Solana RPC

        Returns:
            True if it's a Jupiter swap, False otherwise
        """
        # Check transaction logs for Jupiter program calls
        meta = transaction_data.transaction.meta
        logs = meta.log_messages

        # Look for Jupiter program ID in the logs
        jupiter_program_ids = [
            "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB",  # Jupiter V6
            "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",  # Jupiter V4
            "JUP2jxvXaqu7NQH1GziynfSqceXn9rwbp8f9cK92eZJ6",  # Jupiter V3
        ]

        for log in logs:
            for program_id in jupiter_program_ids:
                if program_id in log:
                    return True

        return False

    async def _check_referral_reward(self, transaction_data: Dict[str, Any]) -> float:
        """
        Check if the referral account received tokens from this transaction and calculate USDC value.

        Args:
            transaction_data: Transaction data from Solana RPC

        Returns:
            The USDC value of tokens received by the referral account (0 if none)
        """
        meta = transaction_data.transaction.meta

        pre_token_balances = meta.pre_token_balances
        post_token_balances = meta.post_token_balances

        # Check for token balance changes for the referral account
        total_usdc_value = 0.0

        # Create maps of pre and post token balances by mint for the referral account
        pre_balances_by_mint = {}
        post_balances_by_mint = {}

        # Check pre-token balances for the referral account
        for balance in pre_token_balances:
            owner = balance.owner
            if owner == self.referral_account:
                mint = balance.mint
                amount = float(balance.ui_token_amount.ui_amount)
                if mint and amount > 0:
                    pre_balances_by_mint[mint] = amount

        # Check post-token balances for the referral account
        for balance in post_token_balances:
            owner = balance.owner
            if owner == self.referral_account:
                mint = balance.mint
                amount = float(balance.ui_token_amount.ui_amount)
                if mint and amount > 0:
                    post_balances_by_mint[mint] = amount

        # Calculate the increase in token balances and their USDC value
        for mint, post_amount in post_balances_by_mint.items():
            pre_amount = pre_balances_by_mint.get(mint, 0)
            if post_amount > pre_amount:
                # Get token metadata to find price
                token_metadata = await self.token_metadata_repo.get_token_metadata(
                    mint, "solana"
                )
                if token_metadata and token_metadata.price:
                    # Calculate USDC value of the token increase
                    token_increase = post_amount - pre_amount
                    usdc_value = token_increase * float(token_metadata.price)
                    total_usdc_value += usdc_value
                    logging.info(
                        f"Referral account received {token_increase} {mint} worth ${usdc_value:.2f}"
                    )

        return total_usdc_value

    async def close(self):
        """
        Close the RPC client connection.
        """
        await self.http_client.close()
