import logging
from typing import Optional, Dict, Any
import httpx
from datadog import statsd
import json

from onchain.tokens.metadata import TokenMetadataRepo


class COWValidator:
    """
    A class for validating COW protocol swap orders and calculating referral rewards.
    """

    def __init__(self, token_metadata_repo: TokenMetadataRepo):
        self.mainnet_url = "https://api.cow.fi/mainnet"
        self.base_url = "https://api.cow.fi/base"
        self.referral_account = "0xd562E50DB16e978593AA8282eF1a2Ceedf5978b3"
        self.app_code = "BitQuant"

        self.referral_fee_bps = 20  # 0.20% referral fee in basis points
        self.token_metadata_repo = token_metadata_repo
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def validate_swap_order(
        self, order_uid: str, network: str
    ) -> Optional[Dict[str, Any]]:
        """
        Validate a COW protocol order by fetching its details from the COW API
        and checking if it's a valid swap order.

        Args:
            order_uid: The order UID to validate
            network: The network to check (ethereum or base)

        Returns:
            Dictionary with order details if valid, None if invalid
        """
        try:
            # Fetch order details from COW API
            order_data = await self._fetch_order(order_uid, network)
            if not order_data:
                logging.error(f"Order data not found for {order_uid} on {network}")
                statsd.increment("cow_validator.order_not_found")
                return None

            # Check if this is a valid COW order
            is_valid_order = await self._is_valid_cow_order(order_data)
            if not is_valid_order:
                logging.error(f"Order is not a valid COW order for {order_uid}")
                statsd.increment("cow_validator.order_not_valid")
                return None

            app_data = order_data.get("fullAppData")
            app_data_parsed = json.loads(app_data)
            order_metadata = app_data_parsed.get("metadata")
            fee_recipient = order_metadata.get("partnerFee", {}).get("recipient", None)
            fee_bps = float(order_metadata.get("partnerFee", {}).get("volumeBps"))
            app_code = app_data_parsed.get("appCode")

            if fee_recipient != self.referral_account:
                logging.error(
                    f"Fee recipient is not the BitQuant referral account for {order_uid}"
                )
                statsd.increment("cow_validator.fee_recipient_not_referral_account")
                return None
            if app_code != self.app_code:
                logging.error(f"App code is not the BitQuant app code for {order_uid}")
                statsd.increment("cow_validator.app_code_not_bitquant")
                return None
            if fee_bps < self.referral_fee_bps:
                logging.error(
                    f"Fee bps is less than the referral fee bps for {order_uid}"
                )
                statsd.increment("cow_validator.fee_bps_less_than_referral_fee_bps")
                return None

            # Calculate referral reward based on executed fee (more accurate)
            referral_reward_usdc = (
                await self._calculate_referral_reward_usdc_from_executed_fee(
                    order_data, network
                )
            )
            points_awarded = self.calculate_points_from_reward(referral_reward_usdc)
            logging.info(f"Referral reward in USDC: {referral_reward_usdc}")
            statsd.increment("cow_validator.success")

            return {
                "valid": True,
                "is_cow_order": True,
                "order_uid": order_uid,
                "network": network,
                "sell_token": order_data.get("sellToken"),
                "buy_token": order_data.get("buyToken"),
                "sell_amount": order_data.get("sellAmount"),
                "buy_amount": order_data.get("buyAmount"),
                "executed_sell_amount": order_data.get("executedSellAmount"),
                "creation_date": order_data.get("creationDate"),
                "owner": order_data.get("owner"),
                "fee_recipient": fee_recipient,
                "referral_reward_usdc": referral_reward_usdc,
                "points_awarded": points_awarded,
            }
        except Exception as e:
            logging.error(f"Error validating COW order {order_uid}: {e}")
            statsd.increment("cow_validator.error")
            return None

    async def _calculate_referral_reward_usdc_from_executed_fee(
        self, order_data: Dict[str, Any], network: str
    ) -> float:
        """
        Calculate USD earnings from the executed fee in the CoW order.
        This method uses the actual executed fee rather than calculating from volume.

        Args:
            order_data: The complete order data from CoW API
            network: The network (ethereum or base)

        Returns:
            The USD value of the executed fee
        """
        # Extract executed fee in wei
        executed_fee_wei = int(order_data.get("executedFee", "0"))

        if executed_fee_wei == 0:
            logging.warning("No executed fee found in order data")
            return 0.0

        # Get fee token address
        fee_token_address = order_data.get("executedFeeToken")
        if not fee_token_address:
            logging.error("No executed fee token found in order data")
            return 0.0

        # Convert fee to token units (assuming 18 decimals for most tokens)
        executed_fee_token = executed_fee_wei / (10**18)

        logging.info(f"Executed fee in token units: {executed_fee_token}")
        logging.info(f"Fee token address: {fee_token_address}")

        # Get fee token price in USD
        fee_token_metadata = await self.token_metadata_repo.get_token_metadata(
            fee_token_address, network
        )
        if not fee_token_metadata or not fee_token_metadata.price:
            logging.error(f"Fee token price not found for {fee_token_address}")
            return 0
        else:
            logging.info(f"Fee token price: {fee_token_metadata.price}")

        # Calculate USD earnings
        usd_earnings = float(executed_fee_token) * float(fee_token_metadata.price)
        logging.info(
            f"Executed fee calculation: {executed_fee_wei} wei -> {executed_fee_token} tokens -> ${usd_earnings:.6f} USD"
        )

        return usd_earnings

    def calculate_points_from_reward(self, referral_reward_usdc: float) -> int:
        """
        Less than $0.01 is 0 points, above that, the number of points is the usdc value times 100.
        """
        if referral_reward_usdc < 0.01:
            return 0

        return round(referral_reward_usdc * 100)

    async def _fetch_order(
        self, order_uid: str, network: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch order details from COW API.

        Args:
            order_uid: The order UID
            network: The network (ethereum or base)

        Returns:
            Order details if found, None otherwise
        """
        base_url = self.mainnet_url if network == "ethereum" else self.base_url

        try:
            response = await self.http_client.get(
                f"{base_url}/api/v1/orders/{order_uid}"
            )
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(
                    f"Failed to fetch order {order_uid}: {response.status_code}"
                )
                return None
        except Exception as e:
            logging.error(f"Error fetching order {order_uid}: {e}")
            return None

    async def _is_valid_cow_order(self, order_data: Dict[str, Any]) -> bool:
        """
        Check if the order data represents a valid COW order.

        Args:
            order_data: Order data from COW API

        Returns:
            True if it's a valid COW order, False otherwise
        """
        # Check for required fields
        required_fields = ["sellToken", "buyToken", "sellAmount", "buyAmount", "owner"]
        for field in required_fields:
            if field not in order_data:
                logging.error(f"Missing required field: {field}")
                return False

        # Check if order has been executed (has executedSellAmount)
        if "executedSellAmount" not in order_data:
            logging.error("Order has not been executed")
            return False

        # Check if executed amount is greater than 0
        executed_amount = int(order_data.get("executedSellAmount", "0"))
        if executed_amount <= 0:
            logging.error("Order has not been executed or executed amount is 0")
            return False

        return True

    async def close(self):
        """
        Close the HTTP client connection.
        """
        await self.http_client.aclose()
