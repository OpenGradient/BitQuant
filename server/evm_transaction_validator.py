import logging
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import httpx
from datadog import statsd

from onchain.tokens.metadata import TokenMetadataRepo

# ERC20 Transfer event topic: keccak256("Transfer(address,address,uint256)")
ERC20_TRANSFER_TOPIC = (
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
)

# Native token addresses used by CoinGecko/DexScreener for price lookups
NATIVE_TOKEN_ADDRESS = {
    "ethereum": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
    "base": "0x4200000000000000000000000000000000000006",  # WETH on Base
    "arbitrum": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",  # WETH on Arbitrum
    "optimism": "0x4200000000000000000000000000000000000006",  # WETH on Optimism
    "polygon": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",  # WMATIC
}

DEFAULT_RPC_URLS = {
    "ethereum": "https://eth.llamarpc.com",
    "base": "https://base.llamarpc.com",
    "arbitrum": "https://arbitrum.llamarpc.com",
    "optimism": "https://optimism.llamarpc.com",
    "polygon": "https://polygon.llamarpc.com",
}


@dataclass
class EvmTransfer:
    token_address: Optional[str]  # None for native ETH
    from_address: str
    to_address: str
    amount_wei: int
    decimals: int


class EvmTransactionValidator:
    """
    Validates EVM transactions and calculates rewards based on the total value
    of assets sent (native ETH/token + ERC20 transfers).
    """

    def __init__(self, token_metadata_repo: TokenMetadataRepo):
        self.token_metadata_repo = token_metadata_repo
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.rpc_urls = {}
        for chain, default_url in DEFAULT_RPC_URLS.items():
            env_key = f"{chain.upper()}_RPC_URL"
            self.rpc_urls[chain] = os.environ.get(env_key, default_url)

    async def validate_transaction(
        self, txid: str, chain: str
    ) -> Optional[Dict[str, Any]]:
        """
        Validate an EVM transaction and calculate the total USD value of assets sent.

        Args:
            txid: The transaction hash
            chain: The chain name (ethereum, base, arbitrum, optimism, polygon)

        Returns:
            Dictionary with transaction details and reward info, or None if invalid
        """
        if chain not in self.rpc_urls:
            logging.error(f"Unsupported chain: {chain}")
            return None

        try:
            # Fetch transaction and receipt in parallel
            tx_data = await self._fetch_transaction(txid, chain)
            if not tx_data:
                logging.error(f"Transaction not found: {txid} on {chain}")
                statsd.increment("evm_validator.transaction_not_found")
                return None

            receipt = await self._fetch_receipt(txid, chain)
            if not receipt:
                logging.error(f"Receipt not found: {txid} on {chain}")
                statsd.increment("evm_validator.receipt_not_found")
                return None

            # Check transaction was successful
            status = int(receipt.get("status", "0x0"), 16)
            if status != 1:
                logging.error(f"Transaction failed: {txid}")
                statsd.increment("evm_validator.transaction_failed")
                return None

            sender = tx_data["from"].lower()

            # Calculate total USD value of assets sent
            total_usd_value = 0.0
            transfers: List[Dict[str, Any]] = []

            # 1. Native ETH/token value
            native_value_wei = int(tx_data.get("value", "0x0"), 16)
            if native_value_wei > 0:
                native_usd = await self._get_native_token_usd_value(
                    native_value_wei, chain
                )
                total_usd_value += native_usd
                transfers.append(
                    {
                        "type": "native",
                        "amount_wei": native_value_wei,
                        "usd_value": native_usd,
                    }
                )

            # 2. ERC20 transfers from the sender
            erc20_transfers = self._parse_erc20_transfers(receipt, sender)
            for transfer in erc20_transfers:
                usd_value = await self._get_erc20_usd_value(
                    transfer.token_address,
                    transfer.amount_wei,
                    transfer.decimals,
                    chain,
                )
                total_usd_value += usd_value
                transfers.append(
                    {
                        "type": "erc20",
                        "token_address": transfer.token_address,
                        "amount_wei": transfer.amount_wei,
                        "decimals": transfer.decimals,
                        "usd_value": usd_value,
                    }
                )

            # Calculate points (same formula as other validators)
            referral_reward_usdc = total_usd_value
            points_awarded = self._calculate_points_from_reward(referral_reward_usdc)

            logging.info(
                f"EVM transaction {txid} on {chain}: "
                f"total value ${total_usd_value:.2f}, "
                f"points {points_awarded}"
            )
            statsd.increment("evm_validator.success")

            return {
                "valid": True,
                "txid": txid,
                "chain": chain,
                "sender": sender,
                "transfers": transfers,
                "total_usd_value": total_usd_value,
                "referral_reward_usdc": referral_reward_usdc,
                "points_awarded": points_awarded,
            }
        except Exception as e:
            logging.error(f"Error validating EVM transaction {txid}: {e}")
            statsd.increment("evm_validator.error")
            return None

    def _calculate_points_from_reward(self, referral_reward_usdc: float) -> int:
        if referral_reward_usdc < 0.01:
            return 0
        return round(referral_reward_usdc * 100)

    async def _get_native_token_usd_value(self, amount_wei: int, chain: str) -> float:
        """Convert native token amount (wei) to USD value."""
        amount = amount_wei / (10**18)
        wrapped_address = NATIVE_TOKEN_ADDRESS.get(chain)
        if not wrapped_address:
            logging.warning(f"No native token address mapping for chain {chain}")
            return 0.0

        metadata = await self.token_metadata_repo.get_token_metadata(
            wrapped_address, chain
        )
        if not metadata or not metadata.price:
            logging.warning(f"Could not get native token price for {chain}")
            return 0.0

        return amount * float(metadata.price)

    def _parse_erc20_transfers(
        self, receipt: Dict[str, Any], sender: str
    ) -> List[EvmTransfer]:
        """Parse ERC20 Transfer events from transaction receipt logs where sender is the from address."""
        transfers = []
        logs = receipt.get("logs", [])

        for log in logs:
            topics = log.get("topics", [])
            if len(topics) < 3:
                continue

            # Check if this is a Transfer event
            if topics[0] != ERC20_TRANSFER_TOPIC:
                continue

            # Decode from and to addresses from indexed topics
            from_address = "0x" + topics[1][-40:]
            to_address = "0x" + topics[2][-40:]

            # Only count transfers FROM the sender
            if from_address.lower() != sender:
                continue

            # Decode amount from data
            data = log.get("data", "0x0")
            amount_wei = int(data, 16) if data != "0x" else 0

            if amount_wei > 0:
                token_address = log.get("address", "").lower()
                transfers.append(
                    EvmTransfer(
                        token_address=token_address,
                        from_address=from_address,
                        to_address=to_address,
                        amount_wei=amount_wei,
                        decimals=18,  # Will be refined per token
                    )
                )

        return transfers

    async def _get_erc20_usd_value(
        self,
        token_address: str,
        amount_wei: int,
        decimals: int,
        chain: str,
    ) -> float:
        """Get USD value of an ERC20 token amount."""
        # Try to get decimals from the token metadata or default to 18
        metadata = await self.token_metadata_repo.get_token_metadata(
            token_address, chain
        )
        if not metadata or not metadata.price:
            logging.warning(
                f"Could not get price for ERC20 token {token_address} on {chain}"
            )
            return 0.0

        # Try to get actual decimals via RPC
        actual_decimals = await self._get_token_decimals(token_address, chain)
        if actual_decimals is not None:
            decimals = actual_decimals

        amount = amount_wei / (10**decimals)
        return amount * float(metadata.price)

    async def _get_token_decimals(
        self, token_address: str, chain: str
    ) -> Optional[int]:
        """Fetch ERC20 token decimals via RPC call."""
        rpc_url = self.rpc_urls.get(chain)
        if not rpc_url:
            return None

        try:
            # decimals() function selector: 0x313ce567
            response = await self.http_client.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_call",
                    "params": [
                        {"to": token_address, "data": "0x313ce567"},
                        "latest",
                    ],
                },
            )
            result = response.json().get("result")
            if result and result != "0x":
                return int(result, 16)
        except Exception as e:
            logging.warning(f"Failed to get decimals for {token_address}: {e}")

        return None

    async def _fetch_transaction(
        self, txid: str, chain: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch transaction data via JSON-RPC."""
        return await self._rpc_call(chain, "eth_getTransactionByHash", [txid])

    async def _fetch_receipt(self, txid: str, chain: str) -> Optional[Dict[str, Any]]:
        """Fetch transaction receipt via JSON-RPC."""
        return await self._rpc_call(chain, "eth_getTransactionReceipt", [txid])

    async def _rpc_call(
        self, chain: str, method: str, params: list
    ) -> Optional[Dict[str, Any]]:
        """Make a JSON-RPC call to the chain's RPC endpoint."""
        rpc_url = self.rpc_urls.get(chain)
        if not rpc_url:
            return None

        try:
            response = await self.http_client.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": params,
                },
            )
            data = response.json()
            return data.get("result")
        except Exception as e:
            logging.error(f"RPC call failed for {chain}/{method}: {e}")
            return None

    async def close(self):
        await self.http_client.aclose()
