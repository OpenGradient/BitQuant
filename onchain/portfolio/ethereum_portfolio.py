"""Ethereum portfolio fetcher using web3.py for native ETH + ERC20 token balances.

Supports two modes:
- **Alchemy mode** (recommended): When ETH_RPC_URL points to an Alchemy endpoint,
  uses ``alchemy_getTokenBalances`` for complete ERC20 coverage in a single call.
- **Fallback mode**: Queries ``balanceOf`` for a curated list of well-known tokens.
"""

from typing import List, Optional, Dict, Any
import logging
import os

import aiohttp
from async_lru import alru_cache
from web3 import AsyncWeb3, AsyncHTTPProvider
from web3.contract import AsyncContract

from onchain.tokens.metadata import TokenMetadataRepo, WETH_ADDRESS
from api.api_types import WalletTokenHolding, Portfolio

logger = logging.getLogger(__name__)

# Minimal ERC20 ABI for balanceOf and decimals
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
]

# Well-known ERC20 tokens — only used as a fallback when Alchemy is not the
# RPC provider. When ETH_RPC_URL points to Alchemy, `alchemy_getTokenBalances`
# discovers all ERC20 holdings automatically, making this list unnecessary.
# address -> decimals
WELL_KNOWN_TOKENS: Dict[str, int] = {
    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48": 6,   # USDC
    "0xdAC17F958D2ee523a2206206994597C13D831ec7": 6,   # USDT
    "0x6B175474E89094C44Da98b954EedeAC495271d0F": 18,  # DAI
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": 18,  # WETH
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599": 8,   # WBTC
    "0x514910771AF9Ca656af840dff83E8264EcF986CA": 18,  # LINK
    "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984": 18,  # UNI
    "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9": 18,  # AAVE
    "0xD533a949740bb3306d119CC777fa900bA034cd52": 18,  # CRV
    "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84": 18,  # stETH
}


def _is_alchemy_url(url: str) -> bool:
    """Check if the RPC URL is an Alchemy endpoint."""
    return "alchemy" in url.lower()


class EthereumPortfolioFetcher:
    """Fetches Ethereum wallet portfolio via Alchemy enhanced API or web3.py fallback."""

    ETH_RPC_URL = os.environ.get("ETH_RPC_URL", "")

    def __init__(self, token_metadata_repo: TokenMetadataRepo):
        self.token_metadata_repo = token_metadata_repo
        self._w3: Optional[AsyncWeb3] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._use_alchemy = _is_alchemy_url(self.ETH_RPC_URL)

    @property
    def w3(self) -> AsyncWeb3:
        if self._w3 is None:
            if not self.ETH_RPC_URL:
                raise ValueError("ETH_RPC_URL environment variable is not set")
            self._w3 = AsyncWeb3(AsyncHTTPProvider(self.ETH_RPC_URL))
        return self._w3

    @property
    async def http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._http_session

    async def close(self):
        """Clean up sessions."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None
        if self._w3 and hasattr(self._w3.provider, "_session"):
            session = self._w3.provider._session
            if session and not session.closed:
                await session.close()
        self._w3 = None

    @alru_cache(maxsize=100_000, ttl=60 * 60)
    async def get_portfolio(self, wallet_address: str) -> Portfolio:
        """Get the complete portfolio of token holdings for an Ethereum wallet."""
        if not wallet_address:
            return Portfolio(holdings=[], total_value_usd=0)

        try:
            address = self.w3.to_checksum_address(wallet_address)
        except Exception:
            logger.error(f"Invalid Ethereum address: {wallet_address}")
            return Portfolio(holdings=[], total_value_usd=0)

        holdings: List[WalletTokenHolding] = []

        # Get native ETH balance
        eth_holding = await self._get_eth_holding(address)
        if eth_holding:
            holdings.append(eth_holding)

        # Get ERC20 token balances — prefer Alchemy for complete coverage
        if self._use_alchemy:
            erc20_holdings = await self._get_alchemy_token_balances(address)
        else:
            erc20_holdings = await self._get_fallback_token_balances(address)

        holdings.extend(erc20_holdings)

        portfolio_value = sum(h.total_value_usd or 0 for h in holdings)
        return Portfolio(holdings=holdings, total_value_usd=portfolio_value)

    # ── Native ETH ──────────────────────────────────────────────

    async def _get_eth_holding(
        self, wallet_address: str
    ) -> Optional[WalletTokenHolding]:
        """Get the native ETH holding for a wallet."""
        try:
            balance_wei = await self.w3.eth.get_balance(wallet_address)
            if balance_wei == 0:
                return None

            eth_amount = balance_wei / 1e18

            metadata = await self.token_metadata_repo.get_token_metadata(
                WETH_ADDRESS, "ethereum"
            )
            if metadata and metadata.price:
                value_usd = float(eth_amount) * float(metadata.price)
            else:
                value_usd = None

            return WalletTokenHolding(
                address=WETH_ADDRESS,
                amount=eth_amount,
                symbol="ETH",
                name="Ethereum",
                total_value_usd=value_usd,
            )
        except Exception as e:
            logger.error(f"Error fetching ETH balance: {e}")
            return None

    # ── Alchemy enhanced API ────────────────────────────────────

    async def _get_alchemy_token_balances(
        self, wallet_address: str
    ) -> List[WalletTokenHolding]:
        """Fetch all ERC20 balances in a single Alchemy RPC call."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "alchemy_getTokenBalances",
            "params": [wallet_address, "DEFAULT_TOKENS"],
        }

        try:
            session = await self.http_session
            async with session.post(
                self.ETH_RPC_URL, json=payload
            ) as response:
                if response.status != 200:
                    logger.warning(
                        f"Alchemy getTokenBalances returned {response.status}, "
                        f"falling back to individual queries"
                    )
                    return await self._get_fallback_token_balances(wallet_address)

                data = await response.json()
        except Exception as e:
            logger.warning(f"Alchemy API error: {e}, falling back")
            return await self._get_fallback_token_balances(wallet_address)

        result_obj = data.get("result", {})
        token_balances = result_obj.get("tokenBalances", [])

        holdings: List[WalletTokenHolding] = []
        for tb in token_balances:
            token_address = tb.get("contractAddress", "")
            hex_balance = tb.get("tokenBalance", "0x0")

            balance_raw = int(hex_balance, 16)
            if balance_raw == 0:
                continue

            # Fetch metadata (includes price) from DexScreener
            metadata = await self.token_metadata_repo.get_token_metadata(
                token_address, "ethereum"
            )
            if metadata is None:
                continue

            # Get decimals — try well-known map first, then on-chain call
            decimals = WELL_KNOWN_TOKENS.get(token_address)
            if decimals is None:
                decimals = await self._get_token_decimals(token_address)

            amount = balance_raw / (10**decimals)

            if metadata.price:
                value_usd = float(amount) * float(metadata.price)
            else:
                value_usd = None

            holdings.append(
                WalletTokenHolding(
                    address=token_address,
                    amount=amount,
                    symbol=metadata.symbol,
                    name=metadata.name,
                    image_url=metadata.image_url,
                    total_value_usd=value_usd,
                )
            )

        return holdings

    async def _get_token_decimals(self, token_address: str) -> int:
        """Query ERC20 decimals() on-chain. Defaults to 18 on failure."""
        try:
            checksum = self.w3.to_checksum_address(token_address)
            contract: AsyncContract = self.w3.eth.contract(
                address=checksum, abi=ERC20_ABI
            )
            return await contract.functions.decimals().call()
        except Exception:
            return 18

    # ── Fallback: individual balanceOf calls ────────────────────

    async def _get_fallback_token_balances(
        self, wallet_address: str
    ) -> List[WalletTokenHolding]:
        """Query balanceOf for each well-known token individually."""
        holdings: List[WalletTokenHolding] = []
        for token_address, decimals in WELL_KNOWN_TOKENS.items():
            holding = await self._get_erc20_holding(
                wallet_address, token_address, decimals
            )
            if holding:
                holdings.append(holding)
        return holdings

    async def _get_erc20_holding(
        self,
        wallet_address: str,
        token_address: str,
        decimals: int,
    ) -> Optional[WalletTokenHolding]:
        """Get a single ERC20 token holding for a wallet."""
        try:
            checksum_token = self.w3.to_checksum_address(token_address)
            contract: AsyncContract = self.w3.eth.contract(
                address=checksum_token, abi=ERC20_ABI
            )

            balance = await contract.functions.balanceOf(wallet_address).call()
            if balance == 0:
                return None

            amount = balance / (10**decimals)

            metadata = await self.token_metadata_repo.get_token_metadata(
                token_address, "ethereum"
            )
            if metadata is None:
                return None

            if metadata.price:
                value_usd = float(amount) * float(metadata.price)
            else:
                value_usd = None

            return WalletTokenHolding(
                address=token_address,
                amount=amount,
                symbol=metadata.symbol,
                name=metadata.name,
                image_url=metadata.image_url,
                total_value_usd=value_usd,
            )
        except Exception as e:
            logger.debug(f"Error fetching ERC20 balance for {token_address}: {e}")
            return None
