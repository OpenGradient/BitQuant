from typing import List, Optional
import os

from async_lru import alru_cache
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TokenAccountOpts, Pubkey
from solders.rpc.responses import RpcKeyedAccountJsonParsed

from onchain.tokens.metadata import TokenMetadataRepo
from api.api_types import WalletTokenHolding, Portfolio


class PortfolioFetcher:
    # Solana mainnet RPC endpoint
    RPC_URL = os.environ.get("SOLANA_RPC_URL")
    TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
    SOL_MINT = "So11111111111111111111111111111111111111112"

    def __init__(self, token_metadata_repo: TokenMetadataRepo):
        self.token_metadata_repo = token_metadata_repo
        self.http_client = AsyncClient(self.RPC_URL)

    async def close(self):
        await self.http_client.close()

    @alru_cache(maxsize=1_000_000, ttl=60 * 60)
    async def get_portfolio(self, wallet_address: str) -> Portfolio:
        if wallet_address == "":
            return Portfolio(holdings=[], total_value_usd=0)

        """Get the complete portfolio of token holdings for a wallet address."""
        token_accounts = await self._get_token_accounts(wallet_address)
        holdings: List[WalletTokenHolding] = []

        # Get native SOL holding if any
        sol_holding = await self._get_sol_holding(wallet_address)
        if sol_holding:
            holdings.append(sol_holding)

        # Process token accounts
        for account in token_accounts:
            account_data = account.account.data.parsed["info"]

            address = account_data["mint"]
            amount = account_data["tokenAmount"]["uiAmount"]

            # Ignore zero-balance token accounts
            if amount == 0:
                continue

            # Get token metadata
            metadata = await self.token_metadata_repo.get_token_metadata(
                address, "solana"
            )
            if metadata is None:
                continue

            if metadata.price:
                total_value_usd = float(amount) * float(metadata.price)
            else:
                total_value_usd = None

            # Create holding
            holding = WalletTokenHolding(
                address=address,
                amount=amount,
                symbol=metadata.symbol,
                name=metadata.name,
                image_url=metadata.image_url,
                total_value_usd=total_value_usd,
            )
            holdings.append(holding)

        portfolio_value = sum(holding.total_value_usd or 0 for holding in holdings)
        return Portfolio(holdings=holdings, total_value_usd=portfolio_value)

    async def _get_sol_holding(
        self, wallet_address: str
    ) -> Optional[WalletTokenHolding]:
        """Get the native SOL holding for a wallet address if any exists."""
        sol_balance = (
            await self.http_client.get_balance(Pubkey.from_string(wallet_address))
        ).value
        if sol_balance == 0:
            return None

        sol_amount = sol_balance / 1e9  # Convert lamports to SOL
        sol_metadata = await self.token_metadata_repo.get_token_metadata(
            PortfolioFetcher.SOL_MINT, "solana"
        )
        if sol_metadata and sol_metadata.price:
            sol_value_usd = float(sol_amount) * float(sol_metadata.price)
        else:
            sol_value_usd = None

        return WalletTokenHolding(
            address=PortfolioFetcher.SOL_MINT,
            amount=sol_amount,
            symbol="SOL",
            name="Solana",
            total_value_usd=sol_value_usd,
        )

    async def _get_token_accounts(
        self, wallet_address: str
    ) -> List[RpcKeyedAccountJsonParsed]:
        """Get all token accounts owned by a wallet address."""
        # Get all token accounts owned by the wallet
        response = await self.http_client.get_token_accounts_by_owner_json_parsed(
            owner=Pubkey.from_string(wallet_address),
            opts=TokenAccountOpts(program_id=self.TOKEN_PROGRAM_ID),
        )
        return response.value
