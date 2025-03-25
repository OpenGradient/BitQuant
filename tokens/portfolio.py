from typing import List
import os

from solana.rpc.api import Client
from solana.rpc.types import TokenAccountOpts, Pubkey
from solders.rpc.responses import RpcKeyedAccountJsonParsed

from tokens.metadata import TokenMetadataRepo
from api.api_types import WalletTokenHolding, Portfolio


class PortfolioFetcher:
    # Solana mainnet RPC endpoint
    RPC_URL = os.environ.get("SOLANA_RPC_URL")
    TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

    def __init__(self, token_metadata_repo: TokenMetadataRepo):
        self.token_metadata_repo = token_metadata_repo
        self.http_client = Client(self.RPC_URL)

    def get_portfolio(self, wallet_address: str) -> Portfolio:
        """Get the complete portfolio of token holdings for a wallet address."""
        token_accounts = self.get_token_accounts(wallet_address)

        holdings: List[WalletTokenHolding] = []

        for account in token_accounts:
            account_data = account.account.data.parsed["info"]

            address = account_data["mint"]
            amount = account_data["tokenAmount"]["uiAmount"]

            # Get token metadata
            metadata = self.token_metadata_repo.get_token_metadata(address)
            if metadata is None:
                continue

            if metadata.price:
                total_value_usd = (float(amount) * float(metadata.price),)
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

    def get_token_accounts(
        self, wallet_address: str
    ) -> List[RpcKeyedAccountJsonParsed]:
        """Get all token accounts owned by a wallet address."""
        # Get all token accounts owned by the wallet
        return self.http_client.get_token_accounts_by_owner_json_parsed(
            owner=Pubkey.from_string(wallet_address),
            opts=TokenAccountOpts(program_id=self.TOKEN_PROGRAM_ID),
        ).value
