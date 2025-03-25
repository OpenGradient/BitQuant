from typing import List
import logging
from solana.rpc.api import Client
from solana.rpc.types import TokenAccountOpts, Pubkey
from solders.rpc.responses import RpcKeyedAccountJsonParsed

from tokens.metadata import TokenMetadataRepo
from api.api_types import WalletTokenHolding

class PortfolioFetcher:
    # Solana mainnet RPC endpoint
    RPC_URL = "https://api.mainnet-beta.solana.com"
    TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

    def __init__(self, token_metadata_repo: TokenMetadataRepo):
        self.token_metadata_repo = token_metadata_repo
        self.http_client = Client(self.RPC_URL)

    def get_portfolio(self, wallet_address: str) -> List[WalletTokenHolding]:
        """Get the complete portfolio of token holdings for a wallet address."""
        token_accounts = self.get_token_accounts(wallet_address)

        holdings = []

        for account in token_accounts:
            print(f"Processing account: {account}")
            account_data = account.account.data.parsed["info"]

            address = account_data["mint"]
            amount = account_data["tokenAmount"]["uiAmount"]

            # Get token metadata
            metadata = self.token_metadata_repo.get_token_metadata(address)
            if metadata is None:
                continue

            # Create holding
            holding = WalletTokenHolding(
                address=address,
                amount=amount,
                symbol=metadata.symbol,
                name=metadata.name,
                image_url=metadata.image_url
            )
            holdings.append(holding)

        return holdings

    def get_token_accounts(self, wallet_address: str) -> List[RpcKeyedAccountJsonParsed]:
        """Get all token accounts owned by a wallet address."""
        # Get all token accounts owned by the wallet
        return self.http_client.get_token_accounts_by_owner_json_parsed(
            owner=Pubkey.from_string(wallet_address),
            opts=TokenAccountOpts(program_id=self.TOKEN_PROGRAM_ID)
        ).value
