from typing import Optional, Tuple

SUPPORTED_CHAINS = [
    "solana",
    "ethereum",
    "polygon",
    "bnb",
    "avax",
    "sui",
    "arbitrum",
    "base",
    "optimism",
    "celo",
    "fantom",
    "gnosis",
    "avalanche",
    "bsc",
]

# Maps internal chain names to CoinGecko API chain names
COINGECKO_CHAIN_MAP = {
    "sui": "sui-network",
    "ethereum": "eth",
    "ethereum-network": "eth",
    "polygon": "polygon_pos",
    "avalanche": "avax",
    "bnb": "bsc",
    "dogecoin": "dogechain",
}


def to_coingecko_chain(chain: str) -> str:
    """Convert an internal chain name to the CoinGecko API chain name."""
    chain = chain.lower()
    return COINGECKO_CHAIN_MAP.get(chain, chain)


def parse_token_id(token_id: str) -> Tuple[str, str]:
    """Parse a token ID string (chain:address) into (chain, address).

    Raises ValueError if the format is invalid.
    """
    if ":" not in token_id:
        raise ValueError(
            f"Invalid token ID '{token_id}': must be in the format chain:address"
        )
    chain, address = token_id.split(":", 1)
    return chain.lower(), address


def format_token_id(chain: str, address: str) -> str:
    """Format a chain and address into a token ID string."""
    return f"{chain}:{address}"


def validate_token_id(token_id: str) -> Optional[str]:
    """Validate a token ID string. Returns an error message if invalid, None if valid."""
    try:
        parse_token_id(token_id)
        return None
    except ValueError as e:
        return str(e)
