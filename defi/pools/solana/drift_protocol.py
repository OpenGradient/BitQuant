from typing import List, Optional, Dict, Any
from enum import IntEnum

from pydantic import BaseModel
import requests

from api.api_types import Pool, Token, Chain
from defi.pools.protocol import Protocol


class DriftProtocol(Protocol):

    PROTOCOL_NAME = "orca"

    def __init__(self, chain_id: str = "solana"):
        """
        Initialize the DriftProtocol client

        Args:
            chain_id: The chain ID to use (default: "solana")
        """
        self.chain_id = chain_id

    @property
    def name(self) -> str:
        return self.PROTOCOL_NAME

    def get_pools(self) -> List[Pool]:
        return []
