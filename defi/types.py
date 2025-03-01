from pydantic import BaseModel
from typing import List, Union, Optional, Dict, Mapping
from enum import IntEnum


class Token(BaseModel):
    address: str
    name: str
    symbol: str


class Chain(IntEnum):
    ETHEREUM = 0
    SOLANA = 1
    BASE = 2
    OTHER = 3


class PoolQuery(BaseModel):
    chain: Optional[Chain] = None
    tokens: List[str] = []
    protocols: List[str] = []
    isStableCoin: Optional[bool] = None
    impermanentLossRisk: Optional[bool] = None


class Pool(BaseModel):
    id: str  # unique ID
    chain: Chain  # Chain pool is deployed on
    protocol: str  # protocol name
    tokens: List[Token]  # list of tokens in pool
    type: str  # Lending or AMM
    TVL: str  # in USD
    APRLastDay: float  # APR for last day (must be present)
    APRLastWeek: Optional[float]  # APR for last week (if known)
    APRLastMonth: Optional[float]  # APR for last month (if known)
    isStableCoin: bool  # whether pool is stablecoin
    impermanentLossRisk: bool
    risk: str  # Risk


class WalletTokenHolding(BaseModel):
    address: str  # token address
    amount: float  # amount of tokens held


class WalletPoolPosition(BaseModel):
    poolId: str  # unique ID of pool
    depositedTokens: Dict[str, float]  # address to token amount


class AgentOutput(BaseModel):
    message: str
    pools: List[Pool]
    suggestions: List[str]


Message = Union[str, AgentOutput]


class Context(BaseModel):
    conversationHistory: List[Message]
    tokens: List[WalletTokenHolding]
    poolPositions: List[WalletPoolPosition]


class AgentChatRequest(BaseModel):
    context: Context
    userInput: str


class AgentSuggestionRequest(BaseModel):
    context: Context
