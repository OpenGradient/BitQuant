from pydantic import BaseModel, computed_field
from typing import List, Union, Optional, Dict, Mapping, Literal
from enum import IntEnum, StrEnum


class Token(BaseModel):
    address: str
    name: str
    symbol: str


class Chain(IntEnum):
    ETHEREUM = 0
    SOLANA = 1
    BASE = 2
    OTHER = 3


class AgentType(StrEnum):
    INVESTOR = "investor_agent"
    ANALYTICS = "analytics_agent"


class WalletTokenHolding(BaseModel):
    address: str  # token address
    amount: float  # amount of tokens held
    symbol: Optional[str] = None  # token symbol
    name: Optional[str] = None  # token name
    image_url: Optional[str] = None  # token image URL
    total_value_usd: Optional[float] = None  # total value of tokens held


class Portfolio(BaseModel):
    holdings: List[WalletTokenHolding]
    total_value_usd: float


class PoolQuery(BaseModel):
    chain: Optional[Chain] = None
    tokens: List[str] = []  # tokens the user is asking about
    protocols: List[str] = []
    isStableCoin: Optional[bool] = None
    impermanentLossRisk: Optional[bool] = None
    user_tokens: List[WalletTokenHolding] = []  # user's actual token holdings


class PoolType(StrEnum):
    AMM = "AMM"
    LENDING = "Lending"
    VAULT = "Vault"


class Pool(BaseModel):
    id: str  # unique ID
    chain: Chain  # Chain pool is deployed on
    protocol: str  # protocol name
    tokens: List[Token]  # list of tokens in pool
    type: PoolType
    TVL: str  # in USD
    APRLastDay: float  # APR for last day (must be present)
    APRLastWeek: Optional[float]  # APR for last week (if known)
    APRLastMonth: Optional[float]  # APR for last month (if known)
    isStableCoin: bool  # whether pool is stablecoin
    impermanentLossRisk: bool

    @computed_field
    def risk(self) -> str:
        if not self.impermanentLossRisk:
            return "Low"
        elif self.isStableCoin and self.impermanentLossRisk:
            return "Medium"
        else:
            return "High"


class WalletPoolPosition(BaseModel):
    poolId: str  # unique ID of pool
    depositedTokens: Dict[str, float]  # address to token amount


class UserMessage(BaseModel):
    type: Literal["user"] = "user"
    message: str


class AgentMessage(BaseModel):
    type: Literal["assistant"] = "assistant"
    message: str
    pools: List[Pool]


Message = Union[UserMessage, AgentMessage]


class Context(BaseModel):
    conversationHistory: List[Message]
    address: Optional[str] # should be set
    tokens: List[WalletTokenHolding]
    poolPositions: List[WalletPoolPosition]


class AgentChatRequest(BaseModel):
    context: Context
    message: UserMessage
    agent: Optional[AgentType] = None
