from pydantic import BaseModel
from typing import List, Union, Optional, Dict
from enum import Enum


class Token(BaseModel):
    symbol: str
    price: float


class Pool(BaseModel):
    id: str  # unique ID
    tokens: List[Token]  # list of tokens in pool
    TVL: str  # in USD
    APRLastDay: float  # APR for last day (must be present)
    APRLastWeek: Optional[float]  # APR for last week (if known)
    APRLastMonth: Optional[float]  # APR for last month (if known)
    protocol: str  # protocol name


class WalletTokenHolding(BaseModel):
    tokenSymbol: str  # token symbol
    amount: float  # amount of tokens held


class WalletPoolPosition(BaseModel):
    poolId: str  # unique ID of pool
    depositedTokens: Dict[str, float]  # deposited Tokens to pool


class ActionType(str, Enum):
    DEPOSIT = "depositToPool"
    WITHDRAW = "withdrawFromPool"


class DepositAction(BaseModel):
    type: ActionType = ActionType.DEPOSIT
    pool: str
    tokens: Dict[str, float]


class WithdrawAction(BaseModel):
    type: ActionType = ActionType.WITHDRAW
    pool: str
    tokens: Dict[str, float]


Action = Union[DepositAction, WithdrawAction]


class AgentOutput(BaseModel):
    message: str
    recommendedActions: List[Action]


Message = Union[str, AgentOutput]


class Context(BaseModel):
    conversationHistory: List[Message]
    tokens: List[WalletTokenHolding]
    poolPositions: List[WalletPoolPosition]
    availablePools: List[Pool]


class AgentChatRequest(BaseModel):
    context: Context
    userInput: str


class AgentSuggestionRequest(BaseModel):
    context: Context
