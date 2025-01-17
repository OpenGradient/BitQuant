from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Any
from enum import Enum


class Token(BaseModel):
    name: str


class Pool(BaseModel):
    # Placeholder for Bluefin API JSON object
    pass


class TokenBalance(BaseModel):
    amount: float
    asset: Token


class PoolPosition(BaseModel):
    pool: Pool
    amountDeposited: float


class ActionType(str, Enum):
    DEPOSIT = "depositToPool"
    WITHDRAW = "withdrawFromPool"
    SWAP = "swapTokens"


class DepositAction(BaseModel):
    type: ActionType = ActionType.DEPOSIT
    poolId: str
    amount: float
    asset: str


class WithdrawAction(BaseModel):
    type: ActionType = ActionType.WITHDRAW
    poolId: str
    amount: float
    asset: str


class SwapAction(BaseModel):
    type: ActionType = ActionType.SWAP
    fromToken: str
    toToken: str
    amount: float


Action = Union[DepositAction, WithdrawAction, SwapAction]


class AgentOutput(BaseModel):
    message: str
    recommendedAction: Optional[Action]


Message = Union[str, AgentOutput]  # UserInput | AgentOutput


class Context(BaseModel):
    conversationHistory: List[Message]
    tokens: List[TokenBalance]
    poolPositions: List[PoolPosition]
    availablePools: List[Pool]


class AgentRequest(BaseModel):
    context: Context
    userInput: str
