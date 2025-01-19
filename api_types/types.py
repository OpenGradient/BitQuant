from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Any
from enum import Enum


class Pool(BaseModel):
    name: str
    tokenA: str
    tokenB: str
    TVL: str
    APRLastDay: float
    APRLastWeek: float
    APRLastMonth: float


class TokenBalance(BaseModel):
    amount: float
    symbol: str


class PoolPosition(BaseModel):
    poolName: str
    amountDeposited: float


class ActionType(str, Enum):
    DEPOSIT = "depositToPool"
    WITHDRAW = "withdrawFromPool"


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


Action = Union[DepositAction, WithdrawAction]


class AgentOutput(BaseModel):
    message: str
    recommendedAction: List[Action]


Message = Union[str, AgentOutput]


class Context(BaseModel):
    conversationHistory: List[Message]
    tokens: List[TokenBalance]
    poolPositions: List[PoolPosition]
    availablePools: List[Pool]


class AgentRequest(BaseModel):
    context: Context
    userInput: str
