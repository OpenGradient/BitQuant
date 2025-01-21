from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Any
from enum import Enum


class Pool(BaseModel):
    name: str
    TVL: str
    APRLastDay: float
    APRLastWeek: float
    APRLastMonth: float


class TokenBalance(BaseModel):
    amount: float
    symbol: str


class PoolPosition(BaseModel):
    # Unique name of pool
    poolName: str

    # User's deposit in USD 
    depositedValue: float


class ActionType(str, Enum):
    DEPOSIT = "depositToPool"
    WITHDRAW = "withdrawFromPool"


class DepositAction(BaseModel):
    type: ActionType = ActionType.DEPOSIT
    pool: str
    amount: float
    asset: str


class WithdrawAction(BaseModel):
    type: ActionType = ActionType.WITHDRAW
    pool: str
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
