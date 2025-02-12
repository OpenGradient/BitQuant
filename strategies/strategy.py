from abc import ABC, abstractmethod
from typing import List
from typing import Generic, TypeVar, Optional

from pydantic import BaseModel
from plugins.types import Pool, Action, WalletTokenHolding, WalletPoolPosition


T = TypeVar("T", bound=BaseModel)


class Strategy(ABC, Generic[T]):

    @abstractmethod
    def name(self) -> str:
        """Returns strategy's name for LLM."""
        pass

    @abstractmethod
    def description(self) -> str:
        """Returns strategy's description for LLM."""
        pass

    @abstractmethod
    def allocate(
        self,
        tokens: List[WalletTokenHolding],
        positions: List[WalletPoolPosition],
        available_pools: List[Pool],
        options: Optional[T],
    ) -> List[Action]:
        """Returns suggested action based on user's holdings and available pools"""
        pass
