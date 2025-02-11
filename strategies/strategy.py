from abc import ABC, abstractmethod
from typing import List
from typing import Generic, TypeVar

from plugins.types import Pool, Action, WalletTokenHolding, WalletPoolPosition


T = TypeVar("T")


class Strategy(ABC, Generic[T]):

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
        options: T,
    ) -> List[Action]:
        """Returns suggested action based on user's holdings and available pools"""
        pass
