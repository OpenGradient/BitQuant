from abc import ABC, abstractmethod
from typing import List

from plugins.types import Pool, Action, TokenBalance, PoolPosition


class Strategy(ABC):

    @abstractmethod
    def description(self) -> str:
        """Returns strategy's description for LLM."""
        pass

    @abstractmethod
    def allocate(
        self,
        tokens: List[TokenBalance],
        positions: List[PoolPosition],
        available_pools: List[Pool],
    ) -> List[Action]:
        """Returns suggested action based on user's holdings and available pools"""
        pass
