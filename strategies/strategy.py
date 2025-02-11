from abc import ABC, abstractmethod
from typing import List, Dict

from plugins.types import Pool, Action, TokenBalance, PoolPosition


def Strategy(ABC):

    @abstractmethod
    def description() -> str:
        """Returns strategy's description for LLM."""
        pass

    @abstractmethod
    def allocate(
        tokens: List[TokenBalance],
        available_pools: List[Pool],
        positions: List[PoolPosition],
    ) -> List[Action]:
        """Returns suggested action based on user's holdings and available pools"""
        pass
