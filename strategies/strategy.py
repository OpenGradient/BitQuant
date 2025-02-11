from abc import ABC, abstractmethod
from typing import List, Dict

from plugins.types import Pool, Action, TokenBalance, PoolPosition


def Strategy(ABC):

    @abstractmethod
    def allocate(
        tokens: List[TokenBalance],
        available_pools: List[Pool],
        positions: List[PoolPosition],
    ) -> List[Action]:
        pass
