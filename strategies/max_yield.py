from typing import List

from plugins.types import Pool, Action, TokenBalance, PoolPosition
from strategies.strategy import Strategy


class MaxYieldStrategy(Strategy):

    def description(self) -> str:
        return ""

    def allocate(
        self,
        tokens: List[TokenBalance],
        available_pools: List[Pool],
        positions: List[PoolPosition],
    ) -> List[Action]:
        ordered_pools = MaxYieldStrategy.sort_pools_by_apy(available_pools)

        actions: List[Action] = []
        return actions

    @staticmethod
    def sort_pools_by_apy(pools: List[Pool]) -> List[Pool]:
        # Sort by APRLastDay in descending order (higher APR first)
        return sorted(pools, key=lambda x: x.APRLastDay, reverse=True)
