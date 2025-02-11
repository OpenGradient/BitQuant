from typing import List

from plugins.types import (
    Pool,
    Action,
    TokenBalance,
    PoolPosition,
    DepositAction,
    WithdrawAction,
)
from strategies.strategy import Strategy


class MaxYieldStrategy(Strategy):

    def description(self) -> str:
        return "Maximizes yield by allocating funds to pools with the highest APR, withdrawing from lower-yield positions when necessary."

    def allocate(
        self,
        tokens: List[TokenBalance],
        positions: List[PoolPosition],
        available_pools: List[Pool],
    ) -> List[Action]:
        ordered_pools = MaxYieldStrategy.sort_pools_by_apy(available_pools)
        actions: List[Action] = []

        # Calculate total value available for allocation
        total_value = sum(token.amount for token in tokens)

        # First, withdraw from any positions in low-yield pools
        current_positions = {pos.poolName: pos for pos in positions}
        for pool in reversed(ordered_pools):  # Start with lowest yield pools
            if pool.name in current_positions:
                position = current_positions[pool.name]
                # Find matching token for withdrawal
                for token in tokens:
                    # If we have a position in a lower-yield pool, withdraw it
                    if position.depositedValue > 0:
                        actions.append(
                            WithdrawAction(
                                pool=pool.name,
                                amount=position.depositedValue,
                                asset=token.symbol,
                            )
                        )
                        total_value += position.depositedValue

        # Then deposit into highest-yield pools
        remaining_value = total_value
        for pool in ordered_pools:
            if remaining_value <= 0:
                break

            # Find suitable token for deposit
            for token in tokens:
                if token.amount > 0:
                    deposit_amount = min(remaining_value, token.amount)
                    actions.append(
                        DepositAction(
                            pool=pool.name, amount=deposit_amount, asset=token.symbol
                        )
                    )
                    remaining_value -= deposit_amount
                    break

        return actions

    @staticmethod
    def sort_pools_by_apy(pools: List[Pool]) -> List[Pool]:
        # Sort by APRLastDay in descending order (higher APR first)
        return sorted(pools, key=lambda x: x.APRLastDay, reverse=True)
