from typing import List, Dict, Optional

from plugins.types import (
    Pool,
    Action,
    WalletTokenHolding,
    WalletPoolPosition,
    DepositAction,
    WithdrawAction,
)
from strategies.strategy import Strategy


class MaxYieldOptions:
    def __init__(self, allow_reallocate: bool):
        self.allow_reallocate = allow_reallocate


class MaxYieldStrategy(Strategy[MaxYieldOptions]):

    def description(self) -> str:
        return "Maximizes yield by allocating funds to pools with the highest APR, withdrawing from lower-yield positions when necessary."

    def allocate(
        self,
        tokens: List[WalletTokenHolding],
        positions: List[WalletPoolPosition],
        available_pools: List[Pool],
        options: MaxYieldOptions,
    ) -> List[Action]:
        # Create a map of token symbol to amount for quick lookup
        token_balances: Dict[str, float] = {
            holding.tokenSymbol: holding.amount for holding in tokens
        }

        # Create a map of pool ID to position for quick lookup
        pool_positions: Dict[str, WalletPoolPosition] = {
            position.poolId: position for position in positions
        }

        # Sort pools by APR in descending order
        ordered_pools = self.sort_pools_by_apy(available_pools)
        actions: List[Action] = []

        # Handle reallocation from low-performing pools if allowed
        if options.allow_reallocate:
            for pool in reversed(ordered_pools):
                position = pool_positions.get(pool.id)
                if position:
                    # Check if there's a better pool for these tokens
                    better_pool = self._find_better_pool(pool, position, ordered_pools)
                    if better_pool:
                        # Withdraw from current position
                        withdraw_action = WithdrawAction(
                            pool=pool.id, tokens=position.depositedTokens
                        )
                        actions.append(withdraw_action)

                        # Update token balances with withdrawn amounts
                        for token, amount in position.depositedTokens.items():
                            token_balances[token] = (
                                token_balances.get(token, 0) + amount
                            )

        # Then, deposit into high-performing pools
        for pool in ordered_pools:
            # Check if we have the required tokens for this pool
            deposit_amounts = self._calculate_deposit_amounts(pool, token_balances)
            if deposit_amounts:
                deposit_action = DepositAction(pool=pool.id, tokens=deposit_amounts)
                actions.append(deposit_action)

                # Update token balances after deposit
                for token, amount in deposit_amounts.items():
                    token_balances[token] -= amount

        return actions

    def _find_better_pool(
        self,
        current_pool: Pool,
        position: WalletPoolPosition,
        ordered_pools: List[Pool],
    ) -> Optional[Pool]:
        """Find a higher-yielding pool that accepts the same tokens."""
        position_tokens = set(position.depositedTokens.keys())

        for pool in ordered_pools:
            if pool.id == current_pool.id:
                continue

            if pool.APRLastDay <= current_pool.APRLastDay:
                break

            pool_tokens = set(pool.tokenSymbols)
            if position_tokens.issubset(pool_tokens):
                return pool

        return None

    def _calculate_deposit_amounts(
        self, pool: Pool, token_balances: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate optimal deposit amounts for a pool based on available balances."""
        deposit_amounts = {}

        # Check if we have all required tokens
        for token in pool.tokenSymbols:
            balance = token_balances.get(token, 0)
            if balance <= 0:
                return {}

            deposit_amounts[token] = balance * 0.95  # Keep 5% in reserve

        return deposit_amounts

    @staticmethod
    def sort_pools_by_apy(pools: List[Pool]) -> List[Pool]:
        # Sort by APRLastDay in descending order (higher APR first)
        return sorted(pools, key=lambda x: x.APRLastDay, reverse=True)
