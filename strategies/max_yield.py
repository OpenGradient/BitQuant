import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from plugins.types import (
    Pool,
    Action,
    WalletTokenHolding,
    WalletPoolPosition,
    DepositAction,
    WithdrawAction,
)
from strategies.strategy import Strategy


class MaxYieldOptions(BaseModel):
    # whether we should withdraw existing positions if there are better pools
    allow_reallocate: bool = Field(
        default=False,
        description="Whether withdrawals should be allowed. Default to False",
    )

    token_allowlist: Optional[List[str]] = Field(
        default=None,
        description="The allowed token symbols to trade. Leave empty unless user explicitly asks",
    )


class MaxYieldStrategy(Strategy[MaxYieldOptions]):
    """
    Implements a strategy that allocates tokens to the overall highest yielding pools.
    Only withdraws and deposits the same $ amount for each token if there are multiple tokens in the pool.
    Optionally re-allocates existing positions if there are better ways to utilize those deployed tokens.
    """

    def name(self) -> str:
        return "MaxYieldStrategy"

    def description(self) -> str:
        return "Maximizes yield by allocating funds to pools with the highest APR, withdrawing from lower-yield positions when necessary."

    def allocate(
        self,
        tokens: List[WalletTokenHolding],
        positions: List[WalletPoolPosition],
        available_pools: List[Pool],
        options: Optional[MaxYieldOptions],
    ) -> List[Action]:
        if options is None:
            # default option
            options = MaxYieldOptions(allow_reallocate=True, token_allowlist=None)

        # Accounting
        total_tokens: Dict[str, float] = {t.tokenSymbol: t.amount for t in tokens}
        for position in positions:
            for token, amount in position.depositedTokens.items():
                total_tokens[token] = total_tokens.get(token, 0) + amount
        token_prices: Dict[str, float] = {
            token.symbol: token.price
            for pool in available_pools
            for token in pool.tokens
        }
        ordered_pools: List[Pool] = sorted(
            available_pools, key=lambda x: x.APRLastDay, reverse=True
        )

        # Calculate optimal allocation
        optimal_allocation = self._calculate_optimal_allocation(
            total_tokens, token_prices, ordered_pools
        )

        # Generate actions to achieve optimal allocation
        return self._generate_actions(
            optimal_allocation, positions, options.allow_reallocate
        )

    def _calculate_optimal_allocation(
        self,
        total_tokens: Dict[str, float],
        token_prices: Dict[str, float],
        ordered_pools: List[Pool],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculates the optimal allocation of tokens across pools.
        Returns a dict mapping pool IDs to token amounts.
        """
        optimal_allocation: Dict[str, Dict[str, float]] = {}
        remaining_tokens = total_tokens.copy()

        for pool in ordered_pools:
            pool_tokens = [t.symbol for t in pool.tokens]

            # Skip if we don't have any of the required tokens
            if not any(
                token in remaining_tokens and remaining_tokens[token] > 0
                for token in pool_tokens
            ):
                continue

            # Calculate maximum possible deposit in USD terms
            max_usd_deposits = [
                remaining_tokens.get(token, 0) * token_prices[token]
                for token in pool_tokens
                if token in remaining_tokens
            ]

            if not max_usd_deposits:
                continue

            # Find minimum USD value that can be deposited
            min_usd_value = min(max_usd_deposits)

            # Calculate token amounts
            allocation: Dict[str, float] = {}
            for token in pool_tokens:
                if token in remaining_tokens:
                    amount = min_usd_value / token_prices[token]
                    if amount > 0:
                        allocation[token] = amount
                        remaining_tokens[token] -= amount

            if allocation:
                optimal_allocation[pool.id] = allocation

        return optimal_allocation

    def _generate_actions(
        self,
        optimal_allocation: Dict[str, Dict[str, float]],
        current_positions: List[WalletPoolPosition],
        allow_reallocate: bool,
    ) -> List[Action]:
        """
        Generates the necessary actions to achieve the optimal allocation.
        """
        actions: List[Action] = []

        # Convert current positions to dict for easier lookup
        current_allocations = {
            pos.poolId: pos.depositedTokens for pos in current_positions
        }

        # Generate withdrawals for positions not in optimal allocation
        if allow_reallocate:
            for pool_id, current_tokens in current_allocations.items():
                if (
                    pool_id not in optimal_allocation
                    or optimal_allocation[pool_id] != current_tokens
                ):
                    actions.append(WithdrawAction(pool=pool_id, tokens=current_tokens))

        # Generate deposits for optimal allocation
        for pool_id, tokens in optimal_allocation.items():
            if (
                pool_id not in current_allocations
                or current_allocations[pool_id] != tokens
            ):
                actions.append(DepositAction(pool=pool_id, tokens=tokens))

        return actions
