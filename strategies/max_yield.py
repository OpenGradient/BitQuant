from typing import List, Dict, Optional, Set
from plugins.types import (
    Pool,
    Action,
    WalletTokenHolding,
    WalletPoolPosition,
    DepositAction,
    WithdrawAction,
    Token,
)
from strategies.strategy import Strategy


class MaxYieldOptions:
    def __init__(self, allow_reallocate: bool):
        self.allow_reallocate = allow_reallocate


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
        options: MaxYieldOptions,
    ) -> List[Action]:
        actions: List[Action] = []

        # Sort pools by APR in descending order
        ordered_pools = self.sort_pools_by_apy(available_pools)

        # Create lookup dictionaries for easier access
        token_holdings = {t.tokenSymbol: t.amount for t in tokens}
        pool_positions = {p.poolId: p.depositedTokens for p in positions}

        # Create a mapping of tokens to their available amounts (including what's in pools)
        total_token_amounts = token_holdings.copy()
        for position in positions:
            for token, amount in position.depositedTokens.items():
                if token not in total_token_amounts:
                    total_token_amounts[token] = 0
                total_token_amounts[token] += amount

        # If reallocation is allowed, consider withdrawing from lower-yield pools
        if options.allow_reallocate:
            # Find pools with lower APR than available alternatives for the same tokens
            for pool_id, deposited_tokens in pool_positions.items():
                current_pool = next(
                    (p for p in available_pools if p.id == pool_id), None
                )
                if not current_pool:
                    continue

                # Find better pools for these tokens
                better_pools = [
                    p
                    for p in ordered_pools
                    if p.APRLastDay > current_pool.APRLastDay
                    and all(
                        token in [t.symbol for t in p.tokens]
                        for token in deposited_tokens.keys()
                    )
                ]

                if better_pools:
                    # Withdraw from current pool to reallocate to better ones
                    actions.append(
                        WithdrawAction(pool=pool_id, tokens=deposited_tokens)
                    )
                    # Update available amounts
                    for token, amount in deposited_tokens.items():
                        token_holdings[token] = token_holdings.get(token, 0) + amount

        # Try to allocate tokens to highest yielding pools
        for pool in ordered_pools:
            pool_tokens = [t.symbol for t in pool.tokens]

            # Skip if we don't have any of the required tokens
            if not any(token in token_holdings for token in pool_tokens):
                continue

            # Calculate the maximum amount we can deposit while keeping ratios equal
            max_deposit_amounts = {}
            for token in pool_tokens:
                if token in token_holdings and token_holdings[token] > 0:
                    token_price = next(
                        t.price for t in pool.tokens if t.symbol == token
                    )
                    max_deposit_amounts[token] = token_holdings[token] * token_price

            if not max_deposit_amounts:
                continue

            # Find the minimum USD value that can be deposited equally across all tokens
            min_usd_value = min(max_deposit_amounts.values())

            # Calculate token amounts to deposit
            deposit_amounts = {}
            for token in pool_tokens:
                if token in token_holdings and token_holdings[token] > 0:
                    token_price = next(
                        t.price for t in pool.tokens if t.symbol == token
                    )
                    amount = min_usd_value / token_price
                    if amount > 0:
                        deposit_amounts[token] = amount
                        token_holdings[token] -= amount

            if deposit_amounts:
                actions.append(DepositAction(pool=pool.id, tokens=deposit_amounts))

        return actions

    @staticmethod
    def sort_pools_by_apy(pools: List[Pool]) -> List[Pool]:
        # Sort by APRLastDay in descending order (higher APR first)
        return sorted(pools, key=lambda x: x.APRLastDay, reverse=True)
