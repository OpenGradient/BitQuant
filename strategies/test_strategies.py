import unittest
from typing import List
from defi.types import (
    Pool,
    Token,
    WalletTokenHolding,
    WalletPoolPosition,
    DepositAction,
    WithdrawAction,
)
from .max_yield import MaxYieldStrategy, MaxYieldOptions


class TestMaxYieldStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MaxYieldStrategy()

        self.base_tokens = [
            Token(symbol="USDC", price=1.0),
            Token(symbol="ETH", price=2000.0),
            Token(symbol="WBTC", price=40000.0),
        ]
        self.sample_pools = [
            Pool(
                id="pool1",
                tokens=[self.base_tokens[0], self.base_tokens[1]],  # USDC-ETH
                TVL="1000000",
                APRLastDay=0.1,  # 10% APR
                APRLastWeek=0.09,
                APRLastMonth=0.08,
                protocol="Protocol1",
            ),
            Pool(
                id="pool2",
                tokens=[self.base_tokens[0], self.base_tokens[2]],  # USDC-WBTC
                TVL="2000000",
                APRLastDay=0.15,  # 15% APR
                APRLastWeek=0.14,
                APRLastMonth=0.13,
                protocol="Protocol2",
            ),
            Pool(
                id="pool3",
                tokens=[self.base_tokens[1], self.base_tokens[2]],  # ETH-WBTC
                TVL="1500000",
                APRLastDay=0.05,  # 5% APR
                APRLastWeek=0.05,
                APRLastMonth=0.05,
                protocol="Protocol3",
            ),
        ]

    def test_basic_allocation_no_existing_positions(self):
        """Test basic allocation with no existing positions"""
        tokens = [
            WalletTokenHolding(tokenSymbol="USDC", amount=1000.0),
            WalletTokenHolding(tokenSymbol="ETH", amount=1.0),
            WalletTokenHolding(tokenSymbol="WBTC", amount=0.05),
        ]
        positions: List[WalletPoolPosition] = []
        options = MaxYieldOptions(allow_reallocate=False)

        actions = self.strategy.allocate(tokens, positions, self.sample_pools, options)

        # Should allocate to highest APR pool first (pool2)
        self.assertTrue(len(actions) > 0)
        first_action = actions[0]
        self.assertIsInstance(first_action, DepositAction)
        self.assertEqual(first_action.pool, "pool2")
        self.assertIn("USDC", first_action.tokens)
        self.assertIn("WBTC", first_action.tokens)

    def test_reallocation_from_lower_to_higher_yield(self):
        """Test reallocation from lower yield to higher yield pools when enabled"""
        tokens = [
            WalletTokenHolding(tokenSymbol="USDC", amount=100.0),
        ]
        positions = [
            WalletPoolPosition(
                poolId="pool3",  # Lowest yield pool
                depositedTokens={"ETH": 0.5, "WBTC": 0.025},
            )
        ]
        options = MaxYieldOptions(allow_reallocate=True)

        actions = self.strategy.allocate(tokens, positions, self.sample_pools, options)

        # Should find at least one withdraw action from pool3
        withdraw_actions = [
            a for a in actions if isinstance(a, WithdrawAction) and a.pool == "pool3"
        ]
        self.assertTrue(len(withdraw_actions) > 0)

        # Should find at least one deposit action to pool2
        deposit_actions = [
            a for a in actions if isinstance(a, DepositAction) and a.pool == "pool2"
        ]
        self.assertTrue(len(deposit_actions) > 0)

    def test_no_reallocation_when_disabled(self):
        """Test that no reallocation happens when it's disabled"""
        tokens = [
            WalletTokenHolding(tokenSymbol="USDC", amount=100.0),
        ]
        positions = [
            WalletPoolPosition(
                poolId="pool3",  # Lowest yield pool
                depositedTokens={"ETH": 0.5, "WBTC": 0.025},
            )
        ]
        options = MaxYieldOptions(allow_reallocate=False)

        actions = self.strategy.allocate(tokens, positions, self.sample_pools, options)

        # Should not find any withdraw actions
        withdraw_actions = [a for a in actions if isinstance(a, WithdrawAction)]
        self.assertEqual(len(withdraw_actions), 0)

    def test_balanced_token_deposits(self):
        """Test that deposits maintain balanced USD values across tokens"""
        tokens = [
            WalletTokenHolding(tokenSymbol="USDC", amount=2000.0),  # $2000
            WalletTokenHolding(tokenSymbol="ETH", amount=1.0),  # $2000
        ]
        positions: List[WalletPoolPosition] = []
        options = MaxYieldOptions(allow_reallocate=False)

        actions = self.strategy.allocate(tokens, positions, self.sample_pools, options)

        for action in actions:
            if isinstance(action, DepositAction):
                # Calculate USD values for each token
                usd_values = {}
                for token, amount in action.tokens.items():
                    if token == "USDC":
                        usd_values[token] = amount * 1.0
                    elif token == "ETH":
                        usd_values[token] = amount * 2000.0
                    elif token == "WBTC":
                        usd_values[token] = amount * 40000.0

                # Check that all USD values are approximately equal
                if len(usd_values) > 1:
                    values = list(usd_values.values())
                    max_diff = max(values) - min(values)
                    self.assertLess(
                        max_diff, 1.0
                    )  # Allow for small rounding differences

    def test_empty_wallet_and_positions(self):
        """Test behavior with empty wallet and no positions"""
        tokens: List[WalletTokenHolding] = []
        positions: List[WalletPoolPosition] = []
        options = MaxYieldOptions(allow_reallocate=False)

        actions = self.strategy.allocate(tokens, positions, self.sample_pools, options)
        self.assertEqual(len(actions), 0)

    def test_insufficient_token_amounts(self):
        """Test behavior with insufficient token amounts"""
        tokens = [
            WalletTokenHolding(tokenSymbol="USDC", amount=0.000001),
            WalletTokenHolding(tokenSymbol="ETH", amount=0.000001),
        ]
        positions: List[WalletPoolPosition] = []
        options = MaxYieldOptions(allow_reallocate=False)

        actions = self.strategy.allocate(tokens, positions, self.sample_pools, options)

        # Check that any deposit actions have positive amounts
        for action in actions:
            if isinstance(action, DepositAction):
                for amount in action.tokens.values():
                    self.assertGreater(amount, 0)

    def test_multiple_reallocations(self):
        """Test multiple reallocations from different pools"""
        tokens = [
            WalletTokenHolding(tokenSymbol="USDC", amount=100.0),
        ]
        positions = [
            WalletPoolPosition(
                poolId="pool1", depositedTokens={"USDC": 500.0, "ETH": 0.25}
            ),
            WalletPoolPosition(
                poolId="pool3", depositedTokens={"ETH": 0.5, "WBTC": 0.025}
            ),
        ]
        options = MaxYieldOptions(allow_reallocate=True)

        actions = self.strategy.allocate(tokens, positions, self.sample_pools, options)

        # Should find withdrawals from both lower yield pools
        withdrawals = [a for a in actions if isinstance(a, WithdrawAction)]
        self.assertEqual(len(withdrawals), 2)

        # Should find at least one deposit to highest yield pool
        deposits = [a for a in actions if isinstance(a, DepositAction)]
        self.assertTrue(len(deposits) > 0)
        self.assertTrue(any(d.pool == "pool2" for d in deposits))

    def test_single_token_pool(self):
        """Test behavior with single-token pools"""
        single_token_pool = Pool(
            id="single_pool",
            tokens=[self.base_tokens[0]],  # Single token pool
            TVL="1000000",
            APRLastDay=0.2,
            APRLastWeek=0.19,
            APRLastMonth=0.18,
            protocol="Protocol1",
        )

        tokens = [WalletTokenHolding(tokenSymbol="USDC", amount=1000.0)]
        positions: List[WalletPoolPosition] = []
        options = MaxYieldOptions(allow_reallocate=False)

        actions = self.strategy.allocate(
            tokens, positions, [single_token_pool], options
        )
        self.assertTrue(len(actions) > 0)


if __name__ == "__main__":
    unittest.main()
