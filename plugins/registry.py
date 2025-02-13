from typing import Set, List, Dict

from plugins.plugin import Plugin
from plugins.navi.navi_plugin import NaviPlugin
from plugins.types import Pool, WalletPoolPosition, WalletTokenHolding


class PoolRegistry:

    pools_by_plugin: Dict[str, List[Pool]]

    def __init__(self, plugins: Set[str]):
        all_plugins: Dict[str, Plugin] = {
            "navi": NaviPlugin(),
        }

        filtered_plugins: Dict[str, Plugin] = {
            name: plugin for name, plugin in all_plugins.items() if name in plugins
        }
        for name, plugin in filtered_plugins.items():
            plugin.initialize()
        pools = {
            name: plugin.fetch_pools() for name, plugin in filtered_plugins.items()
        }

        self.pools_by_plugin = pools

    def get_compatible_pools(
        self, tokens: List[WalletTokenHolding], poolPositions: List[WalletPoolPosition]
    ) -> List[Pool]:
        """
        Returns a list of compatible pools based on user's token holdings and existing pool positions.
        A pool is compatible if either:
        1. The user has an existing position in the pool
        2. The user holds at least one of the tokens required by the pool

        Args:
            tokens (List[WalletTokenHolding]): List of tokens held by the user
            poolPositions (List[WalletPoolPosition]): List of user's existing pool positions

        Returns:
            List[Pool]: List of compatible pools
        """
        compatible_pools = []

        # Create a set of pool IDs where user has positions
        position_pool_ids = {position.poolId for position in poolPositions}

        # Create a set of token symbols held by user
        user_token_symbols = {holding.tokenSymbol for holding in tokens}

        # Iterate through all pools across all plugins
        for plugin_pools in self.pools_by_plugin.values():
            for pool in plugin_pools:
                # Check if user has a position in this pool
                if pool.id in position_pool_ids:
                    compatible_pools.append(pool)
                    continue

                # Check if user holds any of the tokens required by the pool
                pool_token_symbols = {token.symbol for token in pool.tokens}
                if user_token_symbols.intersection(pool_token_symbols):
                    compatible_pools.append(pool)

        return compatible_pools
