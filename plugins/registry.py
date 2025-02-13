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
        pass
