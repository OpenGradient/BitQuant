from typing import List, Dict, Any
from functools import lru_cache

from defillama import DefiLlama


class DefiLlamaMetrics:

    llama: DefiLlama

    def __init__(self):
        self.llama = DefiLlama()

    @lru_cache(maxsize=1)
    def get_protocols(self) -> List[Dict[str, Any]]:
        """Get all DeFi protocols from DefiLlama with caching"""
        protocols_data = self.llama.get_all_protocols()
        return protocols_data

    def get_protocol(self, protocol_slug: str) -> Dict[str, Any]:
        """Get details for a specific protocol by slug"""
        protocol_data = self.llama.get_protocol(protocol_slug)
        if not protocol_data:
            protocol_tvl = self.llama.get_protocol_current_tvl(protocol_slug)
            return {"name": protocol_slug, "tvl": protocol_tvl.get("tvl", 0)}

        if protocol_data and "tvl" in protocol_data:
            if isinstance(protocol_data["tvl"], list):
                last_entry = protocol_data["tvl"][-1] if protocol_data["tvl"] else 0
                if isinstance(last_entry, dict) and "totalLiquidityUSD" in last_entry:
                    protocol_data["tvl"] = last_entry["totalLiquidityUSD"]
                elif isinstance(last_entry, (int, float, str)):
                    protocol_data["tvl"] = float(last_entry)
                else:
                    protocol_data["tvl"] = 0
            elif isinstance(protocol_data["tvl"], dict):
                if "tvl" in protocol_data["tvl"]:
                    protocol_data["tvl"] = protocol_data["tvl"]["tvl"]
                elif "totalLiquidityUSD" in protocol_data["tvl"]:
                    protocol_data["tvl"] = protocol_data["tvl"]["totalLiquidityUSD"]
                else:
                    for k, v in protocol_data["tvl"].items():
                        if (
                            isinstance(v, (int, float, str))
                            and str(v).replace(".", "", 1).isdigit()
                        ):
                            protocol_data["tvl"] = float(v)
                            break
                    else:
                        protocol_data["tvl"] = 0

        return protocol_data

    def get_global_tvl(self) -> float:
        """Get current global TVL across all DeFi protocols"""
        chains_tvl = self.llama.get_chains_current_tvl()

        # Calculate the total TVL across all chains
        total_tvl = sum(float(chain_data.get("tvl", 0)) for chain_data in chains_tvl)

        return total_tvl

    def get_chain_tvl(self, chain: str) -> float:
        """Get TVL for a specific blockchain"""
        chains_tvl = self.llama.get_chains_current_tvl()

        # Find the specific chain we're looking for
        for chain_data in chains_tvl:
            if chain_data.get("name", "").lower() == chain.lower():
                return float(chain_data.get("tvl", 0))

        # If chain not found, return 0
        return 0

    def get_top_pools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top DeFi pools ranked by APY"""
        pools_data = self.llama.get_pools()
        if isinstance(pools_data, dict) and "data" in pools_data:
            sorted_pools = sorted(
                pools_data["data"], key=lambda x: x.get("apy", 0), reverse=True
            )
            return sorted_pools[:limit]
        return []
