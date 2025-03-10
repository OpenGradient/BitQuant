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
        
        # Filter down to just essential information
        simplified_data = []
        for protocol in protocols_data:
            simplified_data.append({
                "name": protocol.get("name", "Unknown"),
                "slug": protocol.get("slug", ""),
                "tvl": protocol.get("tvl", 0),
                "chain": protocol.get("chain", "Unknown"),
                "category": protocol.get("category", "Other")
            })
        
        # Ensure no None TVL values before sorting
        for protocol in simplified_data:
            if protocol.get("tvl") is None:
                protocol["tvl"] = 0
        
        # Now safe to sort
        simplified_data.sort(key=lambda x: x.get("tvl", 0), reverse=True)
        
        # Return only top 50 protocols to prevent data overload
        return simplified_data[:50]

    def get_protocol(self, protocol_slug: str) -> Dict[str, Any]:
        """Get details for a specific protocol by slug"""
        protocol_data = self.llama.get_protocol(protocol_slug)
        
        # If no data returned, get basic TVL info
        if not protocol_data:
            protocol_tvl = self.llama.get_protocol_current_tvl(protocol_slug)
            return {"name": protocol_slug, "tvl": protocol_tvl.get("tvl", 0)}
        
        # Extract and format TVL data
        if protocol_data and 'tvl' in protocol_data:
            if isinstance(protocol_data['tvl'], list):
                last_entry = protocol_data['tvl'][-1] if protocol_data['tvl'] else 0
                if isinstance(last_entry, dict) and 'totalLiquidityUSD' in last_entry:
                    protocol_data['tvl'] = last_entry['totalLiquidityUSD']
                elif isinstance(last_entry, (int, float, str)):
                    protocol_data['tvl'] = float(last_entry)
                else:
                    protocol_data['tvl'] = 0
            elif isinstance(protocol_data['tvl'], dict):
                if 'tvl' in protocol_data['tvl']:
                    protocol_data['tvl'] = protocol_data['tvl']['tvl']
                elif 'totalLiquidityUSD' in protocol_data['tvl']:
                    protocol_data['tvl'] = protocol_data['tvl']['totalLiquidityUSD']
                else:
                    for k, v in protocol_data['tvl'].items():
                        if isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit():
                            protocol_data['tvl'] = float(v)
                            break
                    else:
                        protocol_data['tvl'] = 0
        
        # Create response with only essential data
        parsed_protocol_data = {
            "name": protocol_data.get("name", protocol_slug),
            "slug": protocol_data.get("slug", protocol_slug),
            "tvl": protocol_data.get("tvl", 0),
            "description": protocol_data.get("description", "No description available"),
            "chain": protocol_data.get("chain", "Unknown"),
            "category": protocol_data.get("category", "Other"),
            "url": protocol_data.get("url", ""),
            "twitter": protocol_data.get("twitter", ""),
            "chains": protocol_data.get("chains", [])[:10]  # Limit chains to 10
        }
        
        # If there's audit data, include a simplified version
        if "audit_links" in protocol_data and isinstance(protocol_data["audit_links"], list):
            parsed_protocol_data["has_audits"] = len(protocol_data["audit_links"]) > 0
        
        return parsed_protocol_data

    def get_global_tvl(self) -> float:
        """
        Get current global TVL across all DeFi protocols
        """
        chains_tvl = self.llama.get_chains_current_tvl()

        # Calculate the total TVL across all chains
        total_tvl = sum(float(chain_data.get("tvl", 0)) for chain_data in chains_tvl)

        return total_tvl

    def get_chain_tvl(self, chain: str) -> float:
        """
        Get TVL for a specific blockchain.
        """
        chains_tvl = self.llama.get_chains_current_tvl()

        # Find the specific chain we're looking for
        for chain_data in chains_tvl:
            if chain_data.get("name", "").lower() == chain.lower():
                return float(chain_data.get("tvl", 0))

        # If chain not found, return 0
        return 0

    def get_top_pools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top DeFi pools ranked by APY.
        """
        pools_data = self.llama.get_pools()
        if isinstance(pools_data, dict) and "data" in pools_data:
            sorted_pools = sorted(
                pools_data["data"], key=lambda x: x.get("apy", 0), reverse=True
            )
            return sorted_pools[:limit]
        return []
