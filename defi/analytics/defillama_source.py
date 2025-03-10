from typing import List, Dict, Any
from functools import lru_cache

from defillama import DefiLlama


class DefiLlamaMetrics:
    """Class for interacting with DefiLlama API to fetch DeFi metrics."""

    llama: DefiLlama

    def __init__(self):
        self.llama = DefiLlama()

    @lru_cache(maxsize=1)
    def get_protocols(self) -> List[Dict[str, Any]]:
        """Retrieve all DeFi protocols from DefiLlama with caching.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing essential protocol information,
            limited to the top 50 protocols by TVL.
        """
        protocols_data = self.llama.get_all_protocols()
        
        simplified_data = []
        for protocol in protocols_data:
            simplified_data.append({
                "name": protocol.get("name", "Unknown"),
                "slug": protocol.get("slug", ""),
                "tvl": protocol.get("tvl", 0),
                "chain": protocol.get("chain", "Unknown"),
                "category": protocol.get("category", "Other")
            })
        
        for protocol in simplified_data:
            if protocol.get("tvl") is None:
                protocol["tvl"] = 0
        
        simplified_data.sort(key=lambda x: x.get("tvl", 0), reverse=True)
        
        return simplified_data[:50]

    def get_protocol(self, protocol_slug: str) -> Dict[str, Any]:
        """Get detailed information for a specific protocol identified by its slug.

        Args:
            protocol_slug (str): The slug of the protocol.

        Returns:
            Dict[str, Any]: A dictionary containing the protocol's details including TVL, description,
            chain, category, and audit info if available.
        """
        protocol_data = self.llama.get_protocol(protocol_slug)
        
        if not protocol_data:
            protocol_tvl = self.llama.get_protocol_current_tvl(protocol_slug)
            return {"name": protocol_slug, "tvl": protocol_tvl.get("tvl", 0)}
        
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
        
        parsed_protocol_data = {
            "name": protocol_data.get("name", protocol_slug),
            "slug": protocol_data.get("slug", protocol_slug),
            "tvl": protocol_data.get("tvl", 0),
            "description": protocol_data.get("description", "No description available"),
            "chain": protocol_data.get("chain", "Unknown"),
            "category": protocol_data.get("category", "Other"),
            "url": protocol_data.get("url", ""),
            "twitter": protocol_data.get("twitter", ""),
            "chains": protocol_data.get("chains", [])[:10]
        }
        
        if "audit_links" in protocol_data and isinstance(protocol_data["audit_links"], list):
            parsed_protocol_data["has_audits"] = len(protocol_data["audit_links"]) > 0
        
        return parsed_protocol_data

    def get_global_tvl(self) -> float:
        """Calculate the current global Total Value Locked (TVL) across all DeFi protocols.

        Returns:
            float: The global TVL as a float.
        """
        chains_tvl = self.llama.get_chains_current_tvl()
        total_tvl = sum(float(chain_data.get("tvl", 0)) for chain_data in chains_tvl)
        return total_tvl

    def get_chain_tvl(self, chain: str) -> float:
        """Retrieve the TVL for a specific blockchain.

        Args:
            chain (str): The target blockchain name.

        Returns:
            float: The TVL for the specified chain. Returns 0 if the chain is not found.
        """
        chains_tvl = self.llama.get_chains_current_tvl()

        for chain_data in chains_tvl:
            if chain_data.get("name", "").lower() == chain.lower():
                return float(chain_data.get("tvl", 0))

        return 0

    def get_top_pools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtain the top DeFi pools ranked by Annual Percentage Yield (APY).

        Args:
            limit (int, optional): Maximum number of pools to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with details for each pool.
        """
        pools_data = self.llama.get_pools()
        if isinstance(pools_data, dict) and "data" in pools_data:
            sorted_pools = sorted(
                pools_data["data"], key=lambda x: x.get("apy", 0), reverse=True
            )
            return sorted_pools[:limit]
        return []

    def get_pool(self, pool_id: str) -> Dict[str, Any]:
        """Fetch a specific DeFi pool's historical data using its unique identifier.

        Args:
            pool_id (str): The unique identifier for the pool.

        Returns:
            Dict[str, Any]: A dictionary containing the pool's historical data if found;
            otherwise, a dictionary with an error message.
        """
        try:
            chart_data = self.llama.get_pool(pool_id)
                
            result = {
                "status": chart_data.get("status", "unknown"),
                "data": chart_data.get("data", []),
            }
            
            if result["data"] and len(result["data"]) > 0:
                latest = result["data"][-1]
                result["latest"] = {
                    "tvl": latest.get("tvlUsd", 0),
                    "apy": latest.get("apy", 0),
                    "timestamp": latest.get("timestamp")
                }
            
            return result
                
        except Exception as e:
            return {
                "error": f"Failed to fetch pool with ID '{pool_id}'",
                "details": str(e)
            }

    def get_all_protocol_slugs(self) -> List[str]:
        """Get a list of all protocol slugs available in DeFi Llama.
        
        Returns:
            List[str]: A list of all protocol slugs.
        """
        protocols_data = self.llama.get_all_protocols()
        slugs = [protocol.get("slug") for protocol in protocols_data if protocol.get("slug")]
        return sorted(slugs)
    
    def get_all_pool_ids(self) -> List[Dict[str, str]]:
        """Get a list of all pool IDs with their associated project names.
        
        Returns:
            List[Dict[str, str]]: A list of dictionaries containing pool ID, project name, and symbol.
        """
        pools_data = self.llama.get_pools()
        result = []
        
        if isinstance(pools_data, dict) and "data" in pools_data:
            for pool in pools_data["data"]:
                pool_id = None
                for field in ['id', 'pool', 'identifier']:
                    if field in pool:
                        pool_id = pool[field]
                        break
                
                if not pool_id and 'project' in pool and 'symbol' in pool:
                    pool_id = f"{pool.get('project')}-{pool.get('symbol')}"
                
                if pool_id:
                    result.append({
                        "id": pool_id,
                        "project": pool.get("project", "Unknown"),
                        "symbol": pool.get("symbol", "Unknown"),
                        "chain": pool.get("chain", "Unknown")
                    })
        
        return result