from typing import List, Dict, Any, Optional
import json
import requests
from functools import lru_cache

from defillama import DefiLlama

from defi.types import Pool, Chain, PoolQuery, Token


class DefiMetrics:

    llama: DefiLlama
    pools: List[Pool]
    tokenlist: Dict

    def __init__(self):
        self.llama = DefiLlama()
        self.pools = []

        with open("static/tokenlist.json", "r") as f:
            self.tokenlist = json.load(f)

    def get_pools(self, query: PoolQuery) -> List[Pool]:
        filtered_pools = self.pools.copy()

        if query.chain is not None:
            filtered_pools = [
                pool for pool in filtered_pools if pool.chain == query.chain
            ]

        if query.tokens:
            filtered_pools = [
                pool
                for pool in filtered_pools
                if any(
                    token in [t.address for t in pool.tokens] for token in query.tokens
                )
            ]

        if query.protocols is not None and len(query.protocols) > 0:
            filtered_pools = [
                pool for pool in filtered_pools if pool.protocol in query.protocols
            ]

        if query.isStableCoin is not None:
            filtered_pools = [
                pool
                for pool in filtered_pools
                if pool.isStableCoin == query.isStableCoin
            ]

        if query.impermanentLossRisk is not None:
            filtered_pools = [
                pool
                for pool in filtered_pools
                if pool.impermanentLossRisk == query.impermanentLossRisk
            ]

        return filtered_pools
    
    @lru_cache(maxsize=1)
    def get_protocols(self) -> List[Dict[str, Any]]:
        """Get all DeFi protocols from DefiLlama with caching"""
        try:
            protocols_data = self.llama.get_all_protocols()
            return protocols_data
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"Warning: Error when getting protocols: {e}. Returning empty list.")
            return []
    
    def get_protocol(self, protocol_slug: str) -> Dict[str, Any]:
        """Get details for a specific protocol by slug"""
        try:
            protocol_data = self.llama.get_protocol(protocol_slug)
            if not protocol_data:
                try:
                    protocol_tvl = self.llama.get_protocol_current_tvl(protocol_slug)
                    return {"name": protocol_slug, "tvl": protocol_tvl.get("tvl", 0)}
                except:
                    return {"name": protocol_slug, "tvl": 0}
            
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
            
            return protocol_data
        except Exception as e:
            print(f"Warning: Error with protocol {protocol_slug}: {e}")
            return {"name": protocol_slug, "tvl": 0}
    
    def get_global_tvl(self) -> Dict[str, Any]:
        """Get current global TVL across all DeFi protocols"""
        try:
            # Use get_chains_current_tvl which returns the current TVL for all chains
            chains_tvl = self.llama.get_chains_current_tvl()
            
            # Calculate the total TVL across all chains
            total_tvl = sum(float(chain_data.get("tvl", 0)) for chain_data in chains_tvl)
            
            # Format like the original method expected
            return {"totalLiquidityUSD": total_tvl}
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"Warning: Error when getting global TVL: {e}. Returning empty dict.")
            return {"totalLiquidityUSD": 0}
    
    def get_chain_tvl(self, chain: str) -> Dict[str, Any]:
        """Get TVL for a specific blockchain"""
        try:
            # Use get_chains_current_tvl which returns the current TVL for all chains
            chains_tvl = self.llama.get_chains_current_tvl()
            
            # Find the specific chain we're looking for
            for chain_data in chains_tvl:
                if chain_data.get("name", "").lower() == chain.lower():
                    return {"totalLiquidityUSD": chain_data.get("tvl", 0)}
            
            # If chain not found, return empty result
            return {"totalLiquidityUSD": 0}
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"Warning: Error when getting TVL for chain {chain}: {e}. Returning empty dict.")
            return {"totalLiquidityUSD": 0}
    
    def get_top_pools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top DeFi pools ranked by APY"""
        try:
            pools_data = self.llama.get_pools()
            if isinstance(pools_data, dict) and 'data' in pools_data:
                sorted_pools = sorted(
                    pools_data['data'], 
                    key=lambda x: x.get('apy', 0), 
                    reverse=True
                )
                return sorted_pools[:limit]
            return []
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"Warning: Error when getting top pools: {e}. Returning empty list.")
            return []

    def refresh_metrics(self):
        try:
            pools_response = self.llama.get_pools()
            self.pools = [self._convert_to_pool(p) for p in pools_response["data"]]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"Warning: Error when refreshing pools: {e}. Using existing pools.")
            # Keep using existing pools

    def _convert_to_pool(self, pool_data: Dict) -> Pool:
        return Pool(
            id=pool_data["pool"],
            chain=DefiMetrics._get_chain(pool_data["chain"]),
            tokens=[
                Token(
                    address=token_address,
                    symbol=(
                        self.tokenlist[token_address]["symbol"]
                        if token_address in self.tokenlist
                        else ""
                    ),
                    name=(
                        self.tokenlist[token_address]["name"]
                        if token_address in self.tokenlist
                        else ""
                    ),
                )
                for token_address in (pool_data.get("underlyingTokens") or [])
                if token_address is not None
            ],
            TVL=f"${pool_data['tvlUsd']}",
            APRLastDay=pool_data["apy"],
            APRLastWeek=pool_data["apyMean30d"],  # use 30d
            APRLastMonth=pool_data["apyMean30d"],
            protocol=pool_data["project"],
            isStableCoin=pool_data["stablecoin"],
            impermanentLossRisk=pool_data["ilRisk"],
            type="Lending",
            risk="Low",
        )

    @staticmethod
    def _get_chain(chain: str) -> Chain:
        if chain == "Ethereum":
            return Chain.ETHEREUM
        if chain == "Solana":
            return Chain.SOLANA
        if chain == "Base":
            return Chain.BASE
        else:
            return Chain.OTHER
