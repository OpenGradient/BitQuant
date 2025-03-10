import unittest
from typing import List, Dict, Any
from api.api_types import Chain, Pool, PoolQuery
from defi.analytics.defillama_tools import (
    show_defi_llama_protocols,
    show_defi_llama_protocol,
    show_defi_llama_global_tvl,
    show_defi_llama_chain_tvl,
    show_defi_llama_top_pools,
    show_defi_llama_pool
)

def format_tvl(value) -> str:
    """Helper function to safely format TVL for display"""
    if value is None:
        return "$0.00"
    try:
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return "$0.00"

class TestDefiLlamaSource(unittest.TestCase):
    
    def test_show_defi_llama_protocols(self):
        """Test the show_defi_llama_protocols tool"""
        print("\n=== Testing show_defi_llama_protocols() ===")
        
        protocols = show_defi_llama_protocols.invoke({})
        
        self.assertIsNotNone(protocols)
        self.assertTrue(len(protocols) > 0, "Should find at least one protocol")
        
        print(f"\nTop 10 protocols by TVL:")
        for i, protocol in enumerate(protocols[:10], 1):
            print(f"{i}. {protocol.get('name', 'Unknown')} - {format_tvl(protocol.get('tvl', 0))}")
            print(f"   Slug: {protocol.get('slug', '')}")
            print(f"   Chain: {protocol.get('chain', 'Unknown')}")
            print(f"   Category: {protocol.get('category', 'Other')}")
            print("")
    
    def test_show_defi_llama_protocol(self):
        """Test the show_defi_llama_protocol tool"""
        print("\n=== Testing show_defi_llama_protocol('aave-v3') ===")
        
        protocol = show_defi_llama_protocol.invoke({"protocol_slug": "aave-v3"})
        
        self.assertIsNotNone(protocol)
        self.assertIn("name", protocol)
        self.assertIn("tvl", protocol)
        
        print(f"\nProtocol Details for 'aave-v3':")
        print(f"Name: {protocol.get('name', 'Unknown')}")
        print(f"Slug: {protocol.get('slug', '')}")
        print(f"TVL: {format_tvl(protocol.get('tvl', 0))}")
        print(f"Category: {protocol.get('category', 'Unknown')}")
        
        chains = protocol.get('chains', [])
        if chains:
            print(f"Chains: {', '.join(chains[:5])}")
            if len(chains) > 5:
                print(f"   ...and {len(chains) - 5} more")
        
        if 'description' in protocol:
            print(f"Description: {protocol['description']}")
            
        if 'url' in protocol:
            print(f"Website: {protocol['url']}")
        if 'twitter' in protocol:
            print(f"Twitter: {protocol['twitter']}")
    
    def test_show_defi_llama_global_tvl(self):
        """Test the show_defi_llama_global_tvl tool"""
        print("\n=== Testing show_defi_llama_global_tvl() ===")
        
        global_tvl = show_defi_llama_global_tvl.invoke({})
        
        self.assertIsNotNone(global_tvl)
        
        print(f"\nGlobal DeFi TVL: {format_tvl(global_tvl)}")
    
    def test_show_defi_llama_chain_tvl(self):
        """Test the show_defi_llama_chain_tvl tool"""
        print("\n=== Testing show_defi_llama_chain_tvl('ethereum') ===")
        
        eth_tvl = show_defi_llama_chain_tvl.invoke({"chain": "ethereum"})
        
        self.assertIsNotNone(eth_tvl)
        
        print(f"\nEthereum TVL: {format_tvl(eth_tvl)}")
    
    def test_show_defi_llama_top_pools(self):
        """Test the show_defi_llama_top_pools tool"""
        print("\n=== Testing show_defi_llama_top_pools() ===")
        
        top_pools = show_defi_llama_top_pools.invoke({"limit": 10})
        
        self.assertIsNotNone(top_pools)
        
        print(f"\nTop 10 Pools by APY:")
        
        for i, pool in enumerate(top_pools, 1):
            apy = pool.get('apy')
            apy_formatted = f"{float(apy):,.2f}%" if apy is not None else "N/A"
            
            print(f"{i}. {pool.get('project', 'Unknown')} ({pool.get('symbol', 'Unknown')})")
            print(f"   Chain: {pool.get('chain', 'Unknown')}")
            print(f"   APY: {apy_formatted}")
            print(f"   TVL: {format_tvl(pool.get('tvl', 0))}")
            
            if 'ilRisk' in pool:
                print(f"   IL Risk: {pool.get('ilRisk', 'Unknown')}")
                
            if 'stablecoin' in pool:
                stable = "Yes" if pool.get('stablecoin') else "No"
                print(f"   Stablecoin: {stable}")
                
            print("")
    
    def test_show_defi_llama_pool(self):
        """Test the show_defi_llama_pool tool"""
        print("\n=== Testing show_defi_llama_pool() ===")
        
        top_pools = show_defi_llama_top_pools.invoke({"limit": 1})
        
        if top_pools and len(top_pools) > 0:
            pool_id = None
            for field in ['id', 'pool', 'identifier']:
                if field in top_pools[0]:
                    pool_id = top_pools[0][field]
                    break
            
            if not pool_id:
                pool_id = f"{top_pools[0].get('project')}-{top_pools[0].get('symbol')}"
            
            print(f"Testing with pool ID: {pool_id}")
            
            pool = show_defi_llama_pool.invoke({"pool_id": pool_id})
            
            self.assertIsNotNone(pool)
            self.assertNotIn("error", pool, "The pool should be found")
            
            print("\nPool Details:")
            print(f"Status: {pool.get('status', 'unknown')}")
            
            if 'latest' in pool:
                latest = pool['latest']
                print("\nLatest Data:")
                print(f"  TVL: {format_tvl(latest.get('tvl', 0))}")
                print(f"  APY: {latest.get('apy', 0):.2f}%")
                if 'timestamp' in latest:
                    from datetime import datetime
                    timestamp = latest['timestamp']
                    try:
                        if isinstance(timestamp, (int, float)):
                            ts = datetime.fromtimestamp(timestamp)
                        else:
                            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        print(f"  Timestamp: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
                    except (ValueError, TypeError):
                        print(f"  Timestamp: {timestamp} (raw)")
            
            if 'data' in pool and isinstance(pool['data'], list):
                print(f"\nHistorical Data Points: {len(pool['data'])}")
                if pool['data'] and len(pool['data']) > 0:
                    print("\nSample (First data point):")
                    first = pool['data'][0]
                    print(f"  TVL: {format_tvl(first.get('tvlUsd', 0))}")
                    print(f"  APY: {first.get('apy', 0):.2f}%")
                    
                    other_fields = [k for k in first.keys() if k not in ['tvlUsd', 'apy', 'timestamp']]
                    if other_fields:
                        print("\nOther available fields in data points:", ", ".join(other_fields))
        else:
            self.fail("Could not find any pools to test with")


if __name__ == "__main__":
    unittest.main()
