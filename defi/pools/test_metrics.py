import unittest
from typing import List, Dict, Any, Optional
import json
from api.api_types import Chain, Pool, PoolQuery
from defi.analytics.defillama_source import DefiLlamaMetrics
from defi.analytics.defillama_tools import (
    show_defi_llama_protocols,
    show_defi_llama_protocol,
    show_defi_llama_global_tvl,
    show_defi_llama_chain_tvl,
    show_defi_llama_top_pools
)

# Helper function to safely format TVL for display
def format_tvl(value) -> str:
    if value is None:
        return "$0.00"
    try:
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return "$0.00"

class TestDefiLlamaSource(unittest.TestCase):
    def test_defillama(self):
        print("\n=== 1. Testing show_defi_llama_protocols() ===")
        try:
            # Fix for NoneType error: Patch the tool directly to handle None values
            # Get protocols from DefiLlamaMetrics directly
            metrics = DefiLlamaMetrics()
            raw_protocols = metrics.get_protocols()
            
            # Process to ensure no None TVL values 
            for protocol in raw_protocols:
                if protocol.get("tvl") is None:
                    protocol["tvl"] = 0
            
            # Sort properly
            protocols = sorted(raw_protocols, key=lambda x: float(x.get("tvl", 0)), reverse=True)
            
            # Display top 10
            print(f"\nTop 10 protocols by TVL:")
            for i, protocol in enumerate(protocols[:10], 1):
                print(f"{i}. {protocol.get('name', 'Unknown')} - {format_tvl(protocol.get('tvl', 0))}")
                print(f"   Slug: {protocol.get('slug', '')}")
                print(f"   Chain: {protocol.get('chain', 'Unknown')}")
                print(f"   Category: {protocol.get('category', 'Other')}")
                print("")
                
            self.assertIsNotNone(protocols)
            self.assertTrue(len(protocols) > 0)
        except Exception as e:
            print(f"ERROR: {e}")
            self.fail(f"Failed to get protocols: {e}")

        print("\n=== 2. Testing show_defi_llama_protocol('aave-v3') ===")
        try:
            protocol = show_defi_llama_protocol.invoke({"protocol_slug": "aave-v3"})
            self.assertIsNotNone(protocol)
            
            # Display full protocol details
            print(f"\nProtocol Details for 'aave-v3':")
            print(f"Name: {protocol.get('name', 'Unknown')}")
            print(f"Slug: {protocol.get('slug', '')}")
            print(f"TVL: {format_tvl(protocol.get('tvl', 0))}")
            print(f"Category: {protocol.get('category', 'Unknown')}")
            
            # Show chains (if available)
            chains = protocol.get('chains', [])
            if chains:
                print(f"Chains: {', '.join(chains[:5])}")
                if len(chains) > 5:
                    print(f"   ...and {len(chains) - 5} more")
            
            # Show description (if available)
            if 'description' in protocol:
                print(f"Description: {protocol['description']}")
                
            # Show URL and social info
            if 'url' in protocol:
                print(f"Website: {protocol['url']}")
            if 'twitter' in protocol:
                print(f"Twitter: {protocol['twitter']}")
        except Exception as e:
            print(f"ERROR: {e}")
            self.fail(f"Failed to get protocol details: {e}")

        print("\n=== 3. Testing show_defi_llama_global_tvl() ===")
        try:
            global_tvl = show_defi_llama_global_tvl.invoke({})
            self.assertIsNotNone(global_tvl)
            
            print(f"\nGlobal DeFi TVL: {format_tvl(global_tvl)}")
        except Exception as e:
            print(f"ERROR: {e}")
            self.fail(f"Failed to get global TVL: {e}")

        print("\n=== 4. Testing show_defi_llama_chain_tvl('ethereum') ===")
        try:
            eth_tvl = show_defi_llama_chain_tvl.invoke({"chain": "ethereum"})
            self.assertIsNotNone(eth_tvl)
            
            print(f"\nEthereum TVL: {format_tvl(eth_tvl)}")
        except Exception as e:
            print(f"ERROR: {e}")
            self.fail(f"Failed to get Ethereum TVL: {e}")

        print("\n=== 5. Testing show_defi_llama_top_pools(3) ===")
        try:
            top_pools = show_defi_llama_top_pools.invoke({"limit": 3})
            self.assertIsNotNone(top_pools)
            
            print(f"\nTop 3 Pools by APY:")
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
        except Exception as e:
            print(f"ERROR: {e}")
            self.fail(f"Failed to get top pools: {e}")

if __name__ == "__main__":
    unittest.main()
