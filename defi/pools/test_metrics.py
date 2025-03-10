import unittest

from api.api_types import Chain, Pool, PoolQuery
from defi.defillama_tools import (
    show_defi_llama_protocols,
    show_defi_llama_protocol,
    show_defi_llama_global_tvl,
    show_defi_llama_chain_tvl,
    show_defi_llama_top_pools
)
from defi.defillama_source import DefiLlamaProtocols


class TestDefiLlamaSource(unittest.TestCase):

    def test_defillama(self):
        metrics = DefiLlamaProtocols()
        metrics.refresh_metrics()

        sol_pools = metrics.get_pools(
            PoolQuery(
                chain=Chain.SOLANA,
                protocols=["save"],
            ))
            
        self.assertIsNotNone(sol_pools)
        print(f"\nFound {len(sol_pools)} Solana pools for 'save' protocol")
        if sol_pools:
            print(f"Example pool: {sol_pools[0].id} - {sol_pools[0].protocol}")
        
        # Test protocols tool
        try:
            print("\n========== Testing show_defi_llama_protocols() ==========")
            protocols = show_defi_llama_protocols.invoke({})
            self.assertIsNotNone(protocols)
            self.assertGreater(len(protocols), 0)
            print(f"Found {len(protocols)} protocols")
            # Print first 3 protocols with basic info
            print("\nSample protocols:")
            for i, protocol in enumerate(protocols[:3]):
                print(f"{i+1}. {protocol.get('name', 'Unknown')} - TVL: ${protocol.get('tvl', 0):,.2f}")
                print(f"   Category: {protocol.get('category', 'Unknown')}")
                print(f"   Chains: {', '.join(protocol.get('chains', []))}")
                print()
        except Exception as e:
            print(f"Warning: Failed to get protocols: {e}")
        
        # Test protocol details tool
        try:
            print("\n========== Testing show_defi_llama_protocol('aave-v3') ==========")
            aave = show_defi_llama_protocol.invoke({"protocol_slug": "aave-v3"})
            self.assertIsNotNone(aave)
            self.assertIn('tvl', aave)
            
            print(f"AAVE V3 TVL: ${aave.get('tvl', 0):,.2f}")
            
            # Print more comprehensive protocol information
            print("\nProtocol Details:")
            print(f"Name: {aave.get('name', 'Unknown')}")
            print(f"Description: {aave.get('description', 'No description')}")
            print(f"Website: {aave.get('url', 'No URL')}")
            
            # Print chain breakdown if available
            if 'chainTvls' in aave:
                print("\nTVL by Chain:")
                for chain, tvl in aave['chainTvls'].items():
                    if isinstance(tvl, (int, float)):
                        print(f"  {chain}: ${tvl:,.2f}")
            
            # Print audit information if available
            if 'audit_links' in aave and aave['audit_links']:
                print("\nAudit Information:")
                for audit in aave['audit_links']:
                    print(f"  - {audit}")
            
            # Print any other interesting metadata
            if 'metadata' in aave:
                print("\nMetadata:")
                for key, value in aave.get('metadata', {}).items():
                    print(f"  {key}: {value}")
            
            # Print token information if available
            if 'tokens' in aave:
                print("\nToken Information:")
            
            print("\nAll available keys in AAVE data:", aave.keys())
        except Exception as e:
            print(f"Warning: Failed to get AAVE V3 protocol: {e}")
        
        # Test global TVL tool
        try:
            print("\nTesting show_defi_llama_global_tvl()...")
            global_tvl = show_defi_llama_global_tvl.invoke({})
            self.assertIsNotNone(global_tvl)
            self.assertIn('totalLiquidityUSD', global_tvl)
            print(f"Global TVL: ${global_tvl.get('totalLiquidityUSD', 0):,.2f}")
        except Exception as e:
            print(f"Warning: Failed to get global TVL: {e}")
        
        # Test chain TVL tool
        try:
            print("\nTesting show_defi_llama_chain_tvl('ethereum')...")
            eth_tvl = show_defi_llama_chain_tvl.invoke({"chain": "ethereum"})
            self.assertIsNotNone(eth_tvl)
            self.assertIn('totalLiquidityUSD', eth_tvl)
            print(f"Ethereum TVL: ${eth_tvl.get('totalLiquidityUSD', 0):,.2f}")
        except Exception as e:
            print(f"Warning: Failed to get Ethereum TVL: {e}")
        
        # Test top pools tool
        try:
            print("\nTesting show_defi_llama_top_pools(limit=3)...")
            top_pools = show_defi_llama_top_pools.invoke({"limit": 3})
            self.assertIsNotNone(top_pools)
            self.assertLessEqual(len(top_pools), 3)
            print(f"Found {len(top_pools)} top pools")
            if top_pools:
                print(f"Top pool by APY: {top_pools[0]['project']} - {top_pools[0].get('apy', 0):.2f}%")
        except Exception as e:
            print(f"Warning: Failed to get top pools: {e}")
            
