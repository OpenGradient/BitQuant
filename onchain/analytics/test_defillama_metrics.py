import unittest
from typing import List, Dict, Any
from api.api_types import Chain, Pool, PoolQuery
from onchain.analytics.defillama_tools import (
    show_defi_llama_protocol,
    show_defi_llama_global_tvl,
    show_defi_llama_chain_tvl,
    show_defi_llama_top_pools,
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
            print(
                f"{i}. {protocol.get('name', 'Unknown')} - {format_tvl(protocol.get('tvl', 0))}"
            )
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

        chains = protocol.get("chains", [])
        if chains:
            print(f"Chains: {', '.join(chains[:5])}")
            if len(chains) > 5:
                print(f"   ...and {len(chains) - 5} more")

        if "description" in protocol:
            print(f"Description: {protocol['description']}")

        if "url" in protocol:
            print(f"Website: {protocol['url']}")
        if "twitter" in protocol:
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
            apy = pool.get("apy")
            apy_formatted = f"{float(apy):,.2f}%" if apy is not None else "N/A"

            print(
                f"{i}. {pool.get('project', 'Unknown')} ({pool.get('symbol', 'Unknown')})"
            )
            print(f"   Chain: {pool.get('chain', 'Unknown')}")
            print(f"   APY: {apy_formatted}")
            print(f"   TVL: {format_tvl(pool.get('tvl', 0))}")

            if "ilRisk" in pool:
                print(f"   IL Risk: {pool.get('ilRisk', 'Unknown')}")

            if "stablecoin" in pool:
                stable = "Yes" if pool.get("stablecoin") else "No"
                print(f"   Stablecoin: {stable}")

            print("")


if __name__ == "__main__":
    unittest.main()
