"""
Example usage of the DefiLlama client.
"""
import json
from typing import Dict, Any

from defi.defillama import DefiLlama


def pretty_print(data: Dict[str, Any]) -> None:
    """
    Pretty print a dictionary with JSON formatting.
    """
    print(json.dumps(data, indent=2))


def main():
    # Initialize the client
    llama = DefiLlama()
    
    # Example 1: Get all protocols
    print("\n=== GETTING ALL PROTOCOLS ===")
    protocols = llama.get_protocols()
    print(f"Found {len(protocols)} protocols")
    if protocols:
        print("First protocol example:")
        pretty_print(protocols[0])
    
    # Example 2: Get pools data
    print("\n=== GETTING POOLS DATA ===")
    pools = llama.get_pools()
    print(f"Found {len(pools.get('data', []))} pools")
    if pools and 'data' in pools and pools['data']:
        print("First pool example:")
        pretty_print(pools['data'][0])
    
    # Example 3: Get specific protocol
    print("\n=== GETTING SPECIFIC PROTOCOL ===")
    protocol = llama.get_protocol("aave-v3")
    print(f"Protocol: {protocol.get('name')}")
    print(f"TVL: ${protocol.get('tvl', 0):,.2f}")
    
    # Example 4: Get token prices
    print("\n=== GETTING TOKEN PRICES ===")
    tokens = ["ethereum", "bitcoin", "solana"]
    prices = llama.get_token_prices(tokens)
    pretty_print(prices)
    
    # Example 5: Get chains
    print("\n=== GETTING CHAINS ===")
    chains = llama.get_chains()
    print(f"Available chains: {', '.join(chains[:10])}...")


if __name__ == "__main__":
    main() 