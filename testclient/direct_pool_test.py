import json
import os
import sys

def main():
    """Direct pool finder that bypasses all agent code"""
    print("\n=== DIRECT POOL FINDER ===\n")
    
    # Get command line arguments
    if len(sys.argv) > 1:
        chain = sys.argv[1]
    else:
        chain = input("Enter chain name (Arbitrum, Ethereum, etc.): ")
    
    chain = chain.lower()
    
    # Map to standard chain names
    if chain in ["arbitrum", "arb"]:
        target_chain = "Arbitrum"
    elif chain in ["ethereum", "eth"]:
        target_chain = "Ethereum"
    elif chain in ["polygon", "matic"]:
        target_chain = "Polygon"
    else:
        target_chain = chain.title()
    
    print(f"Looking for pools on chain: {target_chain}")
    
    # Load the pool data directly
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        json_path = os.path.join(project_root, "defi", "analytics", "resources", "pool_ids.json")
        
        print(f"Loading pools from: {json_path}")
        print(f"File exists: {os.path.exists(json_path)}")
        
        with open(json_path, "r") as f:
            all_pools = json.load(f)
        
        print(f"Loaded {len(all_pools)} total pools")
        
        # Find pools for this chain
        chain_pools = []
        for pool in all_pools:
            if 'chain' in pool and pool['chain'] == target_chain:
                chain_pools.append(pool)
        
        print(f"\nFOUND {len(chain_pools)} POOLS FOR {target_chain}")
        
        # Show the first 5
        for i, pool in enumerate(chain_pools[:5]):
            print(f"\n--- POOL {i+1} ---")
            for key, value in pool.items():
                print(f"{key}: {value}")
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 