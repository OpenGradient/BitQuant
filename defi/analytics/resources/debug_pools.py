import json
import os

def debug_pool_file():
    """Direct diagnostic of pool_ids.json structure"""
    print("\n==== POOL JSON DEBUG ====\n")
    
    # Find the file
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, "pool_ids.json")
    print(f"Looking for pool file at: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    print(f"File size: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
    
    try:
        # Load the file
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Basic info
        print(f"\nLoaded data type: {type(data)}")
        print(f"Number of entries: {len(data) if isinstance(data, list) else 'N/A'}")
        
        # Structure check
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            print(f"\nFirst item type: {type(first_item)}")
            print(f"First item keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'N/A'}")
            print(f"Sample item: {first_item}")
            
            # Chain field check
            print("\n--- CHAIN FIELD ANALYSIS ---")
            chain_fields = ['chain', 'chainName', 'chain_name', 'network', 'blockchain']
            found_fields = []
            
            for field in chain_fields:
                if field in first_item:
                    found_fields.append(field)
                    print(f"Field '{field}' exists with value: {first_item[field]}")
            
            if not found_fields:
                print("NO CHAIN FIELDS FOUND IN SAMPLE ITEM!")
            
            # Find all unique chain values
            print("\n--- UNIQUE CHAIN VALUES ---")
            chain_counts = {}
            
            for item in data:
                for field in chain_fields:
                    if field in item and item[field]:
                        chain_value = str(item[field]).lower()
                        if chain_value not in chain_counts:
                            chain_counts[chain_value] = 0
                        chain_counts[chain_value] += 1
                        break
            
            for chain, count in sorted(chain_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"Chain '{chain}': {count} pools")
            
            # Check specifically for Arbitrum
            print("\n--- ARBITRUM SEARCH ---")
            arbitrum_count = 0
            arbitrum_pools = []
            
            for item in data:
                is_arbitrum = False
                for field in chain_fields:
                    if field in item and item[field] and 'arbitrum' in str(item[field]).lower():
                        is_arbitrum = True
                        break
                
                if is_arbitrum:
                    arbitrum_count += 1
                    if arbitrum_count <= 3:  # Just print a few examples
                        arbitrum_pools.append(item)
            
            print(f"Total pools with 'arbitrum' in chain field: {arbitrum_count}")
            if arbitrum_count > 0:
                print(f"Sample Arbitrum pools: {arbitrum_pools}")
            else:
                print("NO ARBITRUM POOLS FOUND")
                
            # Search through all text for "arbitrum"
            print("\n--- FULL TEXT SEARCH FOR 'ARBITRUM' ---")
            arbitrary_count = 0
            arbitrum_fields = set()
            
            for item in data:
                item_str = str(item).lower()
                if 'arbitrum' in item_str:
                    arbitrary_count += 1
                    # Find which field contains arbitrum
                    for k, v in item.items():
                        if isinstance(v, str) and 'arbitrum' in v.lower():
                            arbitrum_fields.add(k)
            
            print(f"Total pools with 'arbitrum' anywhere in data: {arbitrary_count}")
            print(f"Fields containing 'arbitrum': {arbitrum_fields}")
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_pool_file() 