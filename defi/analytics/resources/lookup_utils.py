import json
import os
import re
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from difflib import get_close_matches

# Set up logging instead of print statements
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lookup_utils")

# Global cache of pools by chain - loaded once
POOLS_BY_CHAIN = {}
ALL_POOLS = []
CHAIN_MAP = {
    # Add common variations of chain names
    "arbitrum": ["arbitrum", "arbitrum one", "arb", "arb one"],
    "ethereum": ["ethereum", "eth", "ether", "mainnet"],
    "polygon": ["polygon", "matic", "polygon pos"],
    "binance": ["binance", "bnb", "bsc", "binance smart chain"],
    "optimism": ["optimism", "op"],
    "avalanche": ["avalanche", "avax"],
    "base": ["base"],
    "solana": ["solana", "sol"]
}

def load_pool_data():
    """Load pool data once and index by chain for fast lookup."""
    global POOLS_BY_CHAIN, ALL_POOLS
    
    if ALL_POOLS:  # Already loaded
        return
        
    try:
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pool_ids.json")
        print(f"Loading pools from: {json_path} (exists: {os.path.exists(json_path)})")
        
        with open(json_path, "r") as f:
            ALL_POOLS = json.load(f)
            
        print(f"Loaded {len(ALL_POOLS)} pools, building chain indexes...")
        
        # Check the first pool to understand structure
        if ALL_POOLS and len(ALL_POOLS) > 0:
            first_pool = ALL_POOLS[0]
            print(f"First pool keys: {list(first_pool.keys())}")
        
        # Index pools by normalized chain name
        for pool in ALL_POOLS:
            # Try different chain fields
            chain_value = None
            for field in ['chain', 'chainName', 'chain_name', 'network', 'blockchain']:
                if field in pool and pool[field]:
                    chain_value = str(pool[field]).lower()
                    break
            
            if not chain_value:
                continue
                
            # Map to standard chain name
            standard_chain = None
            for std_name, variations in CHAIN_MAP.items():
                if chain_value in variations:
                    standard_chain = std_name
                    break
            
            # Use the original if no mapping found
            if not standard_chain:
                standard_chain = chain_value
                
            # Add to index
            if standard_chain not in POOLS_BY_CHAIN:
                POOLS_BY_CHAIN[standard_chain] = []
            POOLS_BY_CHAIN[standard_chain].append(pool)
        
        # Report on what we found
        for chain, pools in POOLS_BY_CHAIN.items():
            print(f"Chain '{chain}': {len(pools)} pools")
            
    except Exception as e:
        print(f"ERROR loading pool data: {str(e)}")
        import traceback
        traceback.print_exc()

# Load the data at module import time
load_pool_data()

def get_resource_path(filename: str) -> str:
    """Get the full path to a resource file."""
    # FIX: The resource files are directly in the resources directory,
    # not in resources/resources as the original path implied
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

def search_resource(query: str, resource_file: str, key_field: str = None, 
                  match_fields: List[str] = None, limit: int = 5) -> List[Union[str, Dict]]:
    """Generic search function for resource files with improved token matching."""
    filepath = get_resource_path(resource_file)
    if not os.path.exists(filepath):
        logger.error(f"Resource file not found: {filepath}")
        return []
    
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.debug(f"Loaded {len(data) if isinstance(data, list) else 'object'} from {resource_file}")
    except Exception as e:
        logger.error(f"Error loading resource {resource_file}: {str(e)}")
        return []
    
    # Normalize query - replace periods and hyphens with spaces
    query = query.lower().strip()
    # Normalize by replacing special chars with spaces
    normalized_query = re.sub(r'[.\-_]', ' ', query)
    query_tokens = set(normalized_query.split())
    
    scored_matches = []
    
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
        # String list (like protocol slugs)
        
        # 1. Direct match first
        exact_matches = [item for item in data if item.lower() == query]
        if exact_matches:
            return exact_matches
        
        # 2. Token-level matching
        for item in data:
            # Normalize item the same way we normalized the query
            normalized_item = re.sub(r'[.\-_]', ' ', item.lower())
            item_tokens = set(normalized_item.split())
            
            # Calculate token overlap metrics
            common_tokens = query_tokens.intersection(item_tokens)
            
            if common_tokens:
                # Calculate different scoring metrics
                precision = len(common_tokens) / len(query_tokens)  # % of query tokens found
                recall = len(common_tokens) / len(item_tokens)      # % of item tokens matched
                
                # Higher score for items that contain ALL query tokens
                if len(common_tokens) == len(query_tokens):
                    score = 1.0 + precision  # Boost for containing all tokens
                else:
                    score = precision * 0.8  # Lower score for partial matches
                
                scored_matches.append((item, score))
        
        # Sort by score and return top matches
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in scored_matches[:limit]]
    
    else:
        # Dict list (like pool data)
        if not match_fields:
            logger.error("match_fields must be provided for dictionary data")
            return []
            
        for item in data:
            max_score = 0
            
            for field in match_fields:
                if field not in item:
                    continue
                    
                field_value = str(item[field]).lower()
                field_tokens = set(field_value.replace('-', ' ').split())
                
                # Calculate token overlap
                common_tokens = query_tokens.intersection(field_tokens)
                
                if common_tokens:
                    # Same scoring as for strings
                    precision = len(common_tokens) / len(query_tokens)
                    recall = len(common_tokens) / len(field_tokens)
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    score = f1_score
                    if len(common_tokens) == len(query_tokens):
                        score = 1.0 + (1.0 - recall)
                        
                    max_score = max(max_score, score)
            
            if max_score > 0:
                scored_matches.append((item, max_score))
        
        # Sort by score and return top matches
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in scored_matches[:limit]]

def search_protocol_slugs(query: str, limit: int = 5) -> List[str]:
    """
    Simple protocol slug search that handles basic substring matching and exact matches.
    
    Args:
        query (str): The user query string
        limit (int): Maximum number of results to return
        
    Returns:
        List[str]: Matching protocol slugs
    """
    # FIX: Use the consistent get_resource_path function
    try:
        json_path = get_resource_path("protocol_slugs.json")
        with open(json_path, "r") as f:
            all_slugs = json.load(f)
        logger.debug(f"Loaded {len(all_slugs)} protocol slugs")
    except Exception as e:
        logger.error(f"Error loading protocol slugs: {str(e)}")
        return []
    
    # Normalize query
    query = query.lower().strip()
    
    # 1. Try exact match first
    exact_matches = [s for s in all_slugs if s.lower() == query]
    if exact_matches:
        return exact_matches
    
    # 2. Try substring matching (protocol contains the exact query as a substring)
    query_no_spaces = query.replace(" ", "")
    substring_matches = []
    
    for slug in all_slugs:
        slug_lower = slug.lower()
        # Check if query is in the slug directly
        if query in slug_lower:
            substring_matches.append(slug)
            continue
            
        # Check if query with spaces removed is in slug with special chars removed
        slug_no_special = re.sub(r'[.\-_]', '', slug_lower)
        if query_no_spaces in slug_no_special:
            substring_matches.append(slug)
            continue
            
        # Check if query with spaces becomes hyphens in slug
        query_as_hyphens = query.replace(" ", "-")
        if query_as_hyphens in slug_lower:
            substring_matches.append(slug)
            continue
    
    if substring_matches:
        return substring_matches[:limit]
    
    # 3. Try word-level matching for multi-word queries
    if " " in query:
        query_words = query.split()
        word_matches = []
        
        for slug in all_slugs:
            slug_lower = slug.lower()
            slug_words = re.sub(r'[.\-_]', ' ', slug_lower).split()
            
            # Count how many query words appear in the slug
            matching_words = sum(1 for word in query_words if word in slug_words)
            
            # If all query words are in the slug, consider it a match
            if matching_words == len(query_words):
                word_matches.append(slug)
        
        if word_matches:
            return word_matches[:limit]
    
    # 4. Try fuzzy matching as last resort
    return get_close_matches(query, all_slugs, n=limit, cutoff=0.6)

def search_pool_ids(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Case-insensitive pool search that will actually find Arbitrum pools
    """
    print(f"\n==== POOL SEARCH: '{query}' ====")
    
    try:
        # Load the pool data
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pool_ids.json")
        print(f"Loading from: {json_path}")
        
        with open(json_path, "r") as f:
            all_pools = json.load(f)
        
        print(f"Loaded {len(all_pools)} total pools")
        
        # Find which chain is mentioned in the query
        query_lower = query.lower()
        
        # Determine target chain
        target_chain_lower = None
        if "arbitrum" in query_lower:
            target_chain_lower = "arbitrum"
        elif "ethereum" in query_lower or "eth" in query_lower:
            target_chain_lower = "ethereum" 
        elif "polygon" in query_lower or "matic" in query_lower:
            target_chain_lower = "polygon"
        # Add other chains as needed
        
        if not target_chain_lower:
            print(f"No recognized chain in query")
            return []
            
        print(f"Looking for pools with chain.lower() == '{target_chain_lower}'")
        
        # DEBUG: Look at some sample chains to verify format
        chain_samples = set()
        for i, pool in enumerate(all_pools):
            if 'chain' in pool and i < 100:  # Just check first 100 pools
                chain_samples.add(pool['chain'])
        print(f"Sample chains in data: {sorted(chain_samples)}")
        
        # Find pools using CASE-INSENSITIVE matching
        chain_matches = []
        for pool in all_pools:
            if 'chain' in pool and pool['chain'].lower() == target_chain_lower:
                if len(chain_matches) < 3:  # Print first few matches for debug
                    print(f"MATCH: {pool.get('name', 'unnamed')} - {pool.get('id', 'no-id')} ({pool.get('chain', 'no-chain')})")
                chain_matches.append(pool)
        
        print(f"Found {len(chain_matches)} pools for {target_chain_lower}")
        
        return chain_matches[:limit]
        
    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return []

def extract_chains_from_query(query: str, known_chains: Set[str]) -> Set[str]:
    """Extract chain names from a query string."""
    # First try multi-word chains (exact matches)
    extracted_chains = set()
    
    # Try to match multi-word chains (up to 3 words)
    words = query.split()
    for i in range(len(words)):
        # Try single words
        if words[i].lower() in known_chains:
            extracted_chains.add(words[i].lower())
        
        # Try two-word chains
        if i < len(words) - 1:
            two_word = f"{words[i]} {words[i+1]}".lower()
            if two_word in known_chains:
                extracted_chains.add(two_word)
        
        # Try three-word chains
        if i < len(words) - 2:
            three_word = f"{words[i]} {words[i+1]} {words[i+2]}".lower()
            if three_word in known_chains:
                extracted_chains.add(three_word)
    
    # If we found exact chain matches, return them
    if extracted_chains:
        return extracted_chains
    
    # Otherwise, try using regex patterns to extract potential chains
    chain_patterns = [
        r'(?:on|in|from|at|using|via|for|with)\s+(\w+(?:\s+\w+){0,2})',  # "on Arbitrum"
        r'(\w+(?:\s+\w+){0,2})(?:\s+chain)',                            # "Arbitrum chain"
        r'(\w+(?:\s+\w+){0,2})(?:\s+pools)',                            # "Arbitrum pools"
        r'(\w+(?:\s+\w+){0,2})(?:\s+network)',                          # "Arbitrum network"
        r'show\s+(?:me\s+)?(\w+(?:\s+\w+){0,2})',                       # "show me Arbitrum"
        r'(\w+(?:\s+\w+){0,2})(?:\s+(?:pools|projects))',               # "Arbitrum pools/projects"
        r'available\s+(?:on|in)\s+(\w+(?:\s+\w+){0,2})',                # "available on Arbitrum"
    ]
    
    potential_chains = set()
    for pattern in chain_patterns:
        matches = re.findall(pattern, query)
        for match in matches:
            # Clean up the match and check if it's a known chain
            clean_match = match.lower().strip()
            if clean_match in known_chains:
                potential_chains.add(clean_match)
    
    return potential_chains

def deduplicate_pools(pools: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """
    Remove duplicate pools while preserving order.
    
    Args:
        pools (List[Dict[str, Any]]): List of pool dictionaries
        limit (int): Maximum number of results to return
    """
    unique_pools = []
    seen_ids = set()
    
    for pool in pools:
        if "id" not in pool:
            continue
            
        if pool["id"] not in seen_ids:
            seen_ids.add(pool["id"])
            unique_pools.append(pool)
            
        if len(unique_pools) >= limit:
            break
            
    return unique_pools

def get_protocol_slug(query: str) -> Optional[str]:
    """Find the single best matching protocol slug.
    
    Args:
        query (str): The search term.
        
    Returns:
        Optional[str]: Best matching protocol slug or None.
    """
    matches = search_protocol_slugs(query, limit=1)
    return matches[0] if matches else None

def get_pool_id(query: str) -> Optional[str]:
    """Find the single best matching pool ID."""
    matches = search_pool_ids(query, limit=1)
    return matches[0]["id"] if matches else None
