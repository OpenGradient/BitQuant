from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from defi.analytics.defillama_source import DefiLlamaMetrics
import json
import os

defi_metrics = DefiLlamaMetrics()


@tool()
def show_defi_llama_protocols() -> List[Dict[str, Any]]:
    """Get a list of top DeFi protocols by TVL.
    
    Returns:
        List[Dict[str, Any]]: List of protocols with their details.
    """
    return defi_metrics.get_protocols()


@tool()
def show_defi_llama_protocol(protocol_slug: str) -> Dict[str, Any]:
    """Get detailed information about a DeFi protocol.
    
    Args:
        protocol_slug (str): The protocol slug to look up.
        
    Returns:
        Dict[str, Any]: Protocol details including TVL, chains, etc.
    """
    # FORCE validation against known slugs
    json_path = os.path.join(os.path.dirname(__file__), "resources", "protocol_slugs.json")
    with open(json_path, "r") as f:
        all_slugs = json.load(f)
    
    # Check if the slug exists exactly in our known slugs
    if protocol_slug not in all_slugs:
        # Look for versioned alternatives
        base_slug = protocol_slug.split('-v')[0] if '-v' in protocol_slug else protocol_slug
        versions = [s for s in all_slugs if s.startswith(base_slug + '-v')]
        
        if versions:
            # Sort versions to find latest
            versions.sort(key=lambda x: int(x.split('-v')[-1]) if x.split('-v')[-1].isdigit() else 0)
            latest_version = versions[-1]
            
            return {
                "warning": f"Slug '{protocol_slug}' not found in database. Using latest version: '{latest_version}'",
                "using_slug": latest_version,
                "data": defi_metrics.get_protocol(latest_version)
            }
        else:
            return {
                "error": f"Invalid protocol slug: '{protocol_slug}'. This slug does not exist in our database.",
                "valid_slugs_count": len(all_slugs),
                "similar_slugs": [s for s in all_slugs if base_slug in s][:5]
            }
    
    # Now proceed with the valid slug
    result = defi_metrics.get_protocol(protocol_slug)
    
    # ALWAYS include the exact slug used
    result["exact_slug_used"] = protocol_slug
    
    return result


@tool()
def show_defi_llama_global_tvl() -> float:
    """Get the current global TVL (Total Value Locked) across all DeFi.
    
    Returns:
        float: The global TVL value.
    """
    return defi_metrics.get_global_tvl()


@tool()
def show_defi_llama_chain_tvl(chain: str) -> float:
    """Get the TVL for a specific blockchain.
    
    Args:
        chain (str): The name of the blockchain (e.g., "ethereum", "solana").
        
    Returns:
        float: The chain's TVL value.
    """
    return defi_metrics.get_chain_tvl(chain)


@tool()
def show_defi_llama_top_pools(limit: int = 10) -> List[Dict[str, Any]]:
    """Get the top DeFi liquidity pools ranked by APY.
    
    Args:
        limit (int, optional): Maximum number of pools to return. Defaults to 10.
        
    Returns:
        List[Dict[str, Any]]: List of pools with their details.
    """
    return defi_metrics.get_top_pools(limit)


@tool()
def show_defi_llama_pool(pool_query: str = None, pool_id: str = None) -> Dict[str, Any]:
    """Get information about a specific DeFi pool by ID or search query."""
    # EMERGENCY DEBUG - Check Pool Files
    try:
        import os, json
        # Check ALL possible locations of the file
        potential_paths = [
            os.path.join(os.path.dirname(__file__), "resources", "pool_ids.json"),
            os.path.join(os.path.dirname(__file__), "pool_ids.json"),
            "/Users/oliver/Desktop/code/opengradient/TwoLigma/defi/analytics/resources/pool_ids.json"
        ]
        
        for idx, path in enumerate(potential_paths):
            exists = os.path.exists(path)
            print(f"EMERGENCY DEBUG: Path {idx+1} exists: {exists} - {path}")
            
            if exists:
                try:
                    with open(path, "r") as f:
                        all_pools = json.load(f)
                    arbitrum_pools = [p for p in all_pools if p.get("chain") == "Arbitrum"]
                    print(f"EMERGENCY DEBUG: Found {len(arbitrum_pools)} Arbitrum pools in file {path}")
                    
                    # If we found Arbitrum pools and query is about Arbitrum, return them immediately
                    if arbitrum_pools and pool_query and "arbitrum" in pool_query.lower():
                        pool_options = []
                        for p in arbitrum_pools[:5]:
                            pool_options.append({
                                "id": p["id"],
                                "project": p["project"],
                                "symbol": p["symbol"],
                                "chain": p["chain"]
                            })
                        return {"matches": pool_options, "debug": "DIRECT_PATH_MATCH"}
                except Exception as e:
                    print(f"EMERGENCY DEBUG: Error reading file {path}: {str(e)}")
    except Exception as e:
        print(f"EMERGENCY DEBUG: Outer error: {str(e)}")
    
    # Continue with original function...
    from defi.analytics import defi_metrics
    
    if not pool_id and not pool_query:
        return {
            "error": "Either pool_id or pool_query must be provided"
        }
    
    # Direct ID lookup if provided
    if pool_id:
        return defi_metrics.get_pool(pool_id)
    
    # Otherwise search by query
    try:
        # If we have a search query, try to find matching pools
        if pool_query:
            from defi.analytics.resources.lookup_utils import get_pool_id, search_pool_ids
            
            # Try to find a direct match first
            direct_id = get_pool_id(pool_query)
            
            if direct_id:
                # We found a direct match
                return defi_metrics.get_pool(direct_id)
            
            # No direct match, search for options
            matching_pools = search_pool_ids(pool_query, limit=5)
            
            if not matching_pools:
                # EMERGENCY OVERRIDE - Try direct chain matching
                query_lower = pool_query.lower()
                for chain in ["arbitrum", "ethereum", "polygon", "binance"]:
                    if chain in query_lower:
                        print(f"EMERGENCY: Direct chain matching for '{chain}'")
                        # Actually use the chain match to return pools
                        chain_pools = search_pools_by_chain(chain)
                        if chain_pools:
                            return {
                                "matches": [
                                    {
                                        "id": p.get("id", ""),
                                        "project": p.get("project", "Unknown"),
                                        "symbol": p.get("symbol", "Unknown"), 
                                        "chain": p.get("chain", "Unknown")
                                    } for p in chain_pools[:5]
                                ],
                                "message": f"Found pools on {chain}:"
                            }
            elif len(matching_pools) == 1:
                # Only one match, just return it directly
                return defi_metrics.get_pool(matching_pools[0]["id"])
            else:
                # Return the list of matches for user to select
                pool_options = []
                for p in matching_pools:
                    pool_options.append({
                        "id": p["id"],
                        "project": p["project"],
                        "symbol": p["symbol"],
                        "chain": p["chain"]
                    })
                    
                return {
                    "matches": pool_options,
                    "message": f"Found {len(matching_pools)} matching pools. Please specify which one you want."
                }
    except Exception as e:
        return {
            "error": f"Error fetching pool data: {str(e)}",
            "suggestion": "Try providing a more specific query or a direct pool ID"
        }


@tool
def search_defi_resources(query: str, resource_type: str = "protocols", limit: int = 5) -> Dict[str, Any]:
    """Search for DeFi protocols or pools by name.
    
    Args:
        query (str): The search term.
        resource_type (str): Either "protocols" or "pools". Defaults to "protocols".
        limit (int): Maximum number of results. Defaults to 5.
        
    Returns:
        Dict[str, Any]: Matching resources.
    """
    from defi.analytics.resources.lookup_utils import search_protocol_slugs, search_pool_ids
    
    if resource_type.lower() == "protocols":
        matches = search_protocol_slugs(query, limit=limit)
        return {
            "resource_type": "protocols",
            "matches": matches,
            "count": len(matches)
        }
    elif resource_type.lower() == "pools":
        matches = search_pool_ids(query, limit=limit)
        return {
            "resource_type": "pools",
            "matches": matches,
            "count": len(matches)
        }
    else:
        return {
            "error": f"Invalid resource_type: {resource_type}. Use 'protocols' or 'pools'."
        }

def search_pools_by_chain(chain_name, limit=10):
    """Simple function to search pools by chain name directly from JSON"""
    # Load the pools from JSON file directly
    import os
    import json
    
    json_path = os.path.join(os.path.dirname(__file__), "resources/pool_ids.json")
    
    try:
        with open(json_path, "r") as f:
            all_pools = json.load(f)
        
        # Simple case-insensitive matching
        matching_pools = []
        for pool in all_pools:
            if "chain" in pool and pool["chain"].lower().strip() == chain_name.lower().strip():
                matching_pools.append(pool)
                if len(matching_pools) >= limit:
                    break
        
        return matching_pools
    except Exception as e:
        print(f"Error loading pools: {str(e)}")
        return []

@tool()
def show_pools_by_chain(chain_name: str, limit: int = 10) -> Dict[str, Any]:
    """
    Show pools available on a specific blockchain chain.
    
    Args:
        chain_name (str): Name of the blockchain (e.g., "Arbitrum", "Ethereum")
        limit (int): Maximum number of pools to return
        
    Returns:
        Dict[str, Any]: Pool information or error message
    """
    # Normalize chain name
    chain_name = chain_name.lower()
    
    # Get pools for the chain
    pools = search_pools_by_chain(chain_name, limit)
    
    if pools:
        return {
            "success": True,
            "chain": chain_name,
            "pools": [
                {
                    "id": pool["id"],
                    "project": pool["project"],
                    "symbol": pool["symbol"],
                    "chain": pool["chain"]
                } for pool in pools
            ]
        }
    else:
        return {
            "success": False,
            "error": f"No pools found for chain {chain_name}",
            "suggestion": "Try chains like 'ethereum', 'polygon', or 'binance'"
        }
