# New file: defi/analytics/maintenance.py
"""Maintenance utilities for DeFi Llama data."""

import json
import os
from typing import List, Dict, Any
from defi.analytics.defillama_source import DefiLlamaMetrics

def export_protocol_slugs(output_file: str = "protocol_slugs.json") -> str:
    """Export all available protocol slugs to a JSON file."""
    try:
        metrics = DefiLlamaMetrics()
        protocols_data = metrics.llama.get_all_protocols()
        slugs = [protocol.get("slug") for protocol in protocols_data if protocol.get("slug")]
        slugs = sorted(slugs)
        
        resources_dir = os.path.join(os.path.dirname(__file__), "..", "resources")
        os.makedirs(resources_dir, exist_ok=True)
        
        output_path = os.path.join(resources_dir, output_file)
        with open(output_path, "w") as f:
            json.dump(slugs, f, indent=2)
        
        return f"Successfully exported {len(slugs)} protocol slugs to {output_file}"
    except Exception as e:
        return f"Error exporting protocol slugs: {str(e)}"

def export_pool_ids(output_file: str = "pool_ids.json") -> str:
    """Export all available pool IDs to a JSON file."""
    try:
        metrics = DefiLlamaMetrics()
        pools_data = metrics.llama.get_pools()
        result = []
        
        if isinstance(pools_data, dict) and "data" in pools_data:
            for pool in pools_data["data"]:
                pool_id = None
                for field in ['id', 'pool', 'identifier']:
                    if field in pool:
                        pool_id = pool[field]
                        break
                
                if not pool_id and 'project' in pool and 'symbol' in pool:
                    pool_id = f"{pool.get('project')}-{pool.get('symbol')}"
                
                if pool_id:
                    result.append({
                        "id": pool_id,
                        "project": pool.get("project", "Unknown"),
                        "symbol": pool.get("symbol", "Unknown"),
                        "chain": pool.get("chain", "Unknown")
                    })
        
        resources_dir = os.path.join(os.path.dirname(__file__), "..", "resources")
        os.makedirs(resources_dir, exist_ok=True)
        
        output_path = os.path.join(resources_dir, output_file)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        return f"Successfully exported {len(result)} pool IDs to {output_file}"
    except Exception as e:
        return f"Error exporting pool IDs: {str(e)}"

def update_all_references():
    """Update all reference files at once."""
    protocol_result = export_protocol_slugs()
    pool_result = export_pool_ids()
    return f"{protocol_result}\n{pool_result}"