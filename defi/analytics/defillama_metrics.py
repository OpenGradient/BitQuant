from typing import List, Dict, Any
from functools import lru_cache

from defillama import DefiLlama


class DefiLlamaMetrics:
    """Class for interacting with DefiLlama API to fetch DeFi metrics."""

    llama: DefiLlama

    def __init__(self):
        self.llama = DefiLlama()

    @lru_cache(maxsize=1)
    def get_protocols(self) -> List[Dict[str, Any]]:
        """Retrieve all DeFi protocols from DefiLlama with caching.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing essential protocol information,
            limited to the top 50 protocols by TVL.
        """
        protocols_data = self.llama.get_all_protocols()

        # Filter down to just essential information
        simplified_data = []
        for protocol in protocols_data:
            simplified_data.append(
                {
                    "name": protocol.get("name", "Unknown"),
                    "slug": protocol.get("slug", ""),
                    "tvl": protocol.get("tvl", 0),
                    "chain": protocol.get("chain", "Unknown"),
                    "category": protocol.get("category", "Other"),
                }
            )

        # Ensure no None TVL values before sorting
        for protocol in simplified_data:
            if protocol.get("tvl") is None:
                protocol["tvl"] = 0

        # Now safe to sort
        simplified_data.sort(key=lambda x: x.get("tvl", 0), reverse=True)

        # Return only top 50 protocols to prevent data overload
        return simplified_data[:50]

    def get_protocol(self, protocol_slug: str) -> Dict[str, Any]:
        """Get detailed information for a specific protocol identified by its slug.

        Args:
            protocol_slug (str): The slug of the protocol.

        Returns:
            Dict[str, Any]: A dictionary containing the protocol's details including TVL, description,
            chain, category, and audit info if available.
        """
        protocol_data = self.llama.get_protocol(protocol_slug)

        # If no data returned, get basic TVL info
        if not protocol_data:
            protocol_tvl = self.llama.get_protocol_current_tvl(protocol_slug)
            return {"name": protocol_slug, "tvl": protocol_tvl.get("tvl", 0)}

        # Extract and format TVL data
        if protocol_data and "tvl" in protocol_data:
            if isinstance(protocol_data["tvl"], list):
                last_entry = protocol_data["tvl"][-1] if protocol_data["tvl"] else 0
                if isinstance(last_entry, dict) and "totalLiquidityUSD" in last_entry:
                    protocol_data["tvl"] = last_entry["totalLiquidityUSD"]
                elif isinstance(last_entry, (int, float, str)):
                    protocol_data["tvl"] = float(last_entry)
                else:
                    protocol_data["tvl"] = 0
            elif isinstance(protocol_data["tvl"], dict):
                if "tvl" in protocol_data["tvl"]:
                    protocol_data["tvl"] = protocol_data["tvl"]["tvl"]
                elif "totalLiquidityUSD" in protocol_data["tvl"]:
                    protocol_data["tvl"] = protocol_data["tvl"]["totalLiquidityUSD"]
                else:
                    for k, v in protocol_data["tvl"].items():
                        if (
                            isinstance(v, (int, float, str))
                            and str(v).replace(".", "", 1).isdigit()
                        ):
                            protocol_data["tvl"] = float(v)
                            break
                    else:
                        protocol_data["tvl"] = 0

        # Create response with only essential data
        parsed_protocol_data = {
            "name": protocol_data.get("name", protocol_slug),
            "slug": protocol_data.get("slug", protocol_slug),
            "tvl": protocol_data.get("tvl", 0),
            "description": protocol_data.get("description", "No description available"),
            "chain": protocol_data.get("chain", "Unknown"),
            "category": protocol_data.get("category", "Other"),
            "url": protocol_data.get("url", ""),
            "twitter": protocol_data.get("twitter", ""),
            "chains": protocol_data.get("chains", [])[:10],  # Limit chains to 10
        }

        # If there's audit data, include a simplified version
        if "audit_links" in protocol_data and isinstance(
            protocol_data["audit_links"], list
        ):
            parsed_protocol_data["has_audits"] = len(protocol_data["audit_links"]) > 0

        return parsed_protocol_data

    def get_global_tvl(self) -> float:
        """Calculate the current global Total Value Locked (TVL) across all DeFi protocols.

        Returns:
            float: The global TVL as a float.
        """
        chains_tvl = self.llama.get_chains_current_tvl()

        # Calculate the total TVL across all chains
        total_tvl = sum(float(chain_data.get("tvl", 0)) for chain_data in chains_tvl)

        return total_tvl

    def get_chain_tvl(self, chain: str) -> float:
        """Retrieve the TVL for a specific blockchain.

        Args:
            chain (str): The target blockchain name.

        Returns:
            float: The TVL for the specified chain. Returns 0 if the chain is not found.
        """
        chains_tvl = self.llama.get_chains_current_tvl()

        # Find the specific chain we're looking for
        for chain_data in chains_tvl:
            if chain_data.get("name", "").lower() == chain.lower():
                return float(chain_data.get("tvl", 0))

        # If chain not found, return 0
        return 0

    def get_top_pools(self, chain: str = None, limit: int = 10, min_tvl: float = 500000, max_apy: float = 1000) -> List[Dict[str, Any]]:
        """Obtain the top DeFi pools ranked by Annual Percentage Yield (APY) with configurable TVL threshold.

        Args:
            chain (str, optional): The target blockchain name. If None, returns pools from all chains.
            limit (int, optional): Maximum number of pools to return. Defaults to 10.
            min_tvl (float, optional): Minimum TVL threshold in USD. Defaults to 500000 ($500k).
            max_apy (float, optional): Maximum APY threshold in percentage. Defaults to 1000 (1000%).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with details for each pool.
        """
        pools_data = self.llama.get_pools()
        if isinstance(pools_data, dict) and "data" in pools_data:
            all_pools = pools_data["data"]
            
            # Filter by minimum TVL and maximum APY
            filtered_pools = [
                pool for pool in all_pools
                if float(pool.get("tvlUsd", 0)) >= min_tvl  # Configurable minimum TVL
                and float(pool.get("apy", 0)) <= max_apy  # Maximum APY cap
                and (not chain or pool.get("chain", "").lower() == chain.lower())
            ]
            
            # Sort by APY (highest first)
            sorted_pools = sorted(
                filtered_pools, 
                key=lambda x: float(x.get("apy", 0)), 
                reverse=True
            )

            return sorted_pools[:limit]
        return []

    def get_pool(self, pool_id: str) -> Dict[str, Any]:
        """Fetch a specific DeFi pool's historical data using its unique identifier.

        Args:
            pool_id (str): The unique identifier for the pool.

        Returns:
            Dict[str, Any]: A dictionary containing the pool's historical data if found;
            otherwise, a dictionary with an error message.
        """
        try:
            chart_data = self.llama.get_pool(pool_id)

            result = {
                "status": chart_data.get("status", "unknown"),
                "data": chart_data.get("data", []),
            }

            if result["data"] and len(result["data"]) > 0:
                latest = result["data"][-1]
                result["latest"] = {
                    "tvl": latest.get("tvlUsd", 0),
                    "apy": latest.get("apy", 0),
                    "timestamp": latest.get("timestamp"),
                }

            return result

        except Exception as e:
            return {
                "error": f"Failed to fetch pool with ID '{pool_id}'",
                "details": str(e),
            }

    def get_historical_global_tvl(self, months: int = None, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Get historical TVL data for all DeFi protocols across all chains.
        
        Args:
            months (int, optional): Number of months of history to return. Defaults to None.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to None.
            end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to None.
            
        Returns:
            Dict[str, Any]: A dictionary containing processed historical TVL data points.
        """
        # Get raw historical TVL data
        historical_data = self.llama.get_historical_tvl()
        
        # Process the data with the specified filters
        return self._process_historical_data(historical_data, months, start_date, end_date)

    def get_historical_chain_tvl(self, chain: str, months: int = None, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Get historical TVL data for a specific blockchain.
        
        Args:
            chain (str): The target blockchain name.
            months (int, optional): Number of months of history to return. Defaults to None.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to None.
            end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to None.
            
        Returns:
            Dict[str, Any]: A dictionary containing processed historical TVL data for the chain.
        """
        # Get historical TVL data for the specified chain
        historical_data = self.llama.get_historical_tvl_chain(chain)
        
        # Process the data with the specified filters
        return self._process_historical_data(historical_data, months, start_date, end_date)

    def _process_historical_data(self, historical_data: List[Dict[str, Any]], 
                                months: int = None, start_date: str = None, 
                                end_date: str = None) -> Dict[str, Any]:
        """Process historical data with flexible date filtering options.
        
        Args:
            historical_data (List[Dict[str, Any]]): Raw historical data from DefiLlama API.
            months (int, optional): Number of months to include. Defaults to None.
            start_date (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to None.
            end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to None.
            
        Returns:
            Dict[str, Any]: Processed historical data with formatted dates and TVL values.
        """
        from datetime import datetime, timedelta
        
        # Initialize cutoff timestamps based on provided filters
        cutoff_start_timestamp = None
        cutoff_end_timestamp = None
        
        # Parse date strings if provided
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                cutoff_start_timestamp = int(start_dt.timestamp())
            except ValueError:
                # Handle invalid date format
                return {"error": f"Invalid start_date format: {start_date}. Use YYYY-MM-DD format."}
        
        if end_date:
            try:
                # Set end date to end of day
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
                cutoff_end_timestamp = int(end_dt.timestamp())
            except ValueError:
                # Handle invalid date format
                return {"error": f"Invalid end_date format: {end_date}. Use YYYY-MM-DD format."}
        
        # If months is specified and start_date is not, calculate start date from months
        if months is not None and not start_date:
            cutoff_start_timestamp = int((datetime.now() - timedelta(days=30 * months)).timestamp())
        
        # If no end date is specified, use current time
        if not cutoff_end_timestamp:
            cutoff_end_timestamp = int(datetime.now().timestamp())
        
        # If no start date is specified (and no months), use earliest available data
        # (We'll filter in the processing loop)
        
        # Determine time frame description
        timeframe_desc = "all available data"
        if start_date and end_date:
            timeframe_desc = f"{start_date} to {end_date}"
        elif start_date:
            timeframe_desc = f"from {start_date} to present"
        elif end_date:
            timeframe_desc = f"until {end_date}"
        elif months:
            timeframe_desc = f"last {months} months"
        
        # Initialize the processed data structure
        processed_data = {
            "timeframe": timeframe_desc,
            "summary": {},
            "data_points": []
        }
        
        # Process each data point
        all_tvl_values = []
        
        for entry in historical_data:
            # Handle different formats that might come from the API
            timestamp = entry.get('date') or entry.get('timestamp')
            tvl = entry.get('tvl') or entry.get('totalLiquidityUSD')
            
            if timestamp is not None and tvl is not None:
                # Convert timestamp to int if it's a string
                if isinstance(timestamp, str):
                    timestamp = int(timestamp)
                    
                # Apply date filters
                include_point = True
                if cutoff_start_timestamp and timestamp < cutoff_start_timestamp:
                    include_point = False
                if cutoff_end_timestamp and timestamp > cutoff_end_timestamp:
                    include_point = False
                    
                if include_point:
                    # Convert epoch timestamp to human-readable date
                    date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                    
                    # Format TVL value (convert to billions if large enough)
                    formatted_tvl = tvl
                    if tvl >= 1_000_000_000:  # If TVL is in billions
                        formatted_tvl = f"${tvl / 1_000_000_000:.2f}B"
                    elif tvl >= 1_000_000:  # If TVL is in millions
                        formatted_tvl = f"${tvl / 1_000_000:.2f}M"
                    else:
                        formatted_tvl = f"${tvl:,.2f}"
                    
                    # Add to data points
                    processed_data["data_points"].append({
                        "date": date_str,
                        "timestamp": timestamp,
                        "tvl": tvl,
                        "formatted_tvl": formatted_tvl
                    })
                    
                    all_tvl_values.append(tvl)
        
        # Sort by date ascending
        processed_data["data_points"].sort(key=lambda x: x['timestamp'])
        
        # Calculate summary statistics
        if all_tvl_values:
            processed_data["summary"] = {
                "current_tvl": processed_data["data_points"][-1]["formatted_tvl"] if processed_data["data_points"] else "N/A",
                "min_tvl": f"${min(all_tvl_values) / 1_000_000_000:.2f}B" if any(v >= 1_000_000_000 for v in all_tvl_values) else f"${min(all_tvl_values) / 1_000_000:.2f}M",
                "max_tvl": f"${max(all_tvl_values) / 1_000_000_000:.2f}B" if any(v >= 1_000_000_000 for v in all_tvl_values) else f"${max(all_tvl_values) / 1_000_000:.2f}M",
                "data_points_count": len(processed_data["data_points"]),
                "start_date": processed_data["data_points"][0]["date"] if processed_data["data_points"] else "N/A",
                "end_date": processed_data["data_points"][-1]["date"] if processed_data["data_points"] else "N/A"
            }
        
        return processed_data
