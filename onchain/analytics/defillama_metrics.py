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
        """Get all DeFi protocols from DefiLlama, limited to top 50 by TVL"""
        protocols_data = self.llama.get_all_protocols()

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

        for protocol in simplified_data:
            if protocol.get("tvl") is None:
                protocol["tvl"] = 0

        simplified_data.sort(key=lambda x: x.get("tvl", 0), reverse=True)
        return simplified_data[:50]

    def get_protocol(self, protocol_slug: str) -> Dict[str, Any]:
        """Get protocol details by slug"""
        protocol_data = self.llama.get_protocol(protocol_slug)

        if not protocol_data:
            protocol_tvl = self.llama.get_protocol_current_tvl(protocol_slug)
            return {"name": protocol_slug, "tvl": protocol_tvl.get("tvl", 0)}

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

        parsed_protocol_data = {
            "name": protocol_data.get("name", protocol_slug),
            "slug": protocol_data.get("slug", protocol_slug),
            "tvl": protocol_data.get("tvl", 0),
            "description": protocol_data.get("description", "No description available"),
            "chain": protocol_data.get("chain", "Unknown"),
            "category": protocol_data.get("category", "Other"),
            "url": protocol_data.get("url", ""),
            "twitter": protocol_data.get("twitter", ""),
            "chains": protocol_data.get("chains", [])[:10],
        }

        if "audit_links" in protocol_data and isinstance(
            protocol_data["audit_links"], list
        ):
            parsed_protocol_data["has_audits"] = len(protocol_data["audit_links"]) > 0

        return parsed_protocol_data

    def get_global_tvl(self) -> float:
        """Get global TVL across all DeFi protocols"""
        chains_tvl = self.llama.get_chains_current_tvl()
        total_tvl = sum(float(chain_data.get("tvl", 0)) for chain_data in chains_tvl)
        return total_tvl

    def get_chain_tvl(self, chain: str) -> float:
        """Get TVL for a specific blockchain"""
        chains_tvl = self.llama.get_chains_current_tvl()

        for chain_data in chains_tvl:
            if chain_data.get("name", "").lower() == chain.lower():
                return float(chain_data.get("tvl", 0))

        return 0

    def get_top_pools(
        self,
        chain: str = None,
        limit: int = 10,
        min_tvl: float = 500000,
        max_apy: float = 1000,
    ) -> List[Dict[str, Any]]:
        """Get top DeFi pools ranked by APY with filters"""
        pools_data = self.llama.get_pools()
        if isinstance(pools_data, dict) and "data" in pools_data:
            all_pools = pools_data["data"]

            filtered_pools = [
                pool
                for pool in all_pools
                if float(pool.get("tvlUsd", 0)) >= min_tvl
                and float(pool.get("apy", 0)) <= max_apy
                and (not chain or pool.get("chain", "").lower() == chain.lower())
            ]

            sorted_pools = sorted(
                filtered_pools, key=lambda x: float(x.get("apy", 0)), reverse=True
            )

            return sorted_pools[:limit]
        return []

    def get_pool(self, pool_id: str) -> Dict[str, Any]:
        """Get pool data by ID"""
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

    def get_historical_global_tvl(self, num_months: int = 3) -> Dict[str, Any]:
        """Get historical TVL data for all protocols"""
        historical_data = self.llama.get_historical_tvl()
        return self._process_historical_data(historical_data, num_months)

    def get_historical_chain_tvl(
        self, chain: str, num_months: int = 3
    ) -> Dict[str, Any]:
        """Get historical TVL data for a specific blockchain"""
        historical_data = self.llama.get_historical_tvl_chain(chain)
        return self._process_historical_data(historical_data, num_months)

    def _process_historical_data(
        self, historical_data: List[Dict[str, Any]], num_months: int = 3
    ) -> Dict[str, Any]:
        """Process historical data with month filtering"""
        from datetime import datetime, timedelta

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=30 * num_months)
        cutoff_start_timestamp = int(start_dt.timestamp())
        cutoff_end_timestamp = int(end_dt.timestamp())

        processed_data = {
            "timeframe": f"last {num_months} months",
            "summary": {},
            "data_points": [],
        }

        all_tvl_values = []

        for entry in historical_data:
            timestamp = entry.get("date") or entry.get("timestamp")
            tvl = entry.get("tvl") or entry.get("totalLiquidityUSD")

            if timestamp is not None and tvl is not None:
                if isinstance(timestamp, str):
                    timestamp = int(timestamp)

                if (
                    timestamp >= cutoff_start_timestamp
                    and timestamp <= cutoff_end_timestamp
                ):
                    date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

                    formatted_tvl = tvl
                    if tvl >= 1_000_000_000:
                        formatted_tvl = f"${tvl / 1_000_000_000:.2f}B"
                    elif tvl >= 1_000_000:
                        formatted_tvl = f"${tvl / 1_000_000:.2f}M"
                    else:
                        formatted_tvl = f"${tvl:,.2f}"

                    processed_data["data_points"].append(
                        {
                            "date": date_str,
                            "timestamp": timestamp,
                            "tvl": tvl,
                            "formatted_tvl": formatted_tvl,
                        }
                    )

                    all_tvl_values.append(tvl)

        processed_data["data_points"].sort(key=lambda x: x["timestamp"])

        if all_tvl_values:
            processed_data["summary"] = {
                "current_tvl": (
                    processed_data["data_points"][-1]["formatted_tvl"]
                    if processed_data["data_points"]
                    else "N/A"
                ),
                "min_tvl": (
                    f"${min(all_tvl_values) / 1_000_000_000:.2f}B"
                    if any(v >= 1_000_000_000 for v in all_tvl_values)
                    else f"${min(all_tvl_values) / 1_000_000:.2f}M"
                ),
                "max_tvl": (
                    f"${max(all_tvl_values) / 1_000_000_000:.2f}B"
                    if any(v >= 1_000_000_000 for v in all_tvl_values)
                    else f"${max(all_tvl_values) / 1_000_000:.2f}M"
                ),
                "data_points_count": len(processed_data["data_points"]),
                "start_date": (
                    processed_data["data_points"][0]["date"]
                    if processed_data["data_points"]
                    else "N/A"
                ),
                "end_date": (
                    processed_data["data_points"][-1]["date"]
                    if processed_data["data_points"]
                    else "N/A"
                ),
            }

        return processed_data
