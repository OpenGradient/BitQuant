from typing import List, Tuple, Dict, Any, Type, Optional

from langgraph.graph.graph import RunnableConfig
from langchain_core.tools import BaseTool, tool, StructuredTool

from defi.types import Pool
from defi.stats import DefiMetrics

defi_metrics = DefiMetrics()


@tool(response_format="content_and_artifact")
def show_pools(pool_ids: List[str], config: RunnableConfig) -> Tuple[str, List]:
    """Displays the pools to the user with the given IDs"""
    configurable = config["configurable"]
    available_pools: List[Pool] = configurable["available_pools"]

    pools = [pool.model_dump() for pool in available_pools if pool.id in pool_ids]

    return f"Showing pools to user: {pool_ids}", pools


@tool()
def show_defi_llama_protocols() -> List[Dict[str, Any]]:
    """Show a list of all DeFi protocols"""
    return defi_metrics.get_protocols()


@tool()
def show_defi_llama_protocol(protocol_slug: str) -> Dict[str, Any]:
    """Show details for a specific DeFi protocol by slug"""
    return defi_metrics.get_protocol(protocol_slug)


@tool()
def show_defi_llama_global_tvl() -> Dict[str, Any]:
    """Show current global TVL across all DeFi protocols"""
    return defi_metrics.get_global_tvl()


@tool()
def show_defi_llama_chain_tvl(chain: str) -> Dict[str, Any]:
    """Show TVL for a specific blockchain"""
    return defi_metrics.get_chain_tvl(chain)


@tool()
def show_defi_llama_top_pools(limit: int = 10) -> List[Dict[str, Any]]:
    """Show top DeFi pools ranked by APY"""
    return defi_metrics.get_top_pools(limit)


# Define the tools the agent can use
def create_agent_toolkit() -> List[BaseTool]:
    tools = [
        show_pools,
        show_defi_llama_protocols,
        show_defi_llama_protocol,
        show_defi_llama_global_tvl,
        show_defi_llama_chain_tvl,
        show_defi_llama_top_pools
    ]

    return tools
