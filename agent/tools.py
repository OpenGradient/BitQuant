from typing import List, Tuple, Dict, Any, Type, Optional

from langgraph.graph.graph import RunnableConfig
from langchain_core.tools import BaseTool, tool

from defi.types import Pool
from agent.defi_agent import DeFiDataScientistAgent

# Create a singleton instance of the DeFi agent
defi_agent = DeFiDataScientistAgent()

@tool(response_format="content_and_artifact")
def show_pools(pool_ids: List[str], config: RunnableConfig) -> Tuple[str, List]:
    """Displays the pools to the user with the given IDs"""
    configurable = config["configurable"]
    available_pools: List[Pool] = configurable["available_pools"]

    pools = [pool.model_dump() for pool in available_pools if pool.id in pool_ids]

    return f"Showing pools to user: {pool_ids}", pools

@tool()
def get_protocol_insights(protocol_slug: str) -> Dict[str, Any]:
    """
    Get comprehensive insights about a specific DeFi protocol
    """
    return defi_agent.get_protocol_insights(protocol_slug)

@tool()
def get_global_tvl() -> Dict[str, Any]:
    """
    Get the current global Total Value Locked (TVL) across all DeFi protocols
    """
    return defi_agent.get_global_tvl()

@tool()
def get_chain_tvl(chain: str) -> Dict[str, Any]:
    """
    Get the Total Value Locked (TVL) for a specific blockchain
    """
    return defi_agent.get_chain_tvl(chain)

@tool()
def compare_pools(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Compare top yield-generating pools across different protocols
    """
    return defi_agent.compare_pools(limit)

# New Binance tools
@tool()
def get_price_history(pair: str = "BTCUSDT", interval: str = "1d", limit: int = 365) -> Dict[str, Any]:
    """
    Retrieve historical price data for a cryptocurrency trading pair
    """
    return defi_agent.get_price_history(pair, interval, limit)

@tool()
def analyze_price_trend(pair: str = "BTCUSDT", interval: str = "1d", limit: int = 30) -> Dict[str, Any]:
    """
    Analyze price trends for a cryptocurrency pair, including trend direction and volatility
    """
    return defi_agent.analyze_price_trend(pair, interval, limit)

@tool()
def compare_assets(pairs: List[str], interval: str = "1d", limit: int = 30) -> Dict[str, Any]:
    """
    Compare performance metrics of multiple cryptocurrency assets
    """
    return defi_agent.compare_assets(pairs, interval, limit)

# Define the tools the agent can use
def create_agent_toolkit() -> List[BaseTool]:
    tools = [
        # DeFiLlama tools 
        show_pools,
        get_protocol_insights,
        get_global_tvl,
        get_chain_tvl,
        compare_pools,
        # Binance tools
        get_price_history,
        analyze_price_trend,
        compare_assets,
    ]

    return tools
