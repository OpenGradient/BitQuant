from typing import List
import logging

import jinja2

from defi.types import WalletTokenHolding, Pool, WalletPoolPosition

env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates/"))

agent_template = env.get_template("agent.jinja2")
suggestions_template = env.get_template("suggestions.jinja2")
analytics_template = env.get_template("defi_data_scientist.jinja2")


def get_agent_prompt(
    protocol: str,
    tokens: List[WalletTokenHolding],
    poolDeposits: List[WalletPoolPosition],
    availablePools: List[Pool],
) -> str:
    agent_prompt = agent_template.render(
        protocolName=protocol,
        tokens=tokens,
        poolDeposits=poolDeposits,
        availablePools=availablePools,
    )

    return agent_prompt


def get_suggestions_prompt(
    protocol: str,
    tokens: List[WalletTokenHolding],
    poolDeposits: List[WalletPoolPosition],
    availablePools: List[Pool],
) -> str:
    agent_prompt = suggestions_template.render(
        protocolName=protocol,
        tokens=tokens,
        poolDeposits=poolDeposits,
        availablePools=availablePools,
    )

    return agent_prompt


def get_analytics_prompt(message=None, question=None) -> str:
    """
    Returns the prompt template for the analytics agent.
    
    Args:
        message: The user's message (for compatibility)
        question: Alternative parameter name (for compatibility)
    """
    # Use whichever parameter is provided
    query = message or question or "No question provided"
    
    return f"""You are a DeFi analytics expert who helps users understand market data and trends.

AVAILABLE TOOLS:
- get_protocol_insights(protocol_slug): Get detailed data about a specific DeFi protocol
- get_global_tvl(): Get the total value locked across all DeFi protocols globally
- get_chain_tvl(chain): Get the total value locked on a specific blockchain
- compare_pools(limit): Compare top yield-generating pools across protocols
- get_price_history(pair, interval, limit): Get historical price data for a cryptocurrency
- analyze_price_trend(pair, interval, limit): Analyze price trends with technical indicators
- compare_assets(pairs, interval, limit): Compare performance metrics of multiple assets

When using these tools:
1. ALWAYS use tools to answer questions rather than relying on general knowledge
2. For TVL questions, use get_global_tvl() or get_chain_tvl()
3. For protocol-specific questions, use get_protocol_insights()
4. For yield comparisons, use compare_pools()
5. For price analysis, use the Binance tools

USER QUESTION: {query}

Analyze this request carefully and provide detailed insights using the appropriate tools.
"""
