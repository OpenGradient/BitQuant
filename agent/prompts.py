from typing import List
import jinja2

from api.api_types import WalletTokenHolding, WalletPoolPosition, Message

env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates/"))

investor_agent_template = env.get_template("investor_agent.jinja2")
analytics_agent_template = env.get_template("analytics_agent.jinja2")
suggestions_template = env.get_template("suggestions.jinja2")
router_template = env.get_template("router.jinja2")


def get_investor_agent_prompt(
    tokens: List[WalletTokenHolding],
    poolDeposits: List[WalletPoolPosition],
) -> str:
    # Only include fields that are needed for the prompt
    token_metadata = [
        {
            "address": token.address,
            "symbol": token.symbol,
            "name": token.name,
            "amount": token.amount,
        }
        for token in tokens
    ]

    agent_prompt = investor_agent_template.render(
        tokens=token_metadata or "Wallet not connected",
        poolDeposits=poolDeposits,
    )

    return agent_prompt


def get_suggestions_prompt(
    tokens: List[WalletTokenHolding],
    tools: str,
) -> str:
    agent_prompt = suggestions_template.render(
        tokens=tokens,
        tools=tools,
    )

    return agent_prompt


def get_analytics_prompt(
    protocol: str,
    tokens: List[WalletTokenHolding] = None,
    poolDeposits: List[WalletPoolPosition] = None,
) -> str:
    analytics_agent_prompt = analytics_agent_template.render(
        protocolName=protocol,
        tokens=tokens,
        poolDeposits=poolDeposits,
    )

    return analytics_agent_prompt


def get_router_prompt(message_history: List[Message], current_message: str) -> str:
    """Get the router prompt to determine which agent should handle the request."""
    router_prompt = router_template.render(
        message_history=message_history,
        current_message=current_message,
    )
    return router_prompt
