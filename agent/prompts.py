from typing import List
import jinja2

from api.api_types import WalletTokenHolding, WalletPoolPosition, Message
import os

# Get absolute path of the templates directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
templates_dir = os.path.join(parent_dir, "templates")

env = jinja2.Environment(loader=jinja2.FileSystemLoader(templates_dir))

investor_agent_template = env.get_template("investor_agent.jinja2")
analytics_agent_template = env.get_template("analytics_agent.jinja2")
suggestions_template = env.get_template("suggestions.jinja2")
router_template = env.get_template("router.jinja2")


# We ignore token holdings with a total value of less than $1
MIN_TOKEN_HOLDING_VALUE_USD = 1

# Start filtering out holdings based on MIN_TOKEN_HOLDING_VALUE_USD after this many holdings
NUM_HOLDINGS_CUTOFF = 10


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
        if len(tokens) <= NUM_HOLDINGS_CUTOFF
        or (
            token.total_value_usd is not None
            and token.total_value_usd > MIN_TOKEN_HOLDING_VALUE_USD
        )
    ]

    agent_prompt = investor_agent_template.render(
        tokens=token_metadata or "Wallet not connected",
        poolDeposits=poolDeposits,
    )

    return agent_prompt


def get_suggestions_prompt(
    conversation_history: List[Message],
    tokens: List[WalletTokenHolding],
    tools: str,
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
        if len(tokens) <= NUM_HOLDINGS_CUTOFF
        or (
            token.total_value_usd is not None
            and token.total_value_usd > MIN_TOKEN_HOLDING_VALUE_USD
        )
    ]

    agent_prompt = suggestions_template.render(
        tokens=token_metadata,
        conversation_history=conversation_history,
        tools=tools,
    )

    return agent_prompt


def get_analytics_prompt(
    tokens: List[WalletTokenHolding] = None,
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
        if len(tokens) <= NUM_HOLDINGS_CUTOFF
        or (
            token.total_value_usd is not None
            and token.total_value_usd > MIN_TOKEN_HOLDING_VALUE_USD
        )
    ]

    analytics_agent_prompt = analytics_agent_template.render(
        tokens=token_metadata,
    )

    return analytics_agent_prompt


def get_router_prompt(message_history: List[Message], current_message: str) -> str:
    """Get the router prompt to determine which agent should handle the request."""

    MAX_AGENT_MESSAGE_LENGTH = 400

    # Truncate assistant response to 400 characters, also include the message type
    message_history = [
        {
            "type": message.type,
            "message": (
                message.message[:MAX_AGENT_MESSAGE_LENGTH] + "..."
                if message.type == "assistant"
                and len(message.message) > MAX_AGENT_MESSAGE_LENGTH
                else message.message
            ),
        }
        for message in message_history
    ]

    router_prompt = router_template.render(
        message_history=message_history,
        current_message=current_message,
    )
    return router_prompt
