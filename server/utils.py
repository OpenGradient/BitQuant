from typing import Tuple, List
import re
import json

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from api.api_types import Message, UserMessage, AgentMessage


def convert_to_agent_msg(
    message: Message, truncate=False, max_length=800
) -> BaseMessage:
    if isinstance(message, UserMessage):
        return HumanMessage(content=message.message)
    elif isinstance(message, AgentMessage):
        if truncate and len(message.message) > max_length:
            message_to_return = message.message[:max_length] + "... [truncated]"
        else:
            message_to_return = message.message

        if len(message.tokens) > 0:
            message_to_return += "\nTokens:\n"
            token_strings = []
            for token in message.tokens:
                token_dict = {
                    "id": f"{token.chain}:{token.address}",
                    "address": token.address,
                    "name": token.name,
                    "symbol": token.symbol,
                    "chain": token.chain,
                    "price_usd": token.price_usd,
                }
                token_strings.append(json.dumps(token_dict))
            message_to_return += "\n- ".join(token_strings)

        return AIMessage(content=message_to_return)


def extract_patterns(
    text: str, pattern_type: str, remove_pattern=False
) -> Tuple[str, List[str]]:
    """
    Extract patterns of the form pattern_type:chain:address from text.

    Matches bare token:chain:address patterns as well as backtick-wrapped variants.

    Args:
        text: The text to extract patterns from
        pattern_type: The type of pattern to extract (e.g. 'token', 'swap')
        remove_pattern: If True, remove the pattern markers from the text

    Returns:
        Tuple containing (processed_text, extracted_ids)
    """
    # Match pattern_type:chain:address — with or without backticks
    # chain is lowercase letters, address is alphanumeric 20+ chars
    pattern = f"(?:`{{1,3}})?{pattern_type}:([a-zA-Z]+:[a-zA-Z0-9]{{20,}})(?:`{{1,3}})?"
    matches = re.finditer(pattern, text)

    pattern_ids = []
    for match in matches:
        pattern_ids.append(match.group(1))

    if remove_pattern:
        cleaned_text = re.sub(pattern, "", text)
        return cleaned_text, pattern_ids
    else:
        return text, pattern_ids
