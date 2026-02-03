from typing import Tuple, List
import re
import json

from api.api_types import Message, UserMessage, AgentMessage


def convert_to_agent_msg(
    message: Message, truncate=False, max_length=400
) -> Tuple[str, str]:
    if isinstance(message, UserMessage):
        return ("user", message.message)
    elif isinstance(message, AgentMessage):
        if truncate and len(message.message) > max_length:
            message_to_return = message.message[:max_length] + "... [truncated]"
        else:
            message_to_return = message.message

        # if len(message.pools) > 0:
        #     message_to_return += "\n"
        #     for pool in message.pools:
        #         message_to_return += f"```pool:{pool.id}```\n"

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

        return ("assistant", message_to_return)


def extract_patterns(
    text: str, pattern_type: str, remove_pattern=False
) -> Tuple[str, List[str]]:
    """
    Extract patterns of the form ```pattern_type:ID``` from text and return original text and extracted IDs.

    Args:
        text: The text to extract patterns from
        pattern_type: The type of pattern to extract (e.g. 'pool', 'token')
        remove_pattern: If True, remove the pattern markers from the text

    Returns:
        Tuple containing (processed_text, extracted_ids)
    """
    pattern_ids = []

    # Find all occurrences of ```pattern_type:ID``` patterns
    pattern = f"```{pattern_type}:([^`]+)```"
    matches = re.finditer(pattern, text)

    for match in matches:
        pattern_ids.append(match.group(1))

    if remove_pattern:
        # Remove all pattern markers from the text
        cleaned_text = re.sub(pattern, "", text)
        return cleaned_text, pattern_ids
    else:
        return text, pattern_ids
