from typing import Tuple, List
import re


def extract_patterns(text: str, pattern_type: str) -> Tuple[str, List[str]]:
    """
    Extract patterns of the form ```pattern_type:ID``` from text and return cleaned text and extracted IDs.

    Args:
        text: The text to extract patterns from
        pattern_type: The type of pattern to extract (e.g. 'pool', 'token')

    Returns:
        Tuple containing (cleaned_text, extracted_ids)
    """
    pattern_ids = []

    def extract_id(match):
        pattern_ids.append(match.group(1))
        return ""

    # Find all occurrences of ```pattern_type:ID``` patterns
    pattern = f"```{pattern_type}:([^`]+)```"
    cleaned_text = re.sub(pattern, extract_id, text)

    return cleaned_text, pattern_ids
