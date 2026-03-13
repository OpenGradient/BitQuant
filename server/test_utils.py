import unittest
import re
from typing import Tuple, List


# Inline the extract_patterns regex logic to test without langchain dependency
def extract_patterns(
    text: str, pattern_type: str, remove_pattern=False
) -> Tuple[str, List[str]]:
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


class TestExtractPatterns(unittest.TestCase):
    # --- Token pattern extraction ---

    def test_bare_token_pattern(self):
        text = "Check out token:solana:So11111111111111111111111111111111111111112 for details"
        _, ids = extract_patterns(text, "token")
        self.assertEqual(ids, ["solana:So11111111111111111111111111111111111111112"])

    def test_multiple_tokens(self):
        text = (
            "token:solana:So11111111111111111111111111111111111111112 "
            "and token:ethereum:0xdAC17F958D2ee523a2206206994597C13D831ec7"
        )
        _, ids = extract_patterns(text, "token")
        self.assertEqual(len(ids), 2)
        self.assertIn("solana:So11111111111111111111111111111111111111112", ids)

    def test_no_match(self):
        text = "No tokens here, just regular text"
        _, ids = extract_patterns(text, "token")
        self.assertEqual(ids, [])

    def test_short_address_no_match(self):
        # Address must be 20+ chars
        text = "token:solana:short"
        _, ids = extract_patterns(text, "token")
        self.assertEqual(ids, [])

    # --- Swap pattern extraction ---

    def test_swap_pattern(self):
        text = "Buy via swap:solana:EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        _, ids = extract_patterns(text, "swap")
        self.assertEqual(ids, ["solana:EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"])

    def test_token_not_matched_as_swap(self):
        text = "token:solana:So11111111111111111111111111111111111111112"
        _, ids = extract_patterns(text, "swap")
        self.assertEqual(ids, [])

    # --- Backtick tolerance ---

    def test_single_backtick_tolerance(self):
        text = "`token:solana:So11111111111111111111111111111111111111112`"
        _, ids = extract_patterns(text, "token")
        self.assertEqual(ids, ["solana:So11111111111111111111111111111111111111112"])

    def test_triple_backtick_tolerance(self):
        text = "```token:solana:So11111111111111111111111111111111111111112```"
        _, ids = extract_patterns(text, "token")
        self.assertEqual(ids, ["solana:So11111111111111111111111111111111111111112"])

    # --- Remove pattern ---

    def test_remove_pattern(self):
        text = "Check token:solana:So11111111111111111111111111111111111111112 here"
        cleaned, ids = extract_patterns(text, "token", remove_pattern=True)
        self.assertEqual(ids, ["solana:So11111111111111111111111111111111111111112"])
        self.assertNotIn("token:", cleaned)
        self.assertIn("Check", cleaned)
        self.assertIn("here", cleaned)

    def test_remove_pattern_with_backticks(self):
        text = "Check `token:solana:So11111111111111111111111111111111111111112` here"
        cleaned, ids = extract_patterns(text, "token", remove_pattern=True)
        self.assertEqual(ids, ["solana:So11111111111111111111111111111111111111112"])
        self.assertNotIn("token:", cleaned)

    def test_no_remove_preserves_text(self):
        text = "token:solana:So11111111111111111111111111111111111111112"
        returned_text, ids = extract_patterns(text, "token", remove_pattern=False)
        self.assertEqual(returned_text, text)

    # --- Pool pattern extraction ---

    def test_pool_pattern(self):
        text = "Invest in pool:save_usdc_lending_pool_12345"
        _, ids = extract_patterns(text, "pool")
        # Pool IDs don't follow chain:address format, so this won't match
        # (pool IDs are just plain strings, not chain:address)
        self.assertEqual(ids, [])

    # --- Edge cases ---

    def test_deduplication_not_done_here(self):
        # extract_patterns doesn't deduplicate - that's done in fastapi_server
        text = (
            "token:solana:So11111111111111111111111111111111111111112 "
            "token:solana:So11111111111111111111111111111111111111112"
        )
        _, ids = extract_patterns(text, "token")
        self.assertEqual(len(ids), 2)

    def test_token_at_end_of_text(self):
        text = "Buy token:solana:So11111111111111111111111111111111111111112"
        _, ids = extract_patterns(text, "token")
        self.assertEqual(len(ids), 1)

    def test_token_at_start_of_text(self):
        text = "token:solana:So11111111111111111111111111111111111111112 is trending"
        _, ids = extract_patterns(text, "token")
        self.assertEqual(len(ids), 1)


if __name__ == "__main__":
    unittest.main()
