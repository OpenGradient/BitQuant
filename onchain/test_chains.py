import unittest

from onchain.chains import (
    parse_token_id,
    format_token_id,
    to_coingecko_chain,
    validate_token_id,
    SUPPORTED_CHAINS,
)


class TestParseTokenId(unittest.TestCase):
    def test_basic(self):
        chain, address = parse_token_id(
            "solana:So11111111111111111111111111111111111111112"
        )
        self.assertEqual(chain, "solana")
        self.assertEqual(address, "So11111111111111111111111111111111111111112")

    def test_lowercases_chain(self):
        chain, address = parse_token_id("Solana:abc123")
        self.assertEqual(chain, "solana")

    def test_preserves_address_case(self):
        chain, address = parse_token_id("ethereum:0xAbCdEf1234567890")
        self.assertEqual(address, "0xAbCdEf1234567890")

    def test_address_with_colons(self):
        # If an address somehow contains colons, only the first colon splits
        chain, address = parse_token_id("solana:some:weird:address")
        self.assertEqual(chain, "solana")
        self.assertEqual(address, "some:weird:address")

    def test_missing_colon_raises(self):
        with self.assertRaises(ValueError):
            parse_token_id("solanaAbc123")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            parse_token_id("")


class TestFormatTokenId(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(format_token_id("solana", "abc123"), "solana:abc123")

    def test_roundtrip(self):
        original = "ethereum:0xAbCdEf1234567890"
        chain, address = parse_token_id(original)
        result = format_token_id(chain, address)
        self.assertEqual(result, "ethereum:0xAbCdEf1234567890")


class TestValidateTokenId(unittest.TestCase):
    def test_valid(self):
        self.assertIsNone(validate_token_id("solana:abc123"))

    def test_invalid(self):
        error = validate_token_id("no_colon_here")
        self.assertIsNotNone(error)
        self.assertIn("chain:address", error)


class TestToCoingeckoChain(unittest.TestCase):
    def test_mapped_chains(self):
        self.assertEqual(to_coingecko_chain("ethereum"), "eth")
        self.assertEqual(to_coingecko_chain("sui"), "sui-network")
        self.assertEqual(to_coingecko_chain("polygon"), "polygon_pos")
        self.assertEqual(to_coingecko_chain("avalanche"), "avax")
        self.assertEqual(to_coingecko_chain("bnb"), "bsc")
        self.assertEqual(to_coingecko_chain("dogecoin"), "dogechain")

    def test_unmapped_chains_passthrough(self):
        self.assertEqual(to_coingecko_chain("solana"), "solana")
        self.assertEqual(to_coingecko_chain("arbitrum"), "arbitrum")
        self.assertEqual(to_coingecko_chain("base"), "base")

    def test_case_insensitive(self):
        self.assertEqual(to_coingecko_chain("Ethereum"), "eth")
        self.assertEqual(to_coingecko_chain("POLYGON"), "polygon_pos")

    def test_ethereum_network_alias(self):
        self.assertEqual(to_coingecko_chain("ethereum-network"), "eth")


class TestSupportedChains(unittest.TestCase):
    def test_known_chains_present(self):
        for chain in [
            "solana",
            "ethereum",
            "polygon",
            "bnb",
            "base",
            "arbitrum",
            "sui",
        ]:
            self.assertIn(chain, SUPPORTED_CHAINS)


if __name__ == "__main__":
    unittest.main()
