"""
Test to verify search_token function does not shadow the input parameter.
"""
import unittest
from unittest.mock import AsyncMock, MagicMock


class TestSearchToken(unittest.IsolatedAsyncioTestCase):
    async def test_search_token_no_variable_shadowing(self):
        """
        Ensure search_token uses 'result' variable instead of shadowing
        the 'token' string parameter with a TokenMetadata object.
        """
        # Mock the token metadata repo
        mock_repo = MagicMock()
        mock_token = MagicMock()
        mock_token.chain = "solana"
        mock_token.address = "So111111111"
        mock_token.name = "Wrapped SOL"
        mock_token.symbol = "SOL"
        mock_token.price = 150.0
        mock_repo.search_token = AsyncMock(return_value=mock_token)

        # Import and create the toolkit
        from agent.tools import create_analytics_agent_toolkit
        toolkit = create_analytics_agent_toolkit(mock_repo)

        # Find the search_token tool
        search_tool = next(t for t in toolkit if t.name == "search_token")

        # Call with a string query
        result = await search_tool.ainvoke({"token": "SOL", "chain": "solana"})

        # Verify result is correct
        self.assertEqual(result["symbol"], "SOL")
        self.assertEqual(result["name"], "Wrapped SOL")
        self.assertEqual(result["chain"], "solana")

    async def test_search_token_not_found(self):
        """
        Ensure search_token returns correct message when token not found.
        """
        mock_repo = MagicMock()
        mock_repo.search_token = AsyncMock(return_value=None)

        from agent.tools import create_analytics_agent_toolkit
        toolkit = create_analytics_agent_toolkit(mock_repo)

        search_tool = next(t for t in toolkit if t.name == "search_token")
        result = await search_tool.ainvoke({"token": "UNKNOWN"})

        self.assertEqual(result, "No token found.")


if __name__ == "__main__":
    unittest.main()
