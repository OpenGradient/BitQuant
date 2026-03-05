import logging
import os
from typing import List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

OKX_MCP_URL = "https://web3.okx.com/api/v1/onchainos-mcp"

# Allowlist of OKX MCP tools to expose (read-only market data only)
ALLOWED_TOOLS = {
    # Market prices & charts
    "dex-okx-market-price",
    "dex-okx-market-candlesticks",
    "dex-okx-market-candlesticks-history",
    "dex-okx-market-trades",
    "dex-okx-market-price-chains",
    # Token discovery & analytics
    "dex-okx-market-token-search",
    "dex-okx-market-token-price-info",
    "dex-okx-market-token-basic-info",
    "dex-okx-market-token-ranking",
    "dex-okx-market-token-holder",
    # Smart money signals
    "dex-okx-market-signal-list",
    "dex-okx-market-signal-supported-chains",
    # Index prices
    "dex-okx-index-current-price",
    "dex-okx-index-historical-price",
    # Meme token analytics (read-only)
    "dex-okx-market-memepump-token-details",
    "dex-okx-market-memepump-token-list",
    "dex-okx-market-memepump-token-bundle-info",
    "dex-okx-market-memepump-token-dev-info",
    "dex-okx-market-memepump-similar-token",
    "dex-okx-market-memepump-aped-wallet",
    "dex-okx-market-memepump-supported-chainsprotocol",
    # Balance / portfolio (read-only)
    "dex-okx-balance-chains",
    "dex-okx-balance-total-token-balances",
    "dex-okx-balance-total-value",
    "dex-okx-balance-specific-token-balance",
}


class OKXMCPClient:
    """Manages the OKX MCP connection and exposes market data tools for the agent."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("OKX_API_KEY", "")
        self._client = None
        self._tools: List[BaseTool] = []

    async def connect(self) -> None:
        """Connect to OKX MCP server and load market data tools."""
        self._client = MultiServerMCPClient(
            {
                "okx": {
                    "transport": "streamable_http",
                    "url": OKX_MCP_URL,
                    "headers": {
                        "OK-ACCESS-KEY": self._api_key,
                    },
                }
            }
        )

        all_tools = await self._client.get_tools()
        self._tools = [t for t in all_tools if t.name in ALLOWED_TOOLS]

        blocked = [t.name for t in all_tools if t.name not in ALLOWED_TOOLS]
        logger.info(
            f"OKX MCP: loaded {len(self._tools)} market data tools, "
            f"blocked {len(blocked)} execution tools: {blocked}"
        )

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        self._client = None
        self._tools = []

    def get_tools(self) -> List[BaseTool]:
        """Return the loaded market data tools for use in an agent toolkit."""
        return self._tools
