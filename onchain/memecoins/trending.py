from typing import List

from langchain_core.tools import BaseTool, tool
from langgraph.graph.graph import RunnableConfig

from api.api_types import Token

TRENDING_POOLS_URL = (
    "https://pro-api.coingecko.com/api/v3/onchain/networks/solana/trending_pools"
)


@tool
def get_trending_tokens(
    config: RunnableConfig = None,
) -> List[Token]:
    pass
