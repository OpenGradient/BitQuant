import os
import json
from typing import Tuple, List
from langgraph.graph.graph import RunnableConfig


# Load token list mapping from address to symbol
def load_token_list():
    try:
        token_list_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "static",
            "tokenlist.json",
        )
        with open(token_list_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading token list: {e}")
        return {}

TOKEN_LIST = load_token_list()

def extract_tokens_from_config(config: RunnableConfig) -> Tuple[List[str], List[float]]:
    """Extract token holdings from the configurable context"""
    configurable = config["configurable"]
    tokens = configurable["tokens"]

    symbols = []
    quantities = []

    for token in tokens:
        address = token.get("address")
        amount = token.get("amount")

        # Look up token symbol from the address
        token_info = TOKEN_LIST.get(address)
        if not token_info:
            continue

        symbol = token_info.get("symbol")

        symbols.append(symbol)
        quantities.append(amount)

    return symbols, quantities