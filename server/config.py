import os

SKIP_TOKEN_AUTH_HEADER = os.getenv("SKIP_TOKEN_AUTH_HEADER")
SKIP_TOKEN_AUTH_KEY = os.getenv("SKIP_TOKEN_AUTH_KEY")

OG_RPC_URL: str = os.getenv("OG_RPC_URL", "https://ogevmdevnet.opengradient.ai")
WALLET_PRIV_KEY: str = os.getenv("WALLET_PRIV_KEY")

# Use OG TEE flag for LLM inference
USE_TEE = os.getenv("USE_OG_TEE", "").lower() == "true"
