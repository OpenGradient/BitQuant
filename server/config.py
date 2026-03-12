import os

SKIP_TOKEN_AUTH_HEADER = os.getenv("SKIP_TOKEN_AUTH_HEADER")
SKIP_TOKEN_AUTH_KEY = os.getenv("SKIP_TOKEN_AUTH_KEY")

DUMMY_X402_API_KEY = os.getenv("DUMMY_X402_API_KEY", "dummy")
LLM_SERVER_URL: str = os.getenv("LLM_SERVER_URL", "https://llm.opengradient.ai/v1")
OG_RPC_URL: str = os.getenv("OG_RPC_URL", "https://ogevmdevnet.opengradient.ai")
WALLET_PRIV_KEY: str = os.getenv("WALLET_PRIV_KEY")

# Use OG TEE flag for LLM inference
USE_TEE = os.getenv("USE_OG_TEE", "").lower() == "true"
