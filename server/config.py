import os
import logging

SKIP_TOKEN_AUTH_HEADER = os.getenv("SKIP_TOKEN_AUTH_HEADER")
SKIP_TOKEN_AUTH_KEY = os.getenv("SKIP_TOKEN_AUTH_KEY")

# See if we are running in subnet mode
SUBNET_MODE = os.getenv("subnet_mode", "false").lower() == "true"
logging.info(f"Running in subnet mode: {SUBNET_MODE}")

DUMMY_X402_API_KEY = os.getenv("DUMMY_X402_API_KEY", "dummy")
LLM_SERVER_URL: str = os.getenv("LLM_SERVER_URL", "https://llmogevm.opengradient.ai/v1")
OG_RPC_URL: str = os.getenv("OG_RPC_URL", "https://ogevmdevnet.opengradient.ai")
WALLET_PRIV_KEY: str = os.getenv("WALLET_PRIV_KEY")

# Use OG TEE flag for LLM inference
USE_TEE = os.getenv("USE_OG_TEE", "").lower() == "true"
