import os
import logging

SKIP_TOKEN_AUTH_HEADER = os.getenv("SKIP_TOKEN_AUTH_HEADER")
SKIP_TOKEN_AUTH_KEY = os.getenv("SKIP_TOKEN_AUTH_KEY")

# See if we are running in subnet mode
SUBNET_MODE = os.getenv("subnet_mode", "false").lower() == "true"
logging.info(f"Running in subnet mode: {SUBNET_MODE}")

DUMMY_X402_API_KEY = os.getenv("DUMMY_X402_API_KEY", "dummy")
LLM_SERVER_URL: str = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8080/v1")
OG_RPC_URL: str = os.getenv("OG_RPC_URL", "https://eth-devnet.opengradient.ai")
WALLET_PRIV_KEY: str = os.getenv("WALLET_PRIV_KEY")

# Bypass daily limit for miner wallet
MINER_TOKEN = os.getenv("MINER_TOKEN")

# Use OG TEE flag for LLM inference
USE_TEE = os.getenv("USE_OG_TEE", "").lower() == "true"

DAILY_LIMIT_BYPASS_WALLETS = [
    "7FVPurQDkbj6g9dm5B52oCUr7JxqRpoYQcNitKaWVSgS",  # miner
]
