import os
import logging

SKIP_TOKEN_AUTH_HEADER = os.getenv("SKIP_TOKEN_AUTH_HEADER")
SKIP_TOKEN_AUTH_KEY = os.getenv("SKIP_TOKEN_AUTH_KEY")

# See if we are running in subnet mode
SUBNET_MODE = os.getenv("subnet_mode", "false").lower() == "true"
logging.info(f"Running in subnet mode: {SUBNET_MODE}")

# Bypass daily limit for miner wallet
MINER_TOKEN = os.getenv("MINER_TOKEN")

# Use OG TEE flag for LLM inference
USE_TEE = os.getenv("USE_OG_TEE", "").lower() == "true"

DAILY_LIMIT_BYPASS_WALLETS = [
    "7FVPurQDkbj6g9dm5B52oCUr7JxqRpoYQcNitKaWVSgS",  # miner
]

LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"
