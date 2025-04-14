import os
import logging

# See if we are running in subnet mode
SUBNET_MODE = os.getenv("subnet_mode", "false").lower() == "true"
logging.info(f"Running in subnet mode: {SUBNET_MODE}")

# Bypass daily limit for miner wallet
MINER_WALLET_ADDRESS = "7FVPurQDkbj6g9dm5B52oCUr7JxqRpoYQcNitKaWVSgS"
