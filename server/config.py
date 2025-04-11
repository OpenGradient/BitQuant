import os
import logging

# See if we are running in subnet mode
SUBNET_MODE = os.getenv("subnet_mode", "false").lower() == "true"
logging.info(f"Running in subnet mode: {SUBNET_MODE}")

# Bypass daily limit for miner wallet
MINER_TOKEN = os.getenv("MINER_TOKEN")
