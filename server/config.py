import os

SKIP_TOKEN_AUTH_HEADER = os.getenv("SKIP_TOKEN_AUTH_HEADER")
SKIP_TOKEN_AUTH_KEY = os.getenv("SKIP_TOKEN_AUTH_KEY")

OG_RPC_URL: str = os.getenv("OG_RPC_URL", "https://ogevmdevnet.opengradient.ai")
WALLET_PRIV_KEY: str = os.getenv("WALLET_PRIV_KEY")

# Use OG TEE flag for LLM inference
USE_TEE = os.getenv("USE_OG_TEE", "").lower() == "true"

# Base chain RPC for OPG token gating
BASE_RPC_URL: str = os.getenv(
    "BASE_RPC_URL",
    "https://responsive-attentive-panorama.base-mainnet.quiknode.pro/11a3fd4381ebfe3d6cef02189257575b0b4250cc/",
)
OPG_TOKEN_ADDRESS = "0xFbC2051AE2265686a469421b2C5A2D5462FbF5eB"
OPG_HOLDER_THRESHOLD = 1000 * 10**18  # raw units, 18 decimals
