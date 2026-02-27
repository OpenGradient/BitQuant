import logging
import os
import sys
import asyncio
from dotenv import load_dotenv
import uvicorn

# Load environment variables before all imports (DO NOT MOVE)
load_dotenv()

# Add the current directory to the Python path to make imports work properly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import the server module
from server.fastapi_server import create_fastapi_app

from onchain.pools.solana.orca_protocol import OrcaProtocol
from onchain.pools.solana.save_protocol import SaveProtocol
from onchain.pools.solana.kamino_protocol import KaminoProtocol
from onchain.pools.ethereum.uniswap_v3_protocol import UniswapV3Protocol
from onchain.pools.ethereum.aave_protocol import AaveProtocol

# Define protocols enabled
protocols = [
    OrcaProtocol.PROTOCOL_NAME,
    SaveProtocol.PROTOCOL_NAME,
    KaminoProtocol.PROTOCOL_NAME,
    UniswapV3Protocol.PROTOCOL_NAME,
    AaveProtocol.PROTOCOL_NAME,
]

# Create the FastAPI app
app = create_fastapi_app()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Creating app with protocols enabled: {protocols}")
    logging.info(f"Using current directory: {current_dir}")
    logging.info(f"Python path: {sys.path}")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", workers=1)
