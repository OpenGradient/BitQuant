import logging
import os
import sys
from dotenv import load_dotenv
from flask import jsonify

# Load environment variables before all imports (DO NOT MOVE)
load_dotenv()

# Add the current directory to the Python path to make imports work properly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import the server module
from server import create_flask_app

from onchain.pools.solana.orca_protocol import OrcaProtocol
from onchain.pools.solana.save_protocol import SaveProtocol
from onchain.pools.solana.kamino_protocol import KaminoProtocol

# Define protocols enabled
protocols = [
    OrcaProtocol.PROTOCOL_NAME,
    SaveProtocol.PROTOCOL_NAME,
    KaminoProtocol.PROTOCOL_NAME,
]

# Create flask app
app = create_flask_app()


# Add a health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "quant-agent-server"})


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Creating app with protocols enabled: {protocols}")
    logging.info(f"Using current directory: {current_dir}")
    logging.info(f"Python path: {sys.path}")

    app.run(debug=False)
