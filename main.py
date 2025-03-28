import logging
from dotenv import load_dotenv
from server import create_flask_app

from onchain.pools.solana.orca_protocol import OrcaProtocol
from onchain.pools.solana.save_protocol import SaveProtocol
from onchain.pools.solana.kamino_protocol import KaminoProtocol

# Load environment variables
load_dotenv()

# Define protocols enabled
protocols = [
    OrcaProtocol.PROTOCOL_NAME,
    SaveProtocol.PROTOCOL_NAME,
    KaminoProtocol.PROTOCOL_NAME,
]

# Create flask app
app = create_flask_app()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Creating app with protocols enabled: {protocols}")

    app.run(debug=False)
