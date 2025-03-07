import logging
from dotenv import load_dotenv
from server import create_flask_app

from defi.pools.solana.orca_protocol import OrcaProtocol

# Load environment variables
load_dotenv()

# Define protocols enabled
protocols = [OrcaProtocol.PROTOCOL_NAME]

# Create flask app
app = create_flask_app(protocols)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Creating app with protocols enabled: {protocols}")

    app.run(debug=True)
