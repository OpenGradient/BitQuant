import logging
from dotenv import load_dotenv
from server import create_flask_app

# Load environment variables
load_dotenv()

protocols = ["orca"]

# Create flask app
app = create_flask_app(protocols)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Creating app with protocols enabled: {protocols}")

    app.run(debug=True)
