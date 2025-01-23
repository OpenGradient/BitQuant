import logging
from dotenv import load_dotenv
from server import create_flask_app

# Load environment variables
load_dotenv()

# Create the Flask app at module level
app = create_flask_app()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    app.run(debug=True)
