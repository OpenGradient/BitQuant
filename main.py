import logging

from dotenv import load_dotenv

from server import create_flask_app

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    load_dotenv()

    app = create_flask_app()
    app.run(debug=True)
