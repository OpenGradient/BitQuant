import os
import firebase_admin  # type: ignore[import-untyped]
from firebase_admin import auth  # noqa: F401
from dotenv import dotenv_values


def validate_firebase_env_vars():
    """
    Validates all required Firebase environment variables are present.
    Throws ValueError if any are missing.
    Returns a tuple of all the environment variables needed for Firebase initialization.
    """
    required_vars = [
        "FIREBASE_PROJECT_ID",
        "FIREBASE_PRIVATE_KEY_ID",
        "FIREBASE_PRIVATE_KEY",
        "FIREBASE_CLIENT_EMAIL",
        "FIREBASE_CLIENT_ID",
        "FIREBASE_CLIENT_X509_CERT_URL",
    ]

    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    # Get and process the private key to handle newlines
    private_key = os.environ.get("FIREBASE_PRIVATE_KEY").replace("\\n", "\n")

    # Sometimes, when loading multiline env variables from a .env file,
    # Only the -----BEGIN PRIVATE KEY----- is loaded (\n causes issues).
    # This handles this case.
    if "\n" not in private_key:
        config = dotenv_values()
        private_key = config["FIREBASE_PRIVATE_KEY"]

    return (
        os.environ.get("FIREBASE_PROJECT_ID"),
        os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
        private_key,
        os.environ.get("FIREBASE_CLIENT_EMAIL"),
        os.environ.get("FIREBASE_CLIENT_ID"),
        os.environ.get("FIREBASE_CLIENT_X509_CERT_URL"),
    )

def initialize_firebase():
    """Initialize Firebase with credentials from environment variables."""
    try:
        (
            project_id,
            private_key_id,
            private_key,
            client_email,
            client_id,
            client_x509_cert_url,
        ) = validate_firebase_env_vars()

        cred_obj = firebase_admin.credentials.Certificate(
            {
                "type": "service_account",
                "project_id": project_id,
                "private_key_id": private_key_id,
                "private_key": private_key,
                "client_email": client_email,
                "client_id": client_id,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": client_x509_cert_url,
                "universe_domain": "googleapis.com",
            }
        )

        return firebase_admin.initialize_app(cred_obj)
    except ValueError as e:
        print(f"Firebase initialization failed: {e}")
        raise


firebase = initialize_firebase()
