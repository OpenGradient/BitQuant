import os
import firebase_admin  # type: ignore[import-untyped]
from firebase_admin import auth  # noqa: F401

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
        "FIREBASE_CLIENT_ID"
    ]

    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Get and process the private key to handle newlines
    private_key = os.environ.get("FIREBASE_PRIVATE_KEY").replace("\\n", "\n")

    # Get the optional client_x509_cert_url or use a default value
    client_x509_cert_url = os.environ.get(
        "FIREBASE_CLIENT_X509_CERT_URL",
        "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40bitquant-afc7f.iam.gserviceaccount.com"
    )

    return (
        os.environ.get("FIREBASE_PROJECT_ID"),
        os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
        private_key,
        os.environ.get("FIREBASE_CLIENT_EMAIL"),
        os.environ.get("FIREBASE_CLIENT_ID"),
        client_x509_cert_url
    )

# TODO: SECURITY ISSUE - it's better to use .env variable for this.
# Instead, maybe use AWS secrets manager?
# cred_obj = firebase_admin.credentials.Certificate(
#     "/opt/app/secrets/vanna-portal-418018-1e27956b0918.json"
# )
def initialize_firebase():
    """Initialize Firebase with credentials from environment variables."""
    try:
        project_id, private_key_id, private_key, client_email, client_id, client_x509_cert_url = validate_firebase_env_vars()
        
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
                "universe_domain": "googleapis.com"
            }
        )
        
        return firebase_admin.initialize_app(cred_obj)
    except ValueError as e:
        print(f"Firebase initialization failed: {e}")
        raise

firebase = initialize_firebase()