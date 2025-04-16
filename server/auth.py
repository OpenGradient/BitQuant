from .firebase import auth
from pydantic import BaseModel
from functools import wraps
from typing import Callable, TypeVar, cast
from flask import abort, request, jsonify, g
from .logging import logger
from .config import SKIP_FIREBASE_TOKEN_AUTH

class FirebaseIDTokenData(BaseModel):
    uid: str

def _verify_firebase_id_token(token: str) -> FirebaseIDTokenData:
    """Verify Firebase ID token and returns the UID.

    Args:
        token (str): Firebase ID token, parsed from the Authorization header.

    Returns:
        FirebaseIDTokenData: Pydantic model containing the user's firebase `uid`.
    """
    try:
        user_data = auth.verify_id_token(id_token=token, app=None, check_revoked=True, clock_skew_seconds=10)
        return FirebaseIDTokenData(**user_data)
    except (auth.InvalidIdTokenError, auth.ExpiredIdTokenError, auth.RevokedIdTokenError) as e:
        logger.exception(msg=e, stack_info=True)
        abort(401)
    except auth.UserDisabledError as e:
        logger.exception(msg=e, stack_info=True)
        abort(404)
    except (ValueError, auth.CertificateFetchError, auth.FirebaseError) as e:
        logger.exception(msg=e, stack_info=True)
        abort(500)

T = TypeVar('T')

def protected_route(f: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to require Firebase authentication for Flask routes.
    Stores user data in Flask's `g` object as `g.user`.
    
    Usage:
    ```
    @app.route('/protected')
    @auth_required
    def protected_route():
        return f"Hello, {g.user.uid}" # g.user is the FirebaseIDTokenData pydantic model
    ```
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if SKIP_FIREBASE_TOKEN_AUTH:
            return f(*args, **kwargs)

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header required"}), 401

        user_data = _verify_firebase_id_token(auth_header[7:])

        if not user_data:
            return jsonify({"error": "Invalid or expired token"}), 401

        g.user = user_data

        return f(*args, **kwargs)
        
    return cast(Callable[..., T], decorated_function)