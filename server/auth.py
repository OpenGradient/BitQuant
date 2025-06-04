from pydantic import BaseModel
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from .firebase import auth
from .config import SKIP_TOKEN_AUTH_HEADER, SKIP_TOKEN_AUTH_KEY


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
        user_data = auth.verify_id_token(
            id_token=token, app=None, check_revoked=True, clock_skew_seconds=10
        )
        return FirebaseIDTokenData(**user_data)
    except (
        auth.InvalidIdTokenError,
        auth.ExpiredIdTokenError,
        auth.RevokedIdTokenError,
    ) as e:
        logging.exception(msg=e, stack_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )
    except auth.UserDisabledError as e:
        logging.exception(msg=e, stack_info=True)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User disabled"
        )
    except (ValueError, auth.CertificateFetchError, auth.FirebaseError) as e:
        logging.exception(msg=e, stack_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> FirebaseIDTokenData:
    """
    FastAPI dependency that verifies Firebase authentication.
    Returns the user data if authentication is successful.

    Usage:
    ```python
    @app.get("/protected")
    async def protected_route(current_user: FirebaseIDTokenData = Depends(get_current_user)):
        return f"Hello, {current_user.uid}"
    ```
    """
    if SKIP_TOKEN_AUTH_HEADER and SKIP_TOKEN_AUTH_KEY:
        skip_auth_header = request.headers.get(SKIP_TOKEN_AUTH_HEADER)
        if skip_auth_header and skip_auth_header == SKIP_TOKEN_AUTH_KEY:
            return FirebaseIDTokenData(uid="test_user")

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authenticated"
        )
    return _verify_firebase_id_token(credentials.credentials)
