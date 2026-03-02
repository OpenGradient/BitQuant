from nacl.signing import VerifyKey
from api.api_types import SolanaVerifyRequest
from base58 import b58decode
from server.firebase import auth


def verify_solana_signature(verify_request: SolanaVerifyRequest) -> str:
    """Verify Solana signature and create Firebase custom token.
    
    Args:
        verify_request: Solana verification request containing address, signature, and message.
        
    Returns:
        Firebase custom token as string.
        
    Raises:
        ValueError: If signature verification fails or token creation fails.
    """
    public_key = b58decode(verify_request.address)
    signature = b58decode(verify_request.signature)
    message = verify_request.message.encode("utf-8")

    verify_key = VerifyKey(public_key)
    verify_key.verify(message, signature)

    uid = f"wallet_{verify_request.address}"
    custom_token = auth.create_custom_token(uid)

    if isinstance(custom_token, bytes):
        token_bytes = custom_token
    elif isinstance(custom_token, str):
        token_bytes = custom_token.encode("utf-8")
    else:
        token_bytes = str(custom_token).encode("utf-8")

    return token_bytes.decode("utf-8")
