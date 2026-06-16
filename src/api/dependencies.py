import os
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    expected_key = os.environ.get("API_KEY")
    if not expected_key:
        # If API_KEY environment variable is not set, allow all requests (local dev mode)
        return None
    if api_key == expected_key:
        return api_key
    raise HTTPException(
        status_code=401,
        detail="Unauthorized: Invalid or missing X-API-Key header"
    )
