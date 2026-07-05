from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import get_settings


security = HTTPBasic(auto_error=False)


def require_basic_auth(credentials: HTTPBasicCredentials | None = Depends(security)) -> None:
    """Validate optional HTTP Basic credentials for protected routes."""
    settings = get_settings()
    if not settings.auth_enabled:
        return
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    # Use constant-time comparison so credential checks do not leak timing clues.
    username_ok = secrets.compare_digest(credentials.username, settings.api_basic_auth_username or "")
    password_ok = secrets.compare_digest(credentials.password, settings.api_basic_auth_password or "")
    if not (username_ok and password_ok):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
