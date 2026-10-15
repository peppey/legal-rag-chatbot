"""Who is calling the API and which document groups they may read.

Requests carry the Identity Platform ID token as Bearer token. Its `role` claim selects the
document groups (see access.py). Without a valid token every request is rejected, so a
missing configuration never opens the documents. AUTH_MODE=dev replaces the token by
DEV_ROLE or the X-Dev-Role header and is meant for local work only.
"""

import json
import logging
import os
import threading
import time

from fastapi import HTTPException, Request
from google.auth import exceptions as auth_exceptions
from google.auth import jwt
from google.auth.transport import requests as auth_requests

from legal_rag_chatbot.access import Access, UnknownRole, access_for_roles
from legal_rag_chatbot.rag.query import PROJECT_ID

logger = logging.getLogger(__name__)

CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
CERTS_TTL_SECONDS = 3600
CLOCK_SKEW_SECONDS = 10
ISSUER = f"https://securetoken.google.com/{PROJECT_ID}"

_certs: dict[str, str] = {}
_certs_fetched_at = float("-inf")
_certs_lock = threading.Lock()


def _signing_certs(refresh: bool = False) -> dict[str, str]:
    """The public certificates that sign ID tokens, cached for an hour."""
    global _certs, _certs_fetched_at
    with _certs_lock:
        if refresh or time.monotonic() - _certs_fetched_at > CERTS_TTL_SECONDS:
            response = auth_requests.Request()(CERTS_URL, method="GET")
            if response.status != 200:
                raise auth_exceptions.TransportError(
                    f"Fetching the signing certificates failed ({response.status})"
                )
            _certs = json.loads(response.data)
            _certs_fetched_at = time.monotonic()
        return _certs


def verify_token(token: str) -> dict:
    """Claims of a valid ID token of this project. Raises ValueError otherwise."""
    kid = jwt.decode_header(token).get("kid")
    certs = _signing_certs()
    if kid not in certs:
        # Google rotates its signing keys, so an unknown key id may just be a new one.
        certs = _signing_certs(refresh=True)
    claims = jwt.decode(
        token,
        certs=certs,
        audience=PROJECT_ID,
        clock_skew_in_seconds=CLOCK_SKEW_SECONDS,
    )
    if claims.get("iss") != ISSUER or not claims.get("sub"):
        raise ValueError("Wrong issuer or no subject")
    return claims


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Nicht angemeldet.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def current_access(request: Request) -> Access:
    if os.environ.get("AUTH_MODE") == "dev":
        role = request.headers.get("X-Dev-Role") or os.environ.get("DEV_ROLE", "")
        user_id = f"dev-{role}"
    else:
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise _unauthorized()
        try:
            claims = verify_token(token)
        except auth_exceptions.TransportError:
            logger.exception("Could not verify the token")
            raise HTTPException(
                status_code=503, detail="Anmeldung derzeit nicht möglich."
            ) from None
        except ValueError:
            raise _unauthorized() from None
        user_id, role = claims["sub"], claims.get("role")
    try:
        return access_for_roles(user_id, [role] if isinstance(role, str) else [])
    except UnknownRole:
        raise HTTPException(status_code=403, detail="Keine Berechtigung.") from None


def warn_if_dev_mode() -> None:
    if os.environ.get("AUTH_MODE") == "dev":
        logger.warning(
            "AUTH_MODE=dev: roles are taken from DEV_ROLE, do not use in production"
        )
