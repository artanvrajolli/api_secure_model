"""Simulated OAuth2/JWT authentication and role-based authorization.

A real deployment would verify a JWT signature and read the role/scope
claims. The prototype simulates exactly that contract: a bearer token is
exchanged for an identity (client_id + role) or rejected.
"""
from dataclasses import dataclass
from typing import Optional

from src import config

# Role hierarchy: a higher level implies all lower-level permissions.
ROLE_LEVELS = {"public": 0, "user": 1, "admin": 2}


@dataclass
class AuthResult:
    """Outcome of the (simulated) JWT validation step."""
    authenticated: bool
    client_id: Optional[str] = None
    role: Optional[str] = None
    error: Optional[str] = None


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """Extract the token from an ``Authorization: Bearer <token>`` header."""
    if not authorization_header:
        return None
    parts = authorization_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    # Tolerate a raw token without the Bearer prefix (useful for demos).
    if len(parts) == 1:
        return parts[0]
    return None


def authenticate(token: Optional[str]) -> AuthResult:
    """Validate a bearer token (simulated JWT signature/claims check)."""
    if token is None:
        return AuthResult(authenticated=False, error="missing_token")
    claims = config.API_TOKENS.get(token)
    if claims is None:
        return AuthResult(authenticated=False, error="invalid_token")
    return AuthResult(
        authenticated=True,
        client_id=claims["client_id"],
        role=claims["role"],
    )


def required_role(endpoint_path: str) -> str:
    """Return the minimum role needed for an endpoint.

    Unknown paths are treated as public so that the framework can still
    answer 404; they remain covered by rate limiting, request filtering
    and risk scoring.
    """
    return config.ENDPOINT_ACCESS.get(endpoint_path, "public")


def endpoint_requires_auth(endpoint_path: str) -> bool:
    """True if the endpoint needs a valid token at all."""
    return required_role(endpoint_path) != "public"


def authorize(role: Optional[str], endpoint_path: str) -> bool:
    """Role-based access control: does `role` satisfy the endpoint policy?"""
    needed = required_role(endpoint_path)
    if needed == "public":
        return True
    if role is None:
        return False
    return ROLE_LEVELS.get(role, 0) >= ROLE_LEVELS[needed]
