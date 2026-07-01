"""Tests for the simulated JWT authentication and role-based authorization."""
from src import auth


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------
def test_extract_bearer_token():
    assert auth.extract_bearer_token("Bearer valid_user_token") == "valid_user_token"
    assert auth.extract_bearer_token("bearer abc") == "abc"
    assert auth.extract_bearer_token("valid_user_token") == "valid_user_token"
    assert auth.extract_bearer_token(None) is None
    assert auth.extract_bearer_token("") is None


# ---------------------------------------------------------------------------
# Authentication (simulated JWT validation)
# ---------------------------------------------------------------------------
def test_valid_user_token_is_accepted():
    result = auth.authenticate("valid_user_token")
    assert result.authenticated is True
    assert result.role == "user"
    assert result.client_id == "client_user_1"


def test_valid_admin_token_is_accepted():
    result = auth.authenticate("valid_admin_token")
    assert result.authenticated is True
    assert result.role == "admin"


def test_invalid_token_is_rejected():
    result = auth.authenticate("invalid_token")
    assert result.authenticated is False
    assert result.error == "invalid_token"


def test_missing_token_is_rejected():
    result = auth.authenticate(None)
    assert result.authenticated is False
    assert result.error == "missing_token"


# ---------------------------------------------------------------------------
# Authorization (role-based access control)
# ---------------------------------------------------------------------------
def test_user_can_access_normal_endpoints():
    assert auth.authorize("user", "/users") is True
    assert auth.authorize("user", "/orders") is True
    assert auth.authorize("user", "/payments") is True


def test_user_cannot_access_admin_endpoint():
    assert auth.authorize("user", "/admin") is False


def test_admin_can_access_all_endpoints():
    for endpoint in ("/public", "/users", "/orders", "/payments", "/admin"):
        assert auth.authorize("admin", endpoint) is True


def test_public_endpoint_needs_no_role():
    assert auth.authorize(None, "/public") is True
    assert auth.endpoint_requires_auth("/public") is False
    assert auth.endpoint_requires_auth("/payments") is True
