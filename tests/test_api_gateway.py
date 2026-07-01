"""End-to-end tests of the gateway pipeline through the FastAPI app.

These tests exercise the full middleware chain: authentication,
authorization, rate limiting, request filtering and the risk engine.
They are written to pass both with and without a trained model artifact
on disk (the ML term can only raise risk scores, never lower them).
"""
import pytest
from fastapi.testclient import TestClient

from src import config, gateway
from src.main import app

client = TestClient(app)

USER = {"Authorization": "Bearer valid_user_token"}
ADMIN = {"Authorization": "Bearer valid_admin_token"}
INVALID = {"Authorization": "Bearer invalid_token"}


@pytest.fixture(autouse=True)
def clean_gateway_state():
    """Isolate every test from telemetry left behind by previous tests."""
    gateway.reset_state()
    yield
    gateway.reset_state()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def test_public_endpoint_needs_no_token():
    response = client.get("/public")
    assert response.status_code == 200
    assert response.json()["healthy"] is True


def test_valid_token_is_accepted():
    response = client.get("/users", headers=USER)
    assert response.status_code == 200
    assert "X-Risk-Score" in response.headers
    assert response.headers["X-Risk-Decision"] in ("allow", "flag")


def test_invalid_token_is_rejected():
    response = client.get("/users", headers=INVALID)
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"


def test_missing_token_is_rejected():
    response = client.get("/users")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------
def test_normal_user_blocked_from_admin_endpoint():
    response = client.get("/admin", headers=USER)
    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"


def test_admin_user_allowed_on_admin_endpoint():
    response = client.get("/admin", headers=ADMIN)
    assert response.status_code == 200
    assert response.json()["message"] == "Admin panel"


def test_payments_requires_authentication():
    assert client.post("/payments", headers=USER).status_code == 200
    assert client.post("/payments").status_code == 401


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
def test_rate_limit_returns_429_when_exceeded():
    limit = config.RATE_LIMIT_MAX_REQUESTS
    for _ in range(limit):
        assert client.get("/public").status_code == 200
    response = client.get("/public")
    assert response.status_code == 429
    assert response.json()["error"] == "rate_limited"


def test_rate_limit_is_applied_per_client():
    """Clients identified via X-Forwarded-For are limited independently."""
    first = {"X-Forwarded-For": "198.51.100.1"}
    second = {"X-Forwarded-For": "198.51.100.2"}
    for _ in range(config.RATE_LIMIT_MAX_REQUESTS):
        client.get("/public", headers=first)
    assert client.get("/public", headers=first).status_code == 429
    assert client.get("/public", headers=second).status_code == 200


# ---------------------------------------------------------------------------
# Request filtering
# ---------------------------------------------------------------------------
def test_sql_injection_payload_is_blocked():
    response = client.post(
        "/payments", headers=USER, content="{\"note\": \"' OR 1=1\"}"
    )
    assert response.status_code == 400
    assert response.json()["error"] == "malicious_payload"


def test_script_tag_in_query_is_blocked():
    response = client.get(
        "/users", headers=USER, params={"q": "<script>alert(1)</script>"}
    )
    assert response.status_code == 400


def test_path_traversal_is_blocked():
    response = client.get("/public?file=../../etc/passwd")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Intelligent risk engine
# ---------------------------------------------------------------------------
def test_gateway_blocks_high_risk_request():
    """Credential-stuffing pattern: many failed logins from one client,
    then a valid token targets the sensitive /payments endpoint. The risk
    engine must block it even though the token itself is valid."""
    for _ in range(8):
        assert client.get("/users", headers=INVALID).status_code == 401

    response = client.post("/payments", headers=USER)
    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "high_risk_blocked"
    assert body["risk_score"] > config.RISK_THROTTLE_MAX


def _risk_score_of(response) -> float:
    """Risk score of a response, whether forwarded or rejected."""
    if "X-Risk-Score" in response.headers:
        return float(response.headers["X-Risk-Score"])
    return float(response.json()["risk_score"])


def test_risk_score_increases_after_failed_auth_attempts():
    baseline = _risk_score_of(client.get("/users", headers=USER))
    gateway.reset_state()
    for _ in range(3):
        client.get("/users", headers=INVALID)
    response = client.get("/users", headers=USER)
    # The same request is now visibly riskier than the clean baseline.
    assert _risk_score_of(response) > baseline
