"""Tests for the hybrid risk engine (heuristic scoring + decision bands).

These tests use `use_model=False` so the assertions are exact and do not
depend on whether a trained model artifact happens to exist on disk.
"""
import pytest

from src.risk_engine import RiskEngine, decide


def make_features(**overrides):
    """A perfectly benign request context; override selected fields."""
    features = {
        "requests_per_minute": 2,
        "failed_auth_count": 0,
        "token_valid": 1,
        "endpoint_sensitivity": 0.5,
        "payload_suspicious": 0,
        "request_size_bytes": 500,
        "response_time_ms": 120.0,
    }
    features.update(overrides)
    return features


@pytest.fixture
def engine():
    return RiskEngine(use_model=False)


# ---------------------------------------------------------------------------
# Decision bands (0-30 allow | 31-60 flag | 61-80 throttle | 81-100 block)
# ---------------------------------------------------------------------------
def test_decision_bands():
    assert decide(0) == "allow"
    assert decide(30) == "allow"
    assert decide(31) == "flag"
    assert decide(60) == "flag"
    assert decide(61) == "throttle"
    assert decide(80) == "throttle"
    assert decide(81) == "block"
    assert decide(100) == "block"


# ---------------------------------------------------------------------------
# Scoring behaviour
# ---------------------------------------------------------------------------
def test_benign_request_is_allowed(engine):
    assessment = engine.score(make_features())
    assert assessment.score <= 30
    assert assessment.decision == "allow"


def test_brute_force_behaviour_is_blocked(engine):
    assessment = engine.score(make_features(
        failed_auth_count=10, token_valid=0,
        requests_per_minute=30, endpoint_sensitivity=1.0,
    ))
    assert assessment.score >= 81
    assert assessment.decision == "block"


def test_malicious_scores_higher_than_benign(engine):
    benign = engine.score(make_features()).score
    malicious = engine.score(make_features(
        failed_auth_count=6, token_valid=0, payload_suspicious=1,
    )).score
    assert malicious > benign


def test_suspicious_payload_raises_score(engine):
    base = engine.score(make_features()).score
    with_payload = engine.score(make_features(payload_suspicious=1)).score
    assert with_payload - base >= 40


def test_invalid_token_raises_score(engine):
    base = engine.score(make_features()).score
    invalid = engine.score(make_features(token_valid=0)).score
    assert invalid - base >= 25


def test_score_is_capped_at_100(engine):
    assessment = engine.score(make_features(
        failed_auth_count=50, token_valid=0, payload_suspicious=1,
        requests_per_minute=500, endpoint_sensitivity=1.0,
        request_size_bytes=1_000_000, response_time_ms=9_000,
    ))
    assert assessment.score == 100


def test_engine_without_model_reports_no_probability(engine):
    assessment = engine.score(make_features())
    assert assessment.ml_probability is None
