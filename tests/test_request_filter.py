"""Tests for the signature-based request filter (WAF-style rules)."""
import pytest

from src.request_filter import find_suspicious_pattern, inspect_request


# ---------------------------------------------------------------------------
# Attack signatures are detected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload,expected", [
    ("' OR 1=1", "sql_injection"),
    ("name=admin' or 1=1 --", "sql_injection"),
    ("UNION SELECT password FROM users", "sql_injection"),
    ("DROP TABLE customers", "sql_injection"),
    ("<script>alert('xss')</script>", "xss_script_tag"),
    ("javascript:alert(1)", "xss_script_tag"),
    ("../../etc/passwd", "path_traversal"),
    ("%2e%2e%2fadmin", "path_traversal"),
    ("run cmd.exe /c whoami", "command_injection"),
])
def test_attack_patterns_are_detected(payload, expected):
    assert find_suspicious_pattern(payload) == expected


# ---------------------------------------------------------------------------
# Legitimate content passes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload", [
    "",
    '{"amount": 100, "currency": "EUR"}',
    "ordinary text describing an order",
    "user@example.com",
    "search=laptops&page=2",
])
def test_clean_payloads_pass(payload):
    assert find_suspicious_pattern(payload) is None


# ---------------------------------------------------------------------------
# Full request inspection (path + query + body)
# ---------------------------------------------------------------------------
def test_inspect_request_flags_suspicious_body():
    result = inspect_request("/payments", "", '{"note": "\' OR 1=1"}')
    assert result.suspicious is True
    assert result.reason == "sql_injection"
    assert result.location == "body"


def test_inspect_request_flags_suspicious_query():
    result = inspect_request("/users", "q=<script>steal()</script>", "")
    assert result.suspicious is True
    assert result.location == "query"


def test_inspect_request_flags_suspicious_path():
    result = inspect_request("/files/../../etc/passwd", "", "")
    assert result.suspicious is True
    assert result.location == "path"


def test_inspect_request_passes_clean_request():
    result = inspect_request("/orders", "page=1", '{"item": "laptop"}')
    assert result.suspicious is False
    assert result.reason is None
