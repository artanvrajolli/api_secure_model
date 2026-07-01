"""Signature-based request filtering (WAF-style first line of defence).

Inspects the URL path, the query string and the request body for common
attack patterns (SQL injection, cross-site scripting, path traversal,
command injection). This is intentionally rule-based: it catches known
signatures cheaply, while the ML risk engine covers behavioural anomalies
that rules cannot express.
"""
import re
from dataclasses import dataclass
from typing import Optional

# Each entry: (human-readable name, compiled regex).
SUSPICIOUS_PATTERNS = [
    ("sql_injection", re.compile(r"('|%27)\s*(or|and)\s+\d+\s*=\s*\d+", re.IGNORECASE)),
    ("sql_injection", re.compile(r"\bunion\s+select\b", re.IGNORECASE)),
    ("sql_injection", re.compile(r"\bdrop\s+table\b", re.IGNORECASE)),
    ("sql_injection", re.compile(r";\s*--")),
    ("xss_script_tag", re.compile(r"<\s*script", re.IGNORECASE)),
    ("xss_script_tag", re.compile(r"javascript\s*:", re.IGNORECASE)),
    ("path_traversal", re.compile(r"\.\./")),
    ("path_traversal", re.compile(r"%2e%2e%2f", re.IGNORECASE)),
    ("path_traversal", re.compile(r"/etc/passwd", re.IGNORECASE)),
    ("command_injection", re.compile(r"\b(cmd\.exe|/bin/sh|/bin/bash)\b", re.IGNORECASE)),
]


@dataclass
class FilterResult:
    suspicious: bool
    reason: Optional[str] = None      # pattern category that matched
    location: Optional[str] = None    # where it matched: path/query/body


def find_suspicious_pattern(text: str) -> Optional[str]:
    """Return the category of the first attack signature found in `text`."""
    if not text:
        return None
    for name, pattern in SUSPICIOUS_PATTERNS:
        if pattern.search(text):
            return name
    return None


def inspect_request(path: str, query_string: str = "", body_text: str = "") -> FilterResult:
    """Inspect all attacker-controlled parts of a request."""
    for location, text in (("path", path), ("query", query_string), ("body", body_text)):
        reason = find_suspicious_pattern(text)
        if reason:
            return FilterResult(suspicious=True, reason=reason, location=location)
    return FilterResult(suspicious=False)
