"""Synthetic API traffic dataset generator.

Produces a labelled, reproducible dataset of API gateway traffic containing
~85% normal requests and ~15% anomalies of seven realistic attack /
misbehaviour types:

    brute_force          repeated failed authentications (401)
    rate_abuse           excessive requests per minute (rate-limit abuse)
    unauthorized_access  normal users probing restricted endpoints (403)
    injection_payload    SQLi / XSS / traversal payloads (400)
    slow_response        abnormally slow responses (possible DoS symptom)
    oversized_request    abnormally large request bodies
    scanning_4xx         endpoint scanning producing repeated 401/403/404

The dataset is synthetic (documented in data/README.md) but its feature
distributions are chosen to mimic plausible production traffic, including
mild overlap between normal and anomalous behaviour so that the ML task is
not trivially separable.

Run:  python -m src.generate_dataset
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Allow running both as `python -m src.generate_dataset` and as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config

# Fixed simulation day so the output is fully deterministic.
SIMULATION_START = datetime(2026, 6, 1, 0, 0, 0)
SIMULATION_SECONDS = 24 * 3600

ENDPOINTS = ["/public", "/users", "/orders", "/payments", "/admin"]
NORMAL_ENDPOINT_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

COLUMNS = [
    "timestamp", "client_id", "ip_address", "endpoint", "http_method",
    "status_code", "response_time_ms", "request_size_bytes",
    "requests_per_minute", "failed_auth_count", "token_valid", "user_role",
    "endpoint_sensitivity", "payload_suspicious", "anomaly_label",
    "anomaly_type",
]


def _method_for(endpoints: np.ndarray) -> np.ndarray:
    return np.where(endpoints == "/payments", "POST", "GET")


def _sensitivity_for(endpoints: np.ndarray) -> np.ndarray:
    return np.vectorize(config.ENDPOINT_SENSITIVITY.get)(endpoints)


def _normal_block(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Legitimate traffic from regular API consumers."""
    client_idx = rng.integers(1, 41, n)
    endpoints = rng.choice(ENDPOINTS, n, p=NORMAL_ENDPOINT_WEIGHTS)
    # Admin endpoint is used by admins; other endpoints mostly by users.
    roles = np.where(
        endpoints == "/admin", "admin",
        np.where(rng.random(n) < 0.95, "user", "admin"),
    )
    roles = np.where(
        (endpoints == "/public") & (rng.random(n) < 0.4), "anonymous", roles
    )
    methods = _method_for(endpoints)
    status = rng.choice([200, 404, 500], n, p=[0.95, 0.03, 0.02])
    # Mostly fast responses, but ~3% legitimately slow requests (heavy
    # queries, cold caches) that overlap with the slow_response anomaly.
    response_time = np.clip(rng.lognormal(np.log(110), 0.35, n), 20, 800)
    slow_legit = rng.random(n) < 0.03
    response_time = np.where(slow_legit, rng.uniform(800, 2500, n), response_time)
    size = np.where(
        methods == "POST", rng.integers(500, 4000, n), rng.integers(200, 1200, n)
    )
    # ~2% legitimate large uploads overlapping the oversized_request anomaly.
    big_legit = (rng.random(n) < 0.02) & (methods == "POST")
    size = np.where(big_legit, rng.integers(10_000, 80_000, n), size)
    # Mostly steady clients, but ~5% legitimate activity bursts.
    rpm = np.clip(rng.poisson(7, n) + 1, 1, 14)
    rpm = np.where(rng.random(n) < 0.05, rng.integers(15, 31, n), rpm)
    return pd.DataFrame({
        "client_id": [f"client_{i:03d}" for i in client_idx],
        "ip_address": [f"10.0.{i // 10}.{i % 250 + 1}" for i in client_idx],
        "endpoint": endpoints,
        "http_method": methods,
        "status_code": status,
        "response_time_ms": response_time.round(1),
        "request_size_bytes": size,
        "requests_per_minute": rpm,
        # Occasional mistyped passwords, including a few repeat offenders.
        "failed_auth_count": rng.choice([0, 1, 2, 3], n, p=[0.85, 0.10, 0.04, 0.01]),
        "token_valid": np.ones(n, dtype=int),
        "user_role": roles,
        "payload_suspicious": np.zeros(n, dtype=int),
        "anomaly_label": np.zeros(n, dtype=int),
        "anomaly_type": "none",
    })


def _attacker_ids(rng: np.random.Generator, n: int) -> tuple[list, list]:
    idx = rng.integers(1, 16, n)
    ids = [f"attacker_{i:02d}" for i in idx]
    ips = [f"203.0.113.{i * 3 + 1}" for i in idx]
    return ids, ips


def _anomaly_block(rng: np.random.Generator, n: int, kind: str) -> pd.DataFrame:
    """One block of anomalous traffic of the given attack type."""
    ids, ips = _attacker_ids(rng, n)
    df = pd.DataFrame({
        "client_id": ids,
        "ip_address": ips,
        "endpoint": rng.choice(ENDPOINTS, n, p=NORMAL_ENDPOINT_WEIGHTS),
        "status_code": np.full(n, 200),
        "response_time_ms": np.clip(rng.lognormal(np.log(110), 0.35, n), 20, 800).round(1),
        "request_size_bytes": rng.integers(200, 2000, n),
        "requests_per_minute": np.clip(rng.poisson(7, n) + 1, 1, 14),
        "failed_auth_count": np.zeros(n, dtype=int),
        "token_valid": np.ones(n, dtype=int),
        "user_role": "anonymous",
        "payload_suspicious": np.zeros(n, dtype=int),
        "anomaly_label": np.ones(n, dtype=int),
        "anomaly_type": kind,
    })

    if kind == "brute_force":
        # Includes "low and slow" attempts that overlap with users who
        # simply mistype their password a few times.
        df["endpoint"] = rng.choice(["/users", "/admin", "/payments"], n)
        df["status_code"] = 401
        df["failed_auth_count"] = rng.integers(2, 21, n)
        df["token_valid"] = 0
        df["requests_per_minute"] = rng.integers(4, 41, n)
        df["response_time_ms"] = rng.uniform(30, 150, n).round(1)
        df["request_size_bytes"] = rng.integers(200, 800, n)

    elif kind == "rate_abuse":
        # Lower bound overlaps with legitimate client activity bursts.
        df["status_code"] = rng.choice([200, 429], n, p=[0.4, 0.6])
        df["requests_per_minute"] = rng.integers(20, 301, n)
        df["response_time_ms"] = rng.uniform(100, 700, n).round(1)
        df["user_role"] = "user"

    elif kind == "unauthorized_access":
        df["endpoint"] = rng.choice(["/admin", "/payments"], n, p=[0.7, 0.3])
        df["status_code"] = 403
        df["user_role"] = rng.choice(["user", "anonymous"], n)
        df["failed_auth_count"] = rng.integers(0, 7, n)
        df["requests_per_minute"] = rng.integers(8, 41, n)

    elif kind == "injection_payload":
        df["endpoint"] = rng.choice(["/users", "/orders", "/payments"], n)
        df["status_code"] = 400
        df["payload_suspicious"] = 1
        df["request_size_bytes"] = rng.integers(800, 5000, n)
        df["requests_per_minute"] = rng.integers(5, 31, n)
        df["token_valid"] = rng.choice([0, 1], n, p=[0.5, 0.5])

    elif kind == "slow_response":
        # Only symptom: abnormal latency (e.g. resource-exhaustion attack).
        # The lower bound overlaps with legitimately slow heavy queries.
        df["response_time_ms"] = rng.uniform(900, 8000, n).round(1)
        df["status_code"] = rng.choice([200, 500], n, p=[0.6, 0.4])
        df["user_role"] = "user"

    elif kind == "oversized_request":
        # Overlaps with legitimate large uploads at the lower end.
        df["endpoint"] = rng.choice(["/payments", "/orders"], n)
        df["status_code"] = rng.choice([200, 413], n, p=[0.5, 0.5])
        df["request_size_bytes"] = rng.integers(20_000, 500_001, n)
        df["user_role"] = "user"

    elif kind == "scanning_4xx":
        df["status_code"] = rng.choice([401, 403, 404], n)
        df["requests_per_minute"] = rng.integers(8, 61, n)
        df["failed_auth_count"] = rng.integers(1, 9, n)
        df["token_valid"] = rng.choice([0, 1], n, p=[0.7, 0.3])

    else:
        raise ValueError(f"Unknown anomaly kind: {kind}")

    df["http_method"] = _method_for(df["endpoint"].to_numpy())
    if kind == "oversized_request":
        df["http_method"] = "POST"
    return df


ANOMALY_KINDS = [
    "brute_force", "rate_abuse", "unauthorized_access", "injection_payload",
    "slow_response", "oversized_request", "scanning_4xx",
]


def generate_dataset(
    n_records: int = 12_000,
    anomaly_ratio: float = 0.15,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Build the full labelled dataset (deterministic for a given seed)."""
    rng = np.random.default_rng(seed)
    n_anomalies = int(n_records * anomaly_ratio)
    n_normal = n_records - n_anomalies

    blocks = [_normal_block(rng, n_normal)]
    per_kind = n_anomalies // len(ANOMALY_KINDS)
    remainder = n_anomalies - per_kind * len(ANOMALY_KINDS)
    for i, kind in enumerate(ANOMALY_KINDS):
        blocks.append(_anomaly_block(rng, per_kind + (1 if i < remainder else 0), kind))

    df = pd.concat(blocks, ignore_index=True)
    # Shuffle rows, then assign chronologically sorted timestamps so normal
    # and anomalous traffic interleave across the simulated day.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    offsets = np.sort(rng.integers(0, SIMULATION_SECONDS, len(df)))
    df["timestamp"] = [
        (SIMULATION_START + timedelta(seconds=int(s))).isoformat() for s in offsets
    ]
    df["endpoint_sensitivity"] = _sensitivity_for(df["endpoint"].to_numpy())
    return df[COLUMNS]


def main() -> None:
    df = generate_dataset()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_PATH, index=False)

    print(f"Dataset written to {config.DATASET_PATH}")
    print(f"Total records:   {len(df)}")
    print(f"Normal records:  {(df['anomaly_label'] == 0).sum()}")
    print(f"Anomalies:       {(df['anomaly_label'] == 1).sum()}")
    print("\nAnomaly breakdown:")
    print(df[df.anomaly_label == 1]["anomaly_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
