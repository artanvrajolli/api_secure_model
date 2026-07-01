"""Load test / performance evaluation of the Intelligent Secure API Gateway.

Starts the real FastAPI application (full gateway pipeline + trained model)
in an in-process uvicorn server and drives it with a mix of concurrent
requests representative of the security scenarios:

    - authenticated normal traffic  (valid user token)   -> allowed
    - public traffic                (no token)            -> allowed
    - injection payloads            (SQLi in body)        -> blocked (400)
    - invalid tokens                                      -> blocked (401)
    - privilege escalation          (user hits /admin)    -> blocked (403)
    - burst from a single client                          -> rate-limited (429)

Measured indicators (saved to results/performance_results.csv; the ML
classification metrics live separately in results/model_metrics.csv written
by evaluate_model.py, so the two do not collide):
    total requests, wall-clock duration, throughput (req/s),
    avg / p50 / p95 / max response time, allowed / blocked / rate-limited /
    failed request counts.

Charts:
    results/response_time_chart.png   latency histogram with p50/p95 markers
    results/throughput_chart.png      request-outcome breakdown

NOTE: latency and throughput are wall-clock measurements and therefore vary
between machines and runs; they are not seeded. The request *workload
composition* is seeded and reproducible.

Run:  python -m src.load_test            (defaults: 2000 requests, 20 workers)
      python -m src.load_test 4000 40
"""
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Allow running both as a module and as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

SEED = config.RANDOM_SEED

# Workload mix: (label, weight, request-builder). Builders return a dict of
# httpx request kwargs plus the expected outcome category.
USER = {"Authorization": "Bearer valid_user_token"}
INVALID = {"Authorization": "Bearer invalid_token"}


def _build_workload(n: int, rng: np.random.Generator) -> list:
    """Deterministic list of (category, method, path, headers, body)."""
    specs = [
        # (category, weight, method, path, headers_kind, body)
        ("allowed_auth", 0.35, "GET", "/users", "user", None),
        ("allowed_auth", 0.20, "GET", "/orders", "user", None),
        ("allowed_public", 0.15, "GET", "/public", "none", None),
        ("blocked_injection", 0.10, "POST", "/payments", "user", "{\"q\": \"' OR 1=1 --\"}"),
        ("blocked_auth", 0.08, "GET", "/users", "invalid", None),
        ("blocked_role", 0.07, "GET", "/admin", "user", None),
        ("rate_limited", 0.05, "GET", "/public", "burst", None),
    ]
    categories = [s for s in specs if s[1] > 0]
    weights = np.array([s[1] for s in categories])
    weights = weights / weights.sum()
    choices = rng.choice(len(categories), size=n, p=weights)

    workload = []
    for idx in choices:
        cat, _, method, path, hkind, body = categories[idx]
        if hkind == "user":
            headers = dict(USER)
        elif hkind == "invalid":
            headers = dict(INVALID)
        elif hkind == "burst":
            # All burst requests share ONE client id -> triggers rate limiting.
            headers = {"X-Forwarded-For": "203.0.113.250"}
        else:
            headers = {}
        # Spread non-burst clients across many IPs so normal traffic is not
        # itself rate-limited.
        if hkind != "burst":
            headers = dict(headers)
            headers["X-Forwarded-For"] = f"10.0.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
        workload.append((cat, method, path, headers, body))
    return workload


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int) -> subprocess.Popen:
    """Launch uvicorn in a SEPARATE process.

    Running the server out-of-process is essential for a meaningful load
    test: an in-process server would share the GIL with the client threads,
    serialising all work and reporting contention rather than true gateway
    overhead. A separate process lets the OS schedule server and clients on
    different cores.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.main:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "error"],
        cwd=str(config.PROJECT_ROOT),
    )
    # Health-check until the gateway answers (or the process dies).
    deadline = 30
    for _ in range(deadline * 10):
        if proc.poll() is not None:
            raise RuntimeError("uvicorn subprocess exited during startup")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/public", timeout=1.0).status_code:
                return proc
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("uvicorn subprocess did not become ready in time")


def _categorize(expected: str, status: int) -> str:
    """Map an HTTP status to an outcome bucket."""
    if status == 429:
        return "rate_limited"
    if status in (400, 401, 403):
        return "blocked"
    if 200 <= status < 300:
        return "allowed"
    return "failed"


def run_load_test(n_requests: int = 2000, concurrency: int = 20) -> dict:
    rng = np.random.default_rng(SEED)
    workload = _build_workload(n_requests, rng)

    port = _free_port()
    proc = _start_server(port)
    base = f"http://127.0.0.1:{port}"

    latencies = np.zeros(n_requests)
    outcomes = [""] * n_requests

    # One shared, connection-pooled client (httpx.Client is safe to use from
    # multiple threads); avoids per-request connection setup overhead.
    limits = httpx.Limits(max_connections=concurrency * 2,
                          max_keepalive_connections=concurrency)
    client = httpx.Client(timeout=10.0, limits=limits)

    def _do(i_spec):
        i, (cat, method, path, headers, body) = i_spec
        t0 = time.perf_counter()
        try:
            resp = client.request(method, base + path, headers=headers,
                                  content=body)
            status = resp.status_code
        except Exception:
            status = 0
        latencies[i] = (time.perf_counter() - t0) * 1000.0
        outcomes[i] = _categorize(cat, status)

    try:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(_do, enumerate(workload)))
        duration = time.perf_counter() - start
    finally:
        client.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    counts = {k: outcomes.count(k) for k in ("allowed", "blocked",
                                             "rate_limited", "failed")}
    summary = {
        "total_requests": n_requests,
        "concurrency": concurrency,
        "duration_s": round(duration, 3),
        "throughput_req_s": round(n_requests / duration, 2),
        "avg_response_time_ms": round(float(np.mean(latencies)), 3),
        "p50_response_time_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_response_time_ms": round(float(np.percentile(latencies, 95)), 3),
        "max_response_time_ms": round(float(np.max(latencies)), 3),
        "allowed_requests": counts["allowed"],
        "blocked_requests": counts["blocked"],
        "rate_limited_requests": counts["rate_limited"],
        "failed_requests": counts["failed"],
    }
    return {"summary": summary, "latencies": latencies, "counts": counts}


def _save(result: dict, results_dir: Path = config.RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()]).to_csv(
        results_dir / "performance_results.csv", index=False
    )

    # Response-time histogram with p50/p95 markers.
    lat = result["latencies"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(lat, bins=50, color="#2b8cbe", alpha=0.8)
    for pct, style in [(50, "-"), (95, "--")]:
        v = np.percentile(lat, pct)
        ax.axvline(v, color="#e34a33", linestyle=style, linewidth=1.5,
                   label=f"p{pct} = {v:.1f} ms")
    ax.set_xlabel("Response time (ms)")
    ax.set_ylabel("Number of requests")
    ax.set_title(f"Gateway Response Time Distribution "
                 f"({summary['total_requests']} requests, "
                 f"{summary['concurrency']} concurrent)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "response_time_chart.png", dpi=150)
    plt.close(fig)

    # Outcome breakdown + throughput annotation.
    counts = result["counts"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["allowed", "blocked", "rate_limited", "failed"]
    colors = ["#82c082", "#f0a868", "#e34a33", "#999999"]
    values = [counts[k] for k in labels]
    bars = ax.bar(labels, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of requests")
    ax.set_title(f"Request Outcomes  |  Throughput = "
                 f"{summary['throughput_req_s']} req/s, "
                 f"avg latency = {summary['avg_response_time_ms']} ms")
    fig.tight_layout()
    fig.savefig(results_dir / "throughput_chart.png", dpi=150)
    plt.close(fig)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    c = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print(f"Running load test: {n} requests, concurrency {c} ...")
    result = run_load_test(n, c)
    _save(result)
    print("\nPerformance results:")
    for k, v in result["summary"].items():
        print(f"  {k:<24} {v}")
    print(f"\nSaved: {config.RESULTS_DIR / 'performance_results.csv'}")
    print(f"Saved: {config.RESULTS_DIR / 'response_time_chart.png'}")
    print(f"Saved: {config.RESULTS_DIR / 'throughput_chart.png'}")


if __name__ == "__main__":
    main()
