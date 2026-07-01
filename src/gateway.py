"""API Gateway security middleware - the heart of the proposed model.

Every request passes through this pipeline before reaching a backend
endpoint:

    1. Authentication  (simulated JWT validation)          -> 401
    2. Authorization   (role-based endpoint access control) -> 403
    3. Rate limiting   (sliding window per client)          -> 429
    4. Request filter  (attack-signature inspection)        -> 400
    5. Risk engine     (heuristics + ML anomaly probability)
         allow    -> forward to backend
         flag     -> forward, but log as suspicious
         throttle -> 429 (additional verification required)
         block    -> 403
    6. Logging + telemetry update (feeds future risk decisions)

The pipeline is implemented as a Starlette/FastAPI HTTP middleware so the
backend endpoints stay completely free of security code.
"""
import time
from typing import Optional
from urllib.parse import unquote_plus

from fastapi import Request
from fastapi.responses import JSONResponse

from src import auth, config
from src.logger import SecurityLogger, TrafficStore
from src.rate_limiter import SlidingWindowRateLimiter
from src.request_filter import inspect_request
from src.risk_engine import RiskEngine

# Module-level singletons: shared state of the running gateway.
security_logger = SecurityLogger()
traffic_store = TrafficStore()
rate_limiter = SlidingWindowRateLimiter()
risk_engine = RiskEngine()


def reset_state() -> None:
    """Clear runtime state (used by the test suite for isolation)."""
    traffic_store.reset()
    rate_limiter.reset()


def _client_key(request: Request) -> str:
    """Identify the caller.

    The first X-Forwarded-For entry is honoured when present (standard
    behaviour behind a trusted reverse proxy; also used by the load-test
    harness to simulate distinct clients). Only valid when the gateway is
    deployed behind a proxy that overwrites the header - see the threat
    model (docs/THREAT_MODEL.md) for the spoofing caveat.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _reject(status_code: int, error: str, detail: str,
            risk_score: Optional[float] = None) -> JSONResponse:
    payload = {"error": error, "detail": detail}
    if risk_score is not None:
        payload["risk_score"] = risk_score
    return JSONResponse(status_code=status_code, content=payload)


async def gateway_middleware(request: Request, call_next):
    """The full security pipeline executed for every incoming request."""
    path = request.url.path
    if path in config.BYPASS_PATHS:
        return await call_next(request)

    start = time.perf_counter()
    client = _client_key(request)
    method = request.method
    body_bytes = await request.body()

    def finish(response, failed_auth: bool = False,
               risk_score: float = 0.0, decision: str = "n/a"):
        """Record telemetry + access log and return the response."""
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        traffic_store.record(client, elapsed_ms, failed_auth=failed_auth)
        security_logger.log_request(
            client, method, path, response.status_code, risk_score, decision
        )
        return response

    # ------------------------------------------------------------------
    # 1) Authentication (simulated JWT validation)
    # ------------------------------------------------------------------
    token = auth.extract_bearer_token(request.headers.get("authorization"))
    auth_result = auth.authenticate(token)

    if auth.endpoint_requires_auth(path) and not auth_result.authenticated:
        security_logger.log_security_event(
            "auth_failure", client, f"{auth_result.error} on {path}"
        )
        return finish(
            _reject(401, "unauthorized", f"Authentication failed: {auth_result.error}"),
            failed_auth=True, decision="reject_auth",
        )

    # ------------------------------------------------------------------
    # 2) Authorization (role-based access control)
    # ------------------------------------------------------------------
    if not auth.authorize(auth_result.role, path):
        security_logger.log_security_event(
            "forbidden", client, f"role={auth_result.role} tried {path}"
        )
        return finish(
            _reject(403, "forbidden", f"Role '{auth_result.role}' may not access {path}"),
            decision="reject_role",
        )

    # ------------------------------------------------------------------
    # 3) Rate limiting (hard cap per client)
    # ------------------------------------------------------------------
    if not rate_limiter.allow(client):
        security_logger.log_security_event(
            "rate_limit", client,
            f"more than {rate_limiter.max_requests} requests "
            f"in {rate_limiter.window_seconds}s",
        )
        return finish(
            _reject(429, "rate_limited", "Too many requests, slow down"),
            decision="reject_rate",
        )

    # ------------------------------------------------------------------
    # 4) Request filtering (attack signatures in path/query/body)
    # ------------------------------------------------------------------
    body_text = body_bytes.decode("utf-8", errors="replace")
    # URL-decode the query string so encoded attack payloads are visible.
    query_text = unquote_plus(str(request.url.query))
    filter_result = inspect_request(path, query_text, body_text)
    if filter_result.suspicious:
        security_logger.log_security_event(
            "filter_block", client,
            f"{filter_result.reason} in {filter_result.location} of {path}",
        )
        return finish(
            _reject(400, "malicious_payload",
                    f"Request blocked: {filter_result.reason} detected"),
            decision="reject_filter",
        )

    # ------------------------------------------------------------------
    # 5) Intelligent risk engine (heuristics + ML anomaly probability)
    # ------------------------------------------------------------------
    rpm, failed_auth_count, avg_rt = traffic_store.snapshot(client)
    features = {
        "requests_per_minute": rpm + 1,  # include the current request
        "failed_auth_count": failed_auth_count,
        "token_valid": 1 if (auth_result.authenticated or token is None) else 0,
        "endpoint_sensitivity": config.ENDPOINT_SENSITIVITY.get(
            path, config.DEFAULT_SENSITIVITY
        ),
        "payload_suspicious": 0,  # suspicious payloads were already rejected
        "request_size_bytes": len(body_bytes),
        "response_time_ms": avg_rt,  # client's recent behaviour profile
    }
    assessment = risk_engine.score(features)

    if assessment.decision == "block":
        security_logger.log_security_event(
            "risk_block", client, f"risk={assessment.score} on {path}"
        )
        return finish(
            _reject(403, "high_risk_blocked",
                    "Request blocked by the intelligent risk engine",
                    risk_score=assessment.score),
            risk_score=assessment.score, decision="block",
        )

    if assessment.decision == "throttle":
        security_logger.log_security_event(
            "risk_throttle", client, f"risk={assessment.score} on {path}"
        )
        return finish(
            _reject(429, "throttled",
                    "Elevated risk: additional verification required",
                    risk_score=assessment.score),
            risk_score=assessment.score, decision="throttle",
        )

    if assessment.decision == "flag":
        security_logger.log_security_event(
            "risk_flag", client, f"risk={assessment.score} on {path}"
        )

    # ------------------------------------------------------------------
    # 6) Forward to the backend endpoint
    # ------------------------------------------------------------------
    response = await call_next(request)
    response.headers["X-Risk-Score"] = str(assessment.score)
    response.headers["X-Risk-Decision"] = assessment.decision
    return finish(response, risk_score=assessment.score,
                  decision=assessment.decision)
