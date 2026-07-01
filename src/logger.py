"""Security logging and per-client traffic telemetry.

Two responsibilities:

1. ``SecurityLogger`` writes structured security events (allowed requests,
   auth failures, rate-limit violations, filtered payloads, risk decisions)
   to the console and to ``logs/gateway.log``.

2. ``TrafficStore`` keeps sliding-window statistics per client (requests per
   minute, failed authentications, average response time). The risk engine
   reads these live features at decision time - this is what makes the
   gateway *behaviour-aware* instead of judging every request in isolation.
"""
import logging
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Tuple

from src import config


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("api_gateway")
    if logger.handlers:  # already configured (module re-import, tests)
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        config.LOGS_DIR / "gateway.log", encoding="utf-8", delay=True
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class SecurityLogger:
    """Thin wrapper producing consistent, greppable security log lines."""

    def __init__(self):
        self._logger = _build_logger()

    def log_request(self, client: str, method: str, path: str,
                    status_code: int, risk_score: float, decision: str) -> None:
        self._logger.info(
            "REQUEST client=%s method=%s path=%s status=%s risk=%.1f decision=%s",
            client, method, path, status_code, risk_score, decision,
        )

    def log_security_event(self, event_type: str, client: str, detail: str) -> None:
        """event_type: auth_failure | forbidden | rate_limit | filter_block |
        risk_flag | risk_throttle | risk_block"""
        self._logger.warning(
            "SECURITY event=%s client=%s detail=%s", event_type, client, detail
        )


class TrafficStore:
    """Sliding-window request statistics per client (IP or client id)."""

    def __init__(self, time_func: Callable[[], float] = time.monotonic):
        self._time = time_func
        # timestamps of all requests seen per client
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)
        # timestamps of failed authentications per client
        self._failed_auth: Dict[str, Deque[float]] = defaultdict(deque)
        # (timestamp-free) recent response times per client
        self._response_times: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=20)
        )

    @staticmethod
    def _evict(series: Deque[float], now: float, window: float) -> None:
        while series and now - series[0] >= window:
            series.popleft()

    def record(self, client: str, response_time_ms: float,
               failed_auth: bool = False) -> None:
        """Record the outcome of one handled request."""
        now = self._time()
        self._requests[client].append(now)
        self._response_times[client].append(response_time_ms)
        if failed_auth:
            self._failed_auth[client].append(now)

    def requests_per_minute(self, client: str) -> int:
        now = self._time()
        self._evict(self._requests[client], now, 60.0)
        return len(self._requests[client])

    def failed_auth_count(self, client: str) -> int:
        now = self._time()
        self._evict(self._failed_auth[client], now, config.FAILED_AUTH_WINDOW_SECONDS)
        return len(self._failed_auth[client])

    def avg_response_time_ms(self, client: str) -> float:
        times = self._response_times[client]
        if not times:
            return config.DEFAULT_RESPONSE_TIME_MS
        return sum(times) / len(times)

    def snapshot(self, client: str) -> Tuple[int, int, float]:
        """(requests_per_minute, failed_auth_count, avg_response_time_ms)."""
        return (
            self.requests_per_minute(client),
            self.failed_auth_count(client),
            self.avg_response_time_ms(client),
        )

    def reset(self) -> None:
        self._requests.clear()
        self._failed_auth.clear()
        self._response_times.clear()
