"""Sliding-window rate limiter used by the gateway as a hard traffic cap.

The limiter keeps a per-client deque of request timestamps. A request is
allowed only while the number of requests inside the current window stays
below the configured maximum. The clock is injectable so tests can advance
time without sleeping.
"""
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from src import config


class SlidingWindowRateLimiter:
    def __init__(
        self,
        max_requests: int = config.RATE_LIMIT_MAX_REQUESTS,
        window_seconds: float = config.RATE_LIMIT_WINDOW_SECONDS,
        time_func: Callable[[], float] = time.monotonic,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._time = time_func
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def _evict_old(self, key: str, now: float) -> None:
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window_seconds:
            hits.popleft()

    def allow(self, key: str) -> bool:
        """Register one request for `key`; return False if over the limit."""
        now = self._time()
        self._evict_old(key, now)
        if len(self._hits[key]) >= self.max_requests:
            return False
        self._hits[key].append(now)
        return True

    def current_usage(self, key: str) -> int:
        """Number of requests inside the current window for `key`."""
        self._evict_old(key, self._time())
        return len(self._hits[key])

    def reset(self) -> None:
        """Clear all state (used between tests / demo runs)."""
        self._hits.clear()
