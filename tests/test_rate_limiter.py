"""Tests for the sliding-window rate limiter (with an injectable clock)."""
from src.rate_limiter import SlidingWindowRateLimiter


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_limiter(max_requests=5, window=60):
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(
        max_requests=max_requests, window_seconds=window, time_func=clock
    )
    return limiter, clock


def test_requests_under_limit_are_allowed():
    limiter, _ = make_limiter(max_requests=5)
    assert all(limiter.allow("client_a") for _ in range(5))


def test_request_over_limit_is_rejected():
    limiter, _ = make_limiter(max_requests=5)
    for _ in range(5):
        assert limiter.allow("client_a") is True
    assert limiter.allow("client_a") is False


def test_limit_resets_after_window_expires():
    limiter, clock = make_limiter(max_requests=3, window=60)
    for _ in range(3):
        limiter.allow("client_a")
    assert limiter.allow("client_a") is False
    clock.advance(61)
    assert limiter.allow("client_a") is True


def test_clients_are_limited_independently():
    limiter, _ = make_limiter(max_requests=2)
    assert limiter.allow("client_a") and limiter.allow("client_a")
    assert limiter.allow("client_a") is False
    # A different client is unaffected.
    assert limiter.allow("client_b") is True


def test_reset_clears_all_state():
    limiter, _ = make_limiter(max_requests=1)
    limiter.allow("client_a")
    assert limiter.allow("client_a") is False
    limiter.reset()
    assert limiter.allow("client_a") is True
