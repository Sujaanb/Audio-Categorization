"""
Rate limiting and metrics tracking for the Voice Detection API.
Uses a simple in-memory counter-based approach for single-instance deployments.
Can be extended to use Redis for distributed deployments.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RateLimitCounter:
    """Tracks request count and timestamp for a single API key."""

    requests_in_window: int = 0
    window_start_time: float = field(default_factory=time.time)
    lock: Lock = field(default_factory=Lock)

    def reset_if_expired(self) -> None:
        """Reset counter if the time window has expired."""
        elapsed = time.time() - self.window_start_time
        if elapsed >= settings.RATE_LIMIT_PERIOD_SECONDS:
            self.requests_in_window = 0
            self.window_start_time = time.time()

    def increment_and_check(self) -> tuple[int, bool]:
        """
        Increment request count and check if rate limit exceeded.

        Returns:
            Tuple of (current_count, is_limit_exceeded)
        """
        with self.lock:
            self.reset_if_expired()
            self.requests_in_window += 1
            is_exceeded = self.requests_in_window > settings.RATE_LIMIT_REQUESTS
            return self.requests_in_window, is_exceeded

    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        with self.lock:
            self.reset_if_expired()
            return max(0, settings.RATE_LIMIT_REQUESTS - self.requests_in_window)

    def get_reset_time(self) -> int:
        """Get Unix timestamp when rate limit window resets."""
        with self.lock:
            return int(self.window_start_time + settings.RATE_LIMIT_PERIOD_SECONDS)


@dataclass
class ApiKeyMetrics:
    """Aggregated metrics for a single API key."""

    total_requests: int = 0
    success_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    languages: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    lock: Lock = field(default_factory=Lock)

    def record_request(
        self, language: str, latency_ms: float, success: bool
    ) -> None:
        """Record a request in metrics."""
        with self.lock:
            self.total_requests += 1
            if success:
                self.success_requests += 1
            else:
                self.failed_requests += 1
            self.total_latency_ms += latency_ms
            self.languages[language] += 1

    def get_stats(self) -> dict:
        """Get current metrics as a dictionary."""
        with self.lock:
            avg_latency = (
                self.total_latency_ms / self.total_requests
                if self.total_requests > 0
                else 0
            )
            success_rate = (
                (self.success_requests / self.total_requests)
                if self.total_requests > 0
                else 0
            )

            return {
                "total_requests": self.total_requests,
                "success_requests": self.success_requests,
                "failed_requests": self.failed_requests,
                "success_rate": round(success_rate, 4),
                "average_latency_ms": round(avg_latency, 2),
                "languages": dict(self.languages),
            }


class RateLimiter:
    """
    Rate limiter that tracks requests per API key.
    Thread-safe, in-memory implementation.
    """

    def __init__(self):
        self.limiters: dict[str, RateLimitCounter] = defaultdict(RateLimitCounter)
        self.metrics: dict[str, ApiKeyMetrics] = defaultdict(ApiKeyMetrics)
        self.lock = Lock()

    def check_rate_limit(self, api_key: str) -> tuple[bool, int, int]:
        """
        Check if request is allowed under rate limit.

        Args:
            api_key: The API key to check

        Returns:
            Tuple of (allowed: bool, current_count: int, limit: int)
        """
        api_key_lower = api_key.lower()
        counter = self.limiters[api_key_lower]
        current_count, is_exceeded = counter.increment_and_check()

        if is_exceeded:
            logger.warning(
                f"Rate limit exceeded for API key (last 4 chars: ...{api_key_lower[-4:]}). "
                f"Current: {current_count}, Limit: {settings.RATE_LIMIT_REQUESTS}"
            )
            return False, current_count, settings.RATE_LIMIT_REQUESTS

        return True, current_count, settings.RATE_LIMIT_REQUESTS

    def get_rate_limit_headers(self, api_key: str) -> dict[str, str]:
        """
        Get rate limit headers for the response.
        Follows RateLimit-* specification.
        """
        api_key_lower = api_key.lower()
        counter = self.limiters[api_key_lower]

        return {
            "RateLimit-Limit": str(settings.RATE_LIMIT_REQUESTS),
            "RateLimit-Remaining": str(counter.get_remaining()),
            "RateLimit-Reset": str(counter.get_reset_time()),
        }

    def record_request(
        self,
        api_key: str,
        language: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Record request metrics for an API key."""
        api_key_lower = api_key.lower()
        metrics = self.metrics[api_key_lower]
        metrics.record_request(language, latency_ms, success)

    def get_metrics(self, api_key: str) -> dict:
        """Get metrics for a specific API key."""
        api_key_lower = api_key.lower()
        return self.metrics[api_key_lower].get_stats()

    def get_all_metrics(self) -> dict:
        """Get metrics for all API keys (admin only)."""
        all_metrics = {}
        with self.lock:
            for api_key, metrics in self.metrics.items():
                all_metrics[f"...{api_key[-4:]}"] = metrics.get_stats()
        return all_metrics


# Global rate limiter instance
rate_limiter = RateLimiter()
