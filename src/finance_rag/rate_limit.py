"""Tiny in-memory per-client rate limiter (single-instance deployments)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


def client_ip(request) -> str:
    """Best-effort client address behind a reverse proxy (HF Spaces, Cloud Run)."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class SlidingWindowLimiter:
    def __init__(self, per_minute: int, per_day: int, clock=time.monotonic):
        self.per_minute, self.per_day, self.clock = per_minute, per_day, clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> int:
        """Record a hit and return 0 if allowed, else seconds to wait."""
        now = self.clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= 86400:
                hits.popleft()
            last_minute = [t for t in hits if now - t < 60]
            if self.per_minute and len(last_minute) >= self.per_minute:
                return max(1, int(60 - (now - last_minute[0])) + 1)
            if self.per_day and len(hits) >= self.per_day:
                return max(1, int(86400 - (now - hits[0])) + 1)
            hits.append(now)
            return 0
