"""Rate limiting and the answer cache.

Both protect the free LLM quotas of a public demo: the rate limiter stops one visitor from
using up the day's quota, and the cache answers repeated questions without any LLM call.
"""

import re
import threading
import time
from collections import OrderedDict, defaultdict, deque


class RateLimiter:
    """At most `limit` requests per `window` seconds for each key (an IP or a session)."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> float:
        """Record a request. Returns 0 if allowed, else the seconds to wait."""
        if self.limit <= 0:
            return 0.0
        now = self.clock()
        with self.lock:
            hits = self.hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return round(self.window - (now - hits[0]), 1)
            hits.append(now)
            return 0.0


def cache_key(question: str) -> str:
    """Questions that differ only in case, spacing or final punctuation share an answer."""
    return re.sub(r"\s+", " ", question.lower()).strip().rstrip("?.! ")


class AnswerCache:
    """Least-recently-used cache of answers, with an expiry time."""

    def __init__(self, size: int = 256, ttl: float = 24 * 3600, clock=time.monotonic):
        self.size, self.ttl, self.clock = size, ttl, clock
        self.items: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, question: str):
        key = cache_key(question)
        with self.lock:
            item = self.items.get(key)
            if item is None:
                return None
            stored, value = item
            if self.clock() - stored > self.ttl:
                del self.items[key]
                return None
            self.items.move_to_end(key)
            return value

    def put(self, question: str, value) -> None:
        if self.size <= 0:
            return
        with self.lock:
            self.items[cache_key(question)] = (self.clock(), value)
            self.items.move_to_end(cache_key(question))
            while len(self.items) > self.size:
                self.items.popitem(last=False)
