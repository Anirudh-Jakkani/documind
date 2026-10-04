from documind.limits import AnswerCache, RateLimiter, cache_key


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_rate_limiter_allows_limit_then_asks_to_wait():
    clock = Clock()
    limiter = RateLimiter(limit=2, window=60, clock=clock)
    assert limiter.check("ip-1") == 0 and limiter.check("ip-1") == 0
    clock.now = 20
    assert limiter.check("ip-1") == 40  # oldest request leaves the window at t=60
    assert limiter.check("ip-2") == 0  # other visitors are independent
    clock.now = 61
    assert limiter.check("ip-1") == 0


def test_rate_limiter_off_when_limit_is_zero():
    limiter = RateLimiter(limit=0)
    assert all(limiter.check("x") == 0 for _ in range(100))


def test_cache_key_ignores_case_spacing_and_final_punctuation():
    assert cache_key("  What is the LRS   limit? ") == cache_key("what is the lrs limit")


def test_answer_cache_hit_expiry_and_lru_eviction():
    clock = Clock()
    cache = AnswerCache(size=2, ttl=100, clock=clock)
    cache.put("a?", "A")
    cache.put("b?", "B")
    assert cache.get("A") == "A"  # same question, different case
    cache.put("c?", "C")  # evicts b, the least recently used
    assert cache.get("b") is None and cache.get("c") == "C"
    clock.now = 101
    assert cache.get("a") is None  # expired


def test_answer_cache_off_when_size_is_zero():
    cache = AnswerCache(size=0)
    cache.put("q", "A")
    assert cache.get("q") is None
