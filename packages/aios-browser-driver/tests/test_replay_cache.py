"""The closed-grant replay cache: bounded LRU with recency on get and put."""

from __future__ import annotations

from aios_browser_driver.takeover.state import ReplayCache


def test_get_returns_the_stored_handback() -> None:
    cache = ReplayCache(maxlen=3)
    cache.put("g1", {"url": "https://a.test"})
    assert cache.get("g1") == {"url": "https://a.test"}
    assert cache.get("absent") is None


def test_evicts_the_least_recently_used_beyond_maxlen() -> None:
    cache = ReplayCache(maxlen=2)
    cache.put("g1", {"n": 1})
    cache.put("g2", {"n": 2})
    cache.put("g3", {"n": 3})  # evicts g1
    assert cache.get("g1") is None
    assert cache.get("g2") == {"n": 2}
    assert cache.get("g3") == {"n": 3}


def test_eviction_is_insertion_order_not_access_order() -> None:
    # FIFO: a get does NOT refresh recency (grant ids are single-use, so
    # access-recency would be dead bookkeeping).
    cache = ReplayCache(maxlen=2)
    cache.put("g1", {"n": 1})
    cache.put("g2", {"n": 2})
    assert cache.get("g1") == {"n": 1}  # reading g1 does not save it
    cache.put("g3", {"n": 3})  # evicts g1 (oldest insertion), regardless of the read
    assert cache.get("g1") is None
    assert cache.get("g2") == {"n": 2}
    assert cache.get("g3") == {"n": 3}
