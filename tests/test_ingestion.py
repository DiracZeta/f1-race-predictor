"""Tests for the Jolpica-F1 ingestion client.

Run with:  pytest -q
These tests use a fake session, so they never touch the network.
"""

import tempfile

import pytest

from src.ingestion.jolpica_client import JolpicaClient, JolpicaError, RateLimiter


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    """Returns queued responses and counts how many GETs were issued."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


@pytest.fixture
def cache_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def _client(cache_dir, session, **kwargs):
    kwargs.setdefault("sleep", lambda _s: None)  # never sleep in tests
    return JolpicaClient(cache_dir=cache_dir, session=session, **kwargs)


def _page(total, limit, offset, n_items):
    items = [{"i": offset + k} for k in range(n_items)]
    return FakeResponse(
        200,
        {"MRData": {"total": str(total), "limit": str(limit),
                    "offset": str(offset), "items": items}},
    )


def test_identical_fetch_is_served_from_cache(cache_dir):
    session = FakeSession([FakeResponse(200, {"MRData": {"hello": "world"}})])
    client = _client(cache_dir, session)
    first = client.fetch("2023/results")
    second = client.fetch("2023/results")
    assert first == second
    assert session.calls == 1  # second call hit the cache


def test_current_endpoints_are_not_cached(cache_dir):
    session = FakeSession([
        FakeResponse(200, {"MRData": {"x": 1}}),
        FakeResponse(200, {"MRData": {"x": 2}}),
    ])
    client = _client(cache_dir, session)
    assert client.fetch("current/last/results") != client.fetch("current/last/results")
    assert session.calls == 2


def test_retries_then_succeeds(cache_dir):
    session = FakeSession([FakeResponse(503), FakeResponse(200, {"MRData": {"ok": True}})])
    slept = []
    client = _client(cache_dir, session, sleep=slept.append)
    assert client.fetch("2022/results") == {"MRData": {"ok": True}}
    assert len(slept) >= 1  # backed off before retrying


def test_429_honors_retry_after(cache_dir):
    session = FakeSession([
        FakeResponse(429, headers={"Retry-After": "7"}),
        FakeResponse(200, {"MRData": {"ok": True}}),
    ])
    slept = []
    client = _client(cache_dir, session, sleep=slept.append)
    client.fetch("2021/results")
    assert 7.0 in slept


def test_non_retryable_status_fails_fast(cache_dir):
    session = FakeSession([FakeResponse(404, text="nope")])
    client = _client(cache_dir, session)
    with pytest.raises(JolpicaError):
        client.fetch("9999/results")
    assert session.calls == 1  # a 404 is not retried


def test_gives_up_after_max_retries(cache_dir):
    session = FakeSession([FakeResponse(500)] * 10)
    client = _client(cache_dir, session, max_retries=3)
    with pytest.raises(JolpicaError):
        client.fetch("2020/results")
    assert session.calls == 4  # initial try + 3 retries


def test_fetch_all_paginates(cache_dir):
    session = FakeSession([_page(5, 2, 0, 2), _page(5, 2, 2, 2), _page(5, 2, 4, 1)])
    client = _client(cache_dir, session)
    pages = client.fetch_all("2019/results", page_size=2, use_cache=False)
    assert len(pages) == 3
    assert sum(len(p["MRData"]["items"]) for p in pages) == 5


def test_rate_limiter_throttles_when_budget_spent():
    now = [0.0]
    waits = []

    def fake_sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2, period_seconds=100, clock=lambda: now[0], sleep=fake_sleep)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()  # third must wait for the window to free up
    assert len(waits) == 1 and waits[0] > 0
