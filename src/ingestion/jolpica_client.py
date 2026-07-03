"""Jolpica-F1 API client.

A small, production-minded client for the Jolpica-F1 API (the drop-in successor
to the deprecated Ergast API). It fetches race results, qualifying, and standings
while respecting the API's rate limit (~200 req/hr).

Design goals
------------
- **Rate limiting**: a sliding-window limiter enforces a per-hour request budget
  so re-runs and backfills never get throttled.
- **Retries**: transient failures (5xx, 429, connection errors) are retried with
  exponential backoff + jitter; ``Retry-After`` is honored on 429.
- **Caching**: raw responses are cached on disk keyed by the full URL, so repeat
  runs hit the cache instead of the network. Endpoints whose data is still
  changing (``current`` / ``last``) are *not* cached by default.
- **Pagination**: the Ergast/Jolpica ``limit``/``offset`` scheme is handled
  transparently via :meth:`fetch_all`.

The client returns **raw JSON**. Transformation belongs downstream in the bronze
-> silver -> gold layers, not here.

Run it directly for a quick smoke test::

    python -m src.ingestion.jolpica_client --season current

In a notebook (Jupyter / Databricks), don't run this file as a script — the
``argparse`` block would try to parse the *kernel's* launch arguments (e.g.
``-f connection.json``) and fail. Import and call the client instead::

    from src.ingestion.jolpica_client import JolpicaClient

    client = JolpicaClient()
    data = client.get_race_results("current", "last")   # or e.g. ("2023", "1")
    races = data["MRData"]["RaceTable"]["Races"]

(The command-line block below uses ``parse_known_args`` so it degrades to the
defaults rather than crashing if it is run in a notebook anyway.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

# HTTP statuses worth retrying. 429 = rate limited; 5xx = transient server errors.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class JolpicaError(RuntimeError):
    """Raised when a request ultimately fails after all retries."""


class RateLimiter:
    """Sliding-window rate limiter.

    Allows at most ``max_requests`` within any ``period_seconds`` window.
    :meth:`acquire` blocks (sleeps) only when the budget is exhausted, then
    returns as soon as the oldest in-window request expires.

    The clock and sleep functions are injectable so the limiter is unit-testable
    without real time passing.
    """

    def __init__(
        self,
        max_requests: int,
        period_seconds: float = 3600.0,
        *,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self._clock = clock
        self._sleep = sleep
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        """Block until a request is permitted, then record it."""
        while True:
            now = self._clock()
            # Drop timestamps that have aged out of the window.
            while self._timestamps and now - self._timestamps[0] >= self.period_seconds:
                self._timestamps.popleft()

            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(now)
                return

            # Budget full: wait until the oldest request leaves the window.
            wait = self.period_seconds - (now - self._timestamps[0])
            logger.info("Rate limit reached; sleeping %.1fs", wait)
            self._sleep(max(wait, 0.0))


class JolpicaClient:
    """Client for the Jolpica-F1 (Ergast-compatible) API."""

    def __init__(
        self,
        base_url: str = "https://api.jolpi.ca/ergast/f1",
        *,
        requests_per_hour: int = 180,
        max_retries: int = 5,
        backoff_seconds: float = 2.0,
        cache_dir: str | Path = ".cache",
        timeout: float = 15.0,
        session: requests.Session | None = None,
        sleep=time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout = timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = session or requests.Session()
        self._sleep = sleep
        self._limiter = RateLimiter(requests_per_hour, sleep=sleep)

    @classmethod
    def from_config(cls, path: str | Path = "config/config.yaml") -> "JolpicaClient":
        """Build a client from ``config/config.yaml`` (requires PyYAML)."""
        import yaml  # imported lazily so the module imports without PyYAML

        cfg = yaml.safe_load(Path(path).read_text())
        api = cfg.get("api", {})
        ingestion = cfg.get("ingestion", {})
        return cls(
            base_url=api.get("base_url", "https://api.jolpi.ca/ergast/f1"),
            requests_per_hour=api.get("requests_per_hour", 180),
            max_retries=api.get("max_retries", 5),
            backoff_seconds=api.get("backoff_seconds", 2.0),
            cache_dir=ingestion.get("cache_dir", ".cache"),
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch(
        self,
        endpoint: str,
        params: dict | None = None,
        *,
        use_cache: bool | None = None,
    ) -> dict:
        """Fetch a single endpoint and return the parsed JSON.

        Parameters
        ----------
        endpoint:
            Path under the base URL, e.g. ``"2023/results"`` or
            ``"current/last/results"``. The ``.json`` suffix is added for you.
        params:
            Query parameters (e.g. ``{"limit": 100, "offset": 0}``).
        use_cache:
            Override caching for this call. When ``None`` (default), data that is
            still changing (``current`` / ``last`` endpoints) skips the cache and
            everything else is cached.
        """
        url = self._build_url(endpoint, params)
        cacheable = self._is_cacheable(endpoint) if use_cache is None else use_cache

        if cacheable:
            cached = self._read_cache(url)
            if cached is not None:
                logger.debug("Cache hit: %s", url)
                return cached

        payload = self._request_with_retries(url)

        if cacheable:
            self._write_cache(url, payload)
        return payload

    def fetch_all(
        self,
        endpoint: str,
        params: dict | None = None,
        *,
        page_size: int = 1000,
        use_cache: bool | None = None,
    ) -> list[dict]:
        """Fetch every page of a paginated endpoint.

        Returns the list of raw ``MRData`` page payloads. Iterates using the
        ``limit``/``offset``/``total`` fields the API returns (max page size 1000).
        """
        params = dict(params or {})
        params["limit"] = min(page_size, 1000)
        offset = 0
        pages: list[dict] = []

        while True:
            params["offset"] = offset
            payload = self.fetch(endpoint, params, use_cache=use_cache)
            pages.append(payload)

            mrdata = payload.get("MRData", {})
            total = int(mrdata.get("total", 0))
            limit = int(mrdata.get("limit", params["limit"]))
            offset += limit
            if offset >= total or limit == 0:
                break

        return pages

    # Convenience wrappers ------------------------------------------------
    def get_race_results(self, season: str | int, rnd: str | int | None = None) -> dict:
        return self.fetch(self._race_path(season, rnd, "results"))

    def get_qualifying(self, season: str | int, rnd: str | int | None = None) -> dict:
        return self.fetch(self._race_path(season, rnd, "qualifying"))

    def get_driver_standings(self, season: str | int, rnd: str | int | None = None) -> dict:
        return self.fetch(self._race_path(season, rnd, "driverStandings"))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _race_path(season, rnd, resource) -> str:
        return f"{season}/{rnd}/{resource}" if rnd is not None else f"{season}/{resource}"

    def _build_url(self, endpoint: str, params: dict | None) -> str:
        url = f"{self.base_url}/{endpoint.strip('/')}.json"
        if params:
            url = f"{url}?{urlencode(sorted(params.items()))}"
        return url

    @staticmethod
    def _is_cacheable(endpoint: str) -> bool:
        # 'current' and 'last' resolve to live data, so don't cache them.
        lowered = endpoint.lower()
        return "current" not in lowered and "last" not in lowered

    def _request_with_retries(self, url: str) -> dict:
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._limiter.acquire()
            try:
                resp = self._session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("Request error on %s (attempt %d): %s", url, attempt, exc)
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in RETRYABLE_STATUS:
                    last_exc = JolpicaError(f"HTTP {resp.status_code} for {url}")
                    self._sleep(self._retry_delay(attempt, resp))
                    continue
                # Non-retryable (e.g. 400/404): fail fast with context.
                raise JolpicaError(f"HTTP {resp.status_code} for {url}: {resp.text[:200]}")

            # Reached only on a connection-level error: back off and retry.
            if attempt < self.max_retries:
                self._sleep(self._retry_delay(attempt))

        raise JolpicaError(f"Failed after {self.max_retries} retries: {url}") from last_exc

    def _retry_delay(self, attempt: int, resp: requests.Response | None = None) -> float:
        """Exponential backoff with jitter; honor ``Retry-After`` on 429."""
        if resp is not None and resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                return float(retry_after)
        backoff = self.backoff_seconds * (2 ** attempt)
        return backoff + random.uniform(0, self.backoff_seconds)

    # Disk cache ----------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        key = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, url: str) -> dict | None:
        path = self._cache_path(url)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt cache entry, ignoring: %s", path)
            return None

    def _write_cache(self, url: str, payload: dict) -> None:
        try:
            self._cache_path(url).write_text(json.dumps(payload))
        except OSError as exc:
            logger.warning("Could not write cache for %s: %s", url, exc)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch F1 race results from Jolpica-F1.")
    parser.add_argument("--season", default="current", help="Season, e.g. 2023 or 'current'")
    parser.add_argument("--round", default="last", help="Round number or 'last'")
    # parse_known_args ignores args injected by notebook kernels (e.g. -f
    # connection.json), so this won't crash if run inside Jupyter/Databricks.
    args, _ = parser.parse_known_args()

    client = JolpicaClient()
    data = client.get_race_results(args.season, args.round)
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        print("No race data returned.")
        return

    race = races[0]
    results = race.get("Results", [])
    print(f"{race.get('raceName')} ({race.get('season')} round {race.get('round')})")
    for r in results[:3]:
        driver = r.get("Driver", {})
        name = f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip()
        print(f"  P{r.get('position')}: {name} ({r.get('Constructor', {}).get('name', '')})")


if __name__ == "__main__":
    _main()
