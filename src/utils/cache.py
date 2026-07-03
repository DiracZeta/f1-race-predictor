"""Caching helpers for the ingestion layer.

Cache raw API responses on disk keyed by (endpoint, params) so re-runs only
fetch new races. This is what keeps the pipeline under the API rate limit.
"""


def cache_key(endpoint: str, params: dict | None) -> str:
    raise NotImplementedError
