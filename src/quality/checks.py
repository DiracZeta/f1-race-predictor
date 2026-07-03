"""Data quality gates that run BETWEEN layers.

A bad upstream batch should never get promoted. Checks to implement:
    - Row count / freshness (did the latest race land?)
    - Schema + type enforcement (silver)
    - Null / range / uniqueness assertions on key columns
    - Referential integrity (results map to known drivers and circuits)

Decide the failure mode: fail the run, or quarantine the bad batch.
"""


def validate_silver(df) -> bool:
    raise NotImplementedError
