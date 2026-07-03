"""FastF1 loader for lap timing and telemetry.

FastF1 pulls historical data from Jolpica under the hood and adds detailed
lap-by-lap timing and telemetry. Enable its on-disk cache to avoid refetching.

TODO:
    - Configure fastf1.Cache.enable_cache(...)
    - Implement load_session_laps(season, round, session='R')
"""


def load_session_laps(season, rnd, session: str = "R"):
    """Return lap data for a given session as a DataFrame."""
    raise NotImplementedError
