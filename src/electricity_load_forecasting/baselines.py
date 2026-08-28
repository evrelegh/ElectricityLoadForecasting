# ══════════════════════════════════════════════════════════════════════════
# baselines · civil-time persistence forecasts                       v0.4.0
# provides ▸ CIVIL_KEY · civil_occurrence · seasonal_naive · daily_origin
#            origin_violations
# requires ▸ numpy · pandas
# ══════════════════════════════════════════════════════════════════════════
"""Persistence baselines matched on the Belgian civil clock.

A shift of 96 or 672 rows on the UTC axis is wrong twice a year: the civil day
holds 92 or 100 quarter hours at the transitions, so a fixed row offset silently
misaligns the clock time it claims to reuse. Matching is therefore done on the
civil key (date, quarter of day, occurrence) instead.

Two DST cases have to be decided rather than avoided:

* **Spring forward.** 02:00-02:45 civil does not exist on the transition day.
  When that day is the source, the corresponding quarter hours have no
  counterpart and the forecast is UNAVAILABLE — reported, never interpolated.
* **Fall back.** 02:00-02:45 civil occurs twice. Occurrence is tracked
  explicitly: the first occurrence of a target maps to the first occurrence of
  the source, the second to the second. When the target repeats an hour that
  the source does not, the source's single occurrence is reused for both; the
  reverse case simply leaves the source's second occurrence unused.

Availability of a forecast is kept separate from presence of the target: a
baseline can be unavailable where the target exists, and the target can be
missing where the baseline is perfectly well defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CIVIL_KEY = ["civil_date", "qod", "occ"]


def civil_occurrence(frame: pd.DataFrame) -> pd.Series:
    """Index of repeated civil quarter hours: 0 everywhere except the second
    pass through the repeated hour on the fall-back day, where it is 1."""
    return frame.groupby(["civil_date", "qod"], sort=False).cumcount()


def seasonal_naive(frame: pd.DataFrame, lag_days: int, y: str = "y",
                   time: str = "datetime") -> pd.DataFrame:
    """Forecast each civil quarter hour from the same clock time `lag_days` back.

    Returns a frame aligned to `frame.index` with:
      yhat        the forecast, NaN where unavailable
      source_time the timestamp of the observation used, NaT where unavailable
      available   whether a source observation existed and was not missing
    """
    if lag_days < 1:
        raise ValueError(f"lag_days must be a positive number of days, got {lag_days}")
    for c in ("civil_date", "qod", y, time):
        if c not in frame.columns:
            raise KeyError(f"frame is missing required column {c!r}")

    base = frame[["civil_date", "qod", y, time]].copy()
    base["occ"] = civil_occurrence(base)

    # Source table keyed by the civil slot it will be used to forecast.
    src = base.rename(columns={y: "yhat", time: "source_time"})
    src = src.assign(civil_date=src["civil_date"] + pd.Timedelta(days=lag_days))

    out = base[CIVIL_KEY].merge(src[CIVIL_KEY + ["yhat", "source_time"]],
                                on=CIVIL_KEY, how="left")

    # Fall-back policy: a repeated target hour whose source day has only one
    # occurrence reuses that single observation.
    gap = out["source_time"].isna() & (out["occ"] > 0)
    if gap.any():
        first = src[src["occ"] == 0][["civil_date", "qod", "yhat", "source_time"]]
        fill = (out.loc[gap, ["civil_date", "qod"]]
                   .merge(first, on=["civil_date", "qod"], how="left"))
        out.loc[gap, "yhat"] = fill["yhat"].to_numpy()
        out.loc[gap, "source_time"] = fill["source_time"].to_numpy()

    out.index = frame.index
    out["available"] = out["yhat"].notna()
    return out[["yhat", "source_time", "available"]]


def daily_origin(civil_date, origin_civil_hour: float = 18.0, lead_days: int = 1,
                 tz: str = "Europe/Brussels") -> pd.Series:
    """Forecast origin for each target civil day, as a UTC instant.

    `origin_civil_hour` may be fractional (8.75 for 08:45) so that competing
    published issue times can be compared without changing the caller.
    """
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(civil_date).to_numpy()))
    local = (d - pd.Timedelta(days=lead_days)) + pd.Timedelta(hours=float(origin_civil_hour))
    utc = local.tz_localize(tz, nonexistent="shift_forward",
                            ambiguous=True).tz_convert("UTC")
    return pd.Series(utc)


def origin_violations(source_time, origin) -> pd.DataFrame:
    """Rows whose source observation postdates the forecast origin.

    An empty frame is the only acceptable result for a leakage-safe baseline;
    a non-empty one names the offending timestamps rather than summarising them.
    """
    s = pd.Series(pd.to_datetime(source_time, utc=True)).reset_index(drop=True)
    o = pd.Series(pd.to_datetime(origin, utc=True)).reset_index(drop=True)
    bad = s.notna() & (s > o)
    return pd.DataFrame({"source_time": s[bad], "origin": o[bad],
                         "late_by": s[bad] - o[bad]})
