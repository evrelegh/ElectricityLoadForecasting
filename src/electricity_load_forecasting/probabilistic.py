"""Leakage-safe empirical residual intervals for day-ahead load forecasts.

The point forecast is supplied by the caller.  Historical realised forecast
errors are pooled over a fixed elapsed-time window and conditioned on a civil
calendar group (civil hour by default).  For a target civil day D, only errors
whose actual load was already observed by the stated D-1 forecast origin may
enter the empirical quantiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import daily_origin


def _aligned_series(frame: pd.DataFrame, values, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        if not values.index.equals(frame.index):
            raise ValueError(f"{name} index must match frame.index exactly")
        return values.astype(float)
    arr = np.asarray(values, dtype=float)
    if len(arr) != len(frame):
        raise ValueError(f"{name} must have length {len(frame)}, got {len(arr)}")
    return pd.Series(arr, index=frame.index, dtype=float)


def conditional_empirical_intervals(
    frame: pd.DataFrame,
    point_forecast,
    targets=None,
    *,
    quantiles: tuple[float, float] = (0.10, 0.90),
    window_days: float = 56.0,
    min_rows: int = 80,
    origin_civil_hour: float = 18.0,
    lead_days: int = 1,
    tz: str = "Europe/Brussels",
    y: str = "y",
    time: str = "datetime",
    date_col: str = "civil_date",
    group_col: str = "hour",
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Empirical conditional P10/P90-style intervals around a frozen point forecast.

    Returns a frame aligned to ``frame.index`` with columns ``q_lo``, ``q_hi``,
    ``n_history``, ``history_start``, ``history_end``, ``origin`` and
    ``available``.  The grouping variable is read from each target row and is
    normally Belgian civil hour.  The history window is elapsed time, not a
    count of civil days, so DST transitions are unambiguous.
    """
    for c in (date_col, group_col, y, time):
        if c not in frame.columns:
            raise KeyError(f"frame is missing required column {c!r}")
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1")
    if len(quantiles) != 2 or not (0.0 < quantiles[0] < quantiles[1] < 1.0):
        raise ValueError("quantiles must be two ordered probabilities in (0, 1)")

    point = _aligned_series(frame, point_forecast, "point_forecast")
    t = pd.to_datetime(frame[time], utc=True)
    residual = pd.to_numeric(frame[y], errors="coerce") - point

    days = (
        sorted(pd.unique(pd.to_datetime(frame[date_col])))
        if targets is None
        else sorted(pd.unique(pd.to_datetime(pd.Series(targets))))
    )

    nat = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    out = pd.DataFrame(
        {
            "q_lo": np.nan,
            "q_hi": np.nan,
            "n_history": np.nan,
            "history_start": nat,
            "history_end": nat.copy(),
            "origin": nat.copy(),
        },
        index=frame.index,
    )

    n_days = len(days)
    for day_no, day in enumerate(days, start=1):
        if progress_every and (
            day_no == 1 or day_no % progress_every == 0 or day_no == n_days
        ):
            print(f"Processing target days: {day_no} / {n_days}")

        rows = frame.index[pd.to_datetime(frame[date_col]) == pd.Timestamp(day)]
        if len(rows) == 0:
            continue

        origin = daily_origin(
            [day], origin_civil_hour=origin_civil_hour,
            lead_days=lead_days, tz=tz
        ).iloc[0]
        history_start = origin - pd.Timedelta(days=float(window_days))
        hist = (
            (t <= origin)
            & (t > history_start)
            & residual.notna()
        )

        history = frame.loc[hist, [group_col]].copy()
        history["residual"] = residual.loc[hist].to_numpy()
        if history.empty:
            continue

        grouped = history.groupby(group_col, sort=False)["residual"]
        q_lo_map = grouped.quantile(quantiles[0])
        q_hi_map = grouped.quantile(quantiles[1])
        n_map = grouped.size()

        for group_value, target_idx in frame.loc[rows].groupby(group_col).groups.items():
            n = int(n_map.get(group_value, 0))
            if n < min_rows:
                continue
            idx = pd.Index(target_idx)
            valid = point.loc[idx].notna()
            idx = idx[valid]
            if len(idx) == 0:
                continue
            out.loc[idx, "q_lo"] = point.loc[idx] + float(q_lo_map.loc[group_value])
            out.loc[idx, "q_hi"] = point.loc[idx] + float(q_hi_map.loc[group_value])
            out.loc[idx, "n_history"] = n
            out.loc[idx, "history_start"] = history_start
            # Max timestamp among the actual residuals in this group; this is
            # the strongest audit timestamp rather than merely the window edge.
            used = hist & (frame[group_col] == group_value)
            out.loc[idx, "history_end"] = t.loc[used].max()
            out.loc[idx, "origin"] = origin

    out["available"] = out["q_lo"].notna() & out["q_hi"].notna()
    return out
