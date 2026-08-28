# ══════════════════════════════════════════════════════════════════════════
# level · leakage-safe recent level correction                        v0.1.0
# provides ▸ recent_residual_level_correction
# requires ▸ numpy · pandas · baselines.daily_origin
# ══════════════════════════════════════════════════════════════════════════
"""Recent-level correction for an already leakage-safe point forecast.

The Fourier/calendar experiment shows a useful separation: its remaining error
is dominated by day-level positioning rather than by within-day structure.  The
smallest follow-up is therefore not another harmonic search, but an additive
level correction estimated from *recent realised forecast errors*.

For target civil day D, the correction is the mean error ``actual - forecast``
over a fixed elapsed-time window ending at the forecast origin (18:00 on D-1 by
default).  Only base forecasts that were themselves produced out of sample are
used.  The resulting scalar is added to every available base forecast on D, so
this step cannot alter the forecast's intraday shape.

A 24-hour elapsed-time window is the pre-specified default.  It is deliberately
not a civil-day window: around daylight-saving transitions an elapsed 24 hours
still contains the same amount of recent information, while the target civil
day may contain 92, 96 or 100 quarter-hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import daily_origin


def recent_residual_level_correction(
    frame: pd.DataFrame,
    base_forecast,
    targets=None,
    *,
    origin_civil_hour: float = 18.0,
    lead_days: int = 1,
    tz: str = "Europe/Brussels",
    window_hours: float = 24.0,
    min_rows: int = 48,
    y: str = "y",
    time: str = "datetime",
) -> pd.DataFrame:
    """Shift each target day's base forecast by its recent realised mean error.

    Parameters
    ----------
    frame:
        Must contain ``civil_date``, the realised load column ``y`` and the UTC
        timestamp column ``time``.
    base_forecast:
        Leakage-safe point forecasts aligned to ``frame.index``.  Historical
        values are used only after they have been realised and only up to the
        target day's forecast origin.
    targets:
        Target civil dates.  If omitted, all civil dates in ``frame`` are used.
    window_hours:
        Elapsed-time lookback ending at the origin.  The default 24 hours is a
        design choice, not a tuned hyperparameter.
    min_rows:
        Minimum finite realised forecast errors required to estimate the level
        offset.  The default 48 merely prevents a badly incomplete recent
        window from degenerating into a one- or two-point correction.

    Returns
    -------
    DataFrame aligned to ``frame.index`` with ``yhat``, ``level_offset``,
    ``n_level``, ``level_start``, ``level_end``, ``origin`` and ``available``.
    The same offset is repeated over all slots of a target civil day.
    """
    for c in ("civil_date", y, time):
        if c not in frame.columns:
            raise KeyError(f"frame is missing required column {c!r}")
    if window_hours <= 0:
        raise ValueError(f"window_hours must be positive, got {window_hours}")
    if min_rows < 1:
        raise ValueError(f"min_rows must be at least 1, got {min_rows}")

    if isinstance(base_forecast, pd.Series):
        if not base_forecast.index.equals(frame.index):
            raise ValueError(
                "base_forecast index must match frame.index exactly"
            )
        base = base_forecast.astype(float)
    else:
        values = np.asarray(base_forecast, dtype=float)
        if values.ndim != 1 or len(values) != len(frame):
            raise ValueError(
                "base_forecast must have one value per frame row"
            )
        base = pd.Series(values, index=frame.index, dtype=float)

    t = pd.to_datetime(frame[time], utc=True)
    actual = pd.to_numeric(frame[y], errors="coerce")
    residual = actual - base

    days = (sorted(pd.unique(frame["civil_date"])) if targets is None
            else sorted(pd.unique(pd.to_datetime(pd.Series(targets)))))

    nat = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    out = pd.DataFrame({
        "yhat": np.nan,
        "level_offset": np.nan,
        "n_level": np.nan,
        "level_start": nat,
        "level_end": nat.copy(),
        "origin": nat.copy(),
    }, index=frame.index)

    lookback = pd.Timedelta(hours=float(window_hours))

    for day in days:
        rows = frame.index[frame["civil_date"] == day]
        if len(rows) == 0:
            continue

        origin = daily_origin([day], origin_civil_hour, lead_days, tz).iloc[0]
        recent = (t <= origin) & (t > origin - lookback)
        usable = recent & residual.notna()
        idx = frame.index[usable]
        if len(idx) < min_rows:
            continue

        offset = float(residual.loc[idx].mean())
        target_base = base.loc[rows]
        ok_target = target_base.notna()
        if not ok_target.any():
            continue

        dst = rows[ok_target.to_numpy()]
        out.loc[dst, "yhat"] = target_base.loc[dst] + offset
        out.loc[dst, "level_offset"] = offset
        out.loc[dst, "n_level"] = len(idx)
        out.loc[dst, "level_start"] = t.loc[idx].min()
        out.loc[dst, "level_end"] = t.loc[idx].max()
        out.loc[dst, "origin"] = origin

    out["available"] = out["yhat"].notna()
    return out
