# ══════════════════════════════════════════════════════════════════════════
# fourier · transparent Fourier + calendar point forecast            v0.5.0
# provides ▸ SPEC · fourier_terms · design_matrix · fit_ols
#            forecast_by_origin
# requires ▸ numpy · pandas · baselines.daily_origin
# ══════════════════════════════════════════════════════════════════════════
"""A deliberately small harmonic regression for day-ahead load.

The design is fixed in advance and never searched. Harmonic counts are the
first K harmonics of the day and the week — periods that follow from the
calendar, not from a peak picked out of a spectrum estimated on the evaluation
year. The §4 periodogram is consistent with this choice but did not make it;
selecting frequencies from a full-year spectrum would leak the test period into
the model specification.

The daily block is estimated separately for working and non-working days
because the *shape* of the day differs between them, not merely its level. That
is one pre-specified interaction, not a search.

Phases are built from civil-clock features, so the DST days need no special
case: a 92- or 100-slot day simply supplies fewer or more rows carrying the
same quarter-of-day phase.

No annual harmonic is included. With roughly a month of history available
before the evaluation period begins, a yearly cycle is not identifiable at the
start of the window, and fitting one would amount to extrapolating a period the
data cannot support. The seasonal level is instead carried by the estimation
window; both a rolling and an expanding protocol are provided so the effect of
that choice is visible rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import daily_origin


@dataclass(frozen=True)
class Spec:
    """The model, written down before it is estimated."""
    daily_harmonics: int = 4          # 24, 12, 8, 6 hours
    weekly_harmonics: int = 3         # 168, 84, 56 hours
    split_daily_by_day_class: bool = True
    day_classes: tuple = ("working", "non-working")

    def describe(self) -> str:
        d = [f"{24 / k:g}h" for k in range(1, self.daily_harmonics + 1)]
        w = [f"{168 / k:g}h" for k in range(1, self.weekly_harmonics + 1)]
        return (f"daily harmonics {', '.join(d)}"
                + (" estimated separately for working and non-working days"
                   if self.split_daily_by_day_class else "")
                + f"; weekly harmonics {', '.join(w)}; "
                  "intercept and a non-working level term; ordinary least squares")


SPEC = Spec()


def fourier_terms(phase, n_harmonics: int, prefix: str) -> pd.DataFrame:
    """sin/cos pairs for harmonics 1..n of a phase already scaled to [0, 1)."""
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be at least 1, got {n_harmonics}")
    p = np.asarray(phase, dtype=float)
    if np.isnan(p).any():
        raise ValueError(f"{prefix}: phase contains NaN")
    out = {}
    for k in range(1, n_harmonics + 1):
        out[f"{prefix}_sin{k}"] = np.sin(2 * np.pi * k * p)
        out[f"{prefix}_cos{k}"] = np.cos(2 * np.pi * k * p)
    return pd.DataFrame(out)


def day_class(frame: pd.DataFrame) -> pd.Series:
    """Working versus non-working, the only calendar distinction the model makes.

    A public holiday is treated as non-working: its load shape resembles a
    Sunday far more than the weekday it happens to fall on.
    """
    non = frame["is_weekend"].to_numpy() | frame["is_holiday"].to_numpy()
    return pd.Series(np.where(non, "non-working", "working"), index=frame.index)


def design_matrix(frame: pd.DataFrame, spec: Spec = SPEC) -> pd.DataFrame:
    """Regressors for each row, built only from that row's civil-clock position.

    Nothing here looks at the load, at neighbouring rows, or at anything dated
    after the row itself, so the matrix carries no temporal information beyond
    the calendar.
    """
    for c in ("qod", "dow", "is_weekend", "is_holiday"):
        if c not in frame.columns:
            raise KeyError(f"frame is missing required column {c!r}")

    qod = frame["qod"].to_numpy(dtype=float)
    dow = frame["dow"].to_numpy(dtype=float)
    day_phase = qod / 96.0
    week_phase = (dow * 96.0 + qod) / 672.0

    X = pd.DataFrame({"const": np.ones(len(frame))}, index=frame.index)

    daily = fourier_terms(day_phase, spec.daily_harmonics, "d")
    daily.index = frame.index
    if spec.split_daily_by_day_class:
        cls = day_class(frame)
        non = (cls == "non-working").to_numpy(dtype=float)
        X["nonworking"] = non
        for c in daily.columns:
            X[f"{c}_wk"] = daily[c].to_numpy() * (1.0 - non)
            X[f"{c}_nw"] = daily[c].to_numpy() * non
    else:
        X["nonworking"] = (day_class(frame) == "non-working").astype(float)
        for c in daily.columns:
            X[c] = daily[c].to_numpy()

    weekly = fourier_terms(week_phase, spec.weekly_harmonics, "w")
    weekly.index = frame.index
    for c in weekly.columns:
        X[c] = weekly[c].to_numpy()
    return X


def fit_ols(X: pd.DataFrame, y):
    """Least-squares coefficients, minimum-norm if the design is rank deficient.

    Rank deficiency is possible — the weekly harmonics and the day-class split
    overlap in what they can represent — so the pseudo-inverse solution is used
    rather than assuming full rank and failing at some point in the year.
    """
    A = np.asarray(X, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(b) & np.isfinite(A).all(axis=1)
    if ok.sum() <= A.shape[1]:
        raise ValueError(f"only {int(ok.sum())} usable rows for {A.shape[1]} parameters")
    beta, *_ = np.linalg.lstsq(A[ok], b[ok], rcond=None)
    return beta, int(ok.sum())


def forecast_by_origin(frame: pd.DataFrame, targets=None, spec: Spec = SPEC,
                       origin_civil_hour: float = 18.0, lead_days: int = 1,
                       tz: str = "Europe/Brussels", window_days: int | None = 56,
                       min_history_days: int = 28, y: str = "y",
                       time: str = "datetime") -> pd.DataFrame:
    """Refit once per target civil day, using only observations before its origin.

    `window_days=None` gives an expanding window; an integer gives a rolling one
    of that many days ending at the origin. Returns a frame aligned to
    `frame.index` carrying the forecast and the audit trail needed to prove the
    fit never saw the target: the fit window bounds and its row count.
    """
    for c in ("civil_date", "qod", "dow", "is_weekend", "is_holiday", y, time):
        if c not in frame.columns:
            raise KeyError(f"frame is missing required column {c!r}")
    if window_days is not None:
        if window_days < 7:
            raise ValueError("a window shorter than a week cannot identify the weekly terms")
        if min_history_days >= window_days:
            raise ValueError(
                f"min_history_days={min_history_days} is not shorter than "
                f"window_days={window_days}: a rolling window can never span that "
                "much history, so every day would silently go unforecast")

    t = pd.to_datetime(frame[time], utc=True)
    X = design_matrix(frame, spec)
    days = (sorted(pd.unique(frame["civil_date"])) if targets is None
            else sorted(pd.unique(pd.to_datetime(pd.Series(targets)))))

    nat = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    out = pd.DataFrame({"yhat": np.nan, "n_fit": np.nan,
                        "fit_start": nat, "fit_end": nat.copy(),
                        "origin": nat.copy()}, index=frame.index)

    for day in days:
        rows = frame.index[frame["civil_date"] == day]
        if len(rows) == 0:
            continue
        origin = daily_origin([day], origin_civil_hour, lead_days, tz).iloc[0]

        train = t <= origin                       # strictly before the target day
        if window_days is not None:
            train &= t > origin - pd.Timedelta(days=window_days)
        idx = frame.index[train]
        if len(idx) == 0:
            continue
        span = (t.loc[idx].max() - t.loc[idx].min()).total_seconds() / 86400.0
        if span < min_history_days:
            continue

        try:
            beta, n_fit = fit_ols(X.loc[idx], frame.loc[idx, y])
        except ValueError:
            continue

        out.loc[rows, "yhat"] = np.asarray(X.loc[rows], dtype=float) @ beta
        out.loc[rows, "n_fit"] = n_fit
        out.loc[rows, "fit_start"] = t.loc[idx].min()
        out.loc[rows, "fit_end"] = t.loc[idx].max()
        out.loc[rows, "origin"] = origin

    out["available"] = out["yhat"].notna()
    return out
