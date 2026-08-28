# ══════════════════════════════════════════════════════════════════════════
# metrics · point accuracy, calibration and sharpness              v0.2.0
# provides ▸ pinball · coverage · interval_width · score_point · score_prob
#            score
# requires ▸ numpy · pandas
# ══════════════════════════════════════════════════════════════════════════
"""Forecast scoring.

Point and probabilistic metrics are computed on their own samples. A missing
confidence bound must not silently shrink the population behind MAE, so the
sample size of each block is reported rather than assumed equal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pinball(y, q, alpha: float) -> float:
    """Quantile (pinball) loss at level `alpha` — a proper scoring rule.

    L = mean( max( alpha*(y-q), (alpha-1)*(y-q) ) ), minimised in expectation
    by the true alpha-quantile of the predictive distribution.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1.0) * d)))


def coverage(y, lo, hi) -> float:
    """Empirical frequency of the realised value falling inside [lo, hi]."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(lo, hi) -> float:
    """Mean interval width — sharpness. Coverage alone is trivially gameable
    by widening the band, so the two are always read together."""
    return float(np.mean(np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)))


def score_point(d: pd.DataFrame, y="y", yhat="yhat_da") -> pd.Series:
    """Bias, MAE and RMSE on the sample where actual and point forecast exist."""
    d = d.dropna(subset=[y, yhat])
    if d.empty:
        return pd.Series({"n_point": 0.0, "bias": np.nan, "MAE": np.nan, "RMSE": np.nan})
    e = d[y] - d[yhat]
    return pd.Series({
        "n_point": float(len(d)),
        "bias": float(e.mean()),
        "MAE": float(e.abs().mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
    })


def score_prob(d: pd.DataFrame, y="y", lo="q10_da", hi="q90_da",
               alpha_lo: float = 0.10, alpha_hi: float = 0.90) -> pd.Series:
    """Calibration and sharpness on the sample where both bounds exist.

    `pin_mean` is a compact interval indicator, NOT a CRPS: two quantiles do
    not determine the predictive distribution.
    """
    d = d.dropna(subset=[y, lo, hi])
    if d.empty:
        return pd.Series({"n_prob": 0.0, "cov": np.nan, "width": np.nan,
                          "pin_lo": np.nan, "pin_hi": np.nan, "pin_mean": np.nan})
    p_lo = pinball(d[y], d[lo], alpha_lo)
    p_hi = pinball(d[y], d[hi], alpha_hi)
    return pd.Series({
        "n_prob": float(len(d)),
        "cov": coverage(d[y], d[lo], d[hi]),
        "width": interval_width(d[lo], d[hi]),
        "pin_lo": p_lo,
        "pin_hi": p_hi,
        "pin_mean": 0.5 * (p_lo + p_hi),
    })


def score(d: pd.DataFrame, **kw) -> pd.Series:
    """Point and probabilistic metrics side by side, each on its own sample."""
    point_kw = {k: v for k, v in kw.items() if k in ("y", "yhat")}
    prob_kw = {k: v for k, v in kw.items() if k in ("y", "lo", "hi", "alpha_lo", "alpha_hi")}
    return pd.concat([score_point(d, **point_kw), score_prob(d, **prob_kw)])
