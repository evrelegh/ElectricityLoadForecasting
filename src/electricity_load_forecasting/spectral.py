# ══════════════════════════════════════════════════════════════════════════
# spectral · gap handling and periodogram estimation                v0.2.0
# provides ▸ fill_short_gaps · periodogram · dominant_periods
# requires ▸ numpy · pandas · scipy.signal
# ══════════════════════════════════════════════════════════════════════════
"""Spectral estimation on a uniformly sampled series.

The FFT requires a gapless uniform grid, so short holes must be filled before
transforming. Filling is deliberate and reported, never silent: a gap longer
than `max_gap` raises rather than being smoothed over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, get_window


class GapTooLongError(ValueError):
    """A run of missing observations exceeds what interpolation can justify."""


def fill_short_gaps(y, max_gap: int = 4, method: str = "linear"):
    """Fill runs of at most `max_gap` missing values; raise on anything longer.

    Returns (filled, was_filled). `method` is 'linear' or 'weekly' — the latter
    substitutes the value one week earlier, giving an independent fill for
    cross-checking that the spectrum does not depend on the interpolation.
    """
    s = pd.Series(np.asarray(y, dtype=float)).copy()
    na = s.isna()
    if not na.any():
        return s.to_numpy(), na.to_numpy()

    run = na.ne(na.shift()).cumsum()
    longest = int(na.groupby(run).sum().max())
    if longest > max_gap:
        raise GapTooLongError(
            f"longest run of missing values is {longest} slots, above max_gap={max_gap}; "
            "interpolation is not defensible here — inspect the source data")

    if method == "linear":
        out = s.interpolate(method="linear", limit_direction="both")
    elif method == "weekly":
        lag = 96 * 7
        out = s.copy()
        idx = np.flatnonzero(na.to_numpy())
        for i in idx:
            j = i - lag if i - lag >= 0 else i + lag
            out.iloc[i] = s.iloc[j]
        out = out.interpolate(method="linear", limit_direction="both")
    else:
        raise ValueError(f"unknown fill method {method!r}")
    return out.to_numpy(), na.to_numpy()


def periodogram(x, fs: float, detrend: bool = True, window: str | None = "hann"):
    """One-sided power spectral density of `x` sampled at `fs` samples per unit.

    Scaled so that integrating the density over frequency recovers the variance
    of the detrended, windowed signal. With `fs` in samples/day the frequency
    axis is in cycles/day, which is the unit the periods are read in.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 8:
        raise ValueError("x must be a 1-D series of at least 8 samples")
    if np.isnan(x).any():
        raise ValueError("x contains NaN — fill gaps before transforming")

    n = x.size
    if detrend:                                    # remove mean and linear trend
        t = np.arange(n, dtype=float)
        x = x - np.polyval(np.polyfit(t, x, 1), t)

    w = np.ones(n) if window is None else get_window(window, n)
    xw = x * w
    spec = np.abs(np.fft.rfft(xw)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    psd = 2.0 * spec / (fs * np.sum(w ** 2))
    psd[0] /= 2.0                                  # DC is not mirrored
    if n % 2 == 0:
        psd[-1] /= 2.0                             # nor is Nyquist
    return freqs, psd


def dominant_periods(freqs, psd, n_peaks: int = 12, min_period_h: float = 1.0,
                     max_period_h: float | None = None, prominence_db: float = 6.0
                     ) -> pd.DataFrame:
    """Rank spectral peaks found in the data, then express them as periods.

    Peaks are located by prominence on the log spectrum, so nothing about the
    expected daily or weekly cycle is supplied in advance; the periods that
    come out are whatever the series actually contains.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)

    keep = freqs > 0
    f, p = freqs[keep], psd[keep]
    period_h = 24.0 / f                             # freqs are cycles/day

    band = period_h >= min_period_h
    if max_period_h is not None:
        band &= period_h <= max_period_h
    f, p, period_h = f[band], p[band], period_h[band]

    logp = 10.0 * np.log10(np.maximum(p, np.finfo(float).tiny))
    idx, _ = find_peaks(logp, prominence=prominence_db)
    if idx.size == 0:
        return pd.DataFrame(columns=["freq_cpd", "period_h", "period_d", "psd", "share"])

    order = idx[np.argsort(p[idx])[::-1]][:n_peaks]
    total = float(np.sum(p))
    return pd.DataFrame({
        "freq_cpd": f[order],
        "period_h": period_h[order],
        "period_d": period_h[order] / 24.0,
        "psd": p[order],
        "share": p[order] / total,
    }).reset_index(drop=True)
