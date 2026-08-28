"""Spectral estimation: does the periodogram recover what was put into it?"""

import numpy as np
import pytest

from electricity_load_forecasting.spectral import (
    GapTooLongError, dominant_periods, fill_short_gaps, periodogram)

FS = 96.0            # quarter-hour sampling, in samples per day


def _t(days=365, fs=FS):
    return np.arange(days * int(fs)) / fs


def test_periodogram_recovers_a_known_daily_cycle():
    x = 3.0 * np.sin(2 * np.pi * _t())                    # exactly 1 cycle/day
    f, p = periodogram(x, FS)
    assert 24.0 / f[int(np.argmax(p))] == pytest.approx(24.0, rel=1e-3)


def test_periodogram_separates_two_injected_frequencies():
    """Nothing tells the estimator which periods to look for."""
    t = _t()
    x = 5.0 * np.sin(2 * np.pi * t) + 2.0 * np.sin(2 * np.pi * t / 7.0)
    got = sorted(dominant_periods(*periodogram(x, FS), n_peaks=2)["period_h"])
    assert got[0] == pytest.approx(24.0, rel=0.02)
    assert got[1] == pytest.approx(168.0, rel=0.05)       # weekly falls off-bin


def test_integrated_density_recovers_the_variance():
    """Parseval: the scaling must make the spectrum readable as variance."""
    x = np.random.default_rng(0).normal(0.0, 2.0, 96 * 200)
    f, p = periodogram(x, FS, window=None)
    assert np.trapezoid(p, f) == pytest.approx(np.var(x), rel=0.02)


def test_detrending_removes_a_ramp_rather_than_spreading_it():
    t = _t(60)
    x = 2.0 * np.sin(2 * np.pi * t) + 0.5 * t             # cycle plus drift
    f, p_on = periodogram(x, FS, detrend=True)
    _, p_off = periodogram(x, FS, detrend=False)
    low = f < 0.5
    assert p_on[low].sum() < p_off[low].sum()


def test_periodogram_refuses_a_series_with_holes():
    x = np.sin(np.arange(1000) / 10.0)
    x[17] = np.nan
    with pytest.raises(ValueError):
        periodogram(x, FS)


def test_dominant_periods_are_ranked_by_power():
    t = _t()
    x = 5.0 * np.sin(2 * np.pi * t) + 1.0 * np.sin(2 * np.pi * 2 * t)
    d = dominant_periods(*periodogram(x, FS), n_peaks=2)
    assert list(d["psd"]) == sorted(d["psd"], reverse=True)
    assert d.loc[0, "period_h"] == pytest.approx(24.0, rel=0.02)


def test_dominant_periods_respects_the_period_band():
    t = _t()
    x = np.sin(2 * np.pi * t) + np.sin(2 * np.pi * t / 7.0)
    d = dominant_periods(*periodogram(x, FS), min_period_h=48.0)
    assert (d["period_h"] >= 48.0).all()


def test_short_gaps_are_filled_and_reported():
    x = np.arange(100.0)
    x[[10, 11, 40]] = np.nan
    filled, was = fill_short_gaps(x, max_gap=4)
    assert not np.isnan(filled).any()
    assert was.sum() == 3
    assert filled[10] == pytest.approx(10.0)              # linear through a ramp
    assert (filled[~was] == x[~was]).all()                # observed data untouched


def test_a_long_gap_fails_loudly_instead_of_being_smoothed():
    x = np.arange(100.0)
    x[20:30] = np.nan
    with pytest.raises(GapTooLongError):
        fill_short_gaps(x, max_gap=4)


def test_weekly_fill_is_an_independent_alternative_to_interpolation():
    """Two unrelated fills let the spectrum be cross-checked, not asserted."""
    x = np.random.default_rng(1).normal(size=96 * 30)
    x[500] = np.nan
    lin, _ = fill_short_gaps(x, method="linear")
    wk, _ = fill_short_gaps(x, method="weekly")
    assert lin[500] != wk[500]
    assert not np.isnan(wk).any()
