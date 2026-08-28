"""Harmonic regression: temporal integrity, DST behaviour, design correctness.

The risky behaviour here is not the arithmetic of least squares. It is that a
model refit per day can quietly be fitted on the day it is asked to predict, and
that a design matrix built from clock features can go wrong on the two days of
the year whose clock is irregular.
"""

import numpy as np
import pandas as pd
import pytest

from electricity_load_forecasting.calendar import civil_features
from electricity_load_forecasting.fourier import (
    SPEC, Spec, day_class, design_matrix, fit_ols, forecast_by_origin,
    fourier_terms)
from electricity_load_forecasting.validation import assert_no_lookahead


def frame(start="2023-05-01 00:00Z", days=90, seed=0):
    idx = pd.Series(pd.date_range(start, periods=days * 96, freq="15min", tz="UTC"))
    f = pd.concat([pd.DataFrame({"datetime": idx}), civil_features(idx)], axis=1)
    rng = np.random.default_rng(seed)
    phase = 2 * np.pi * f["qod"].to_numpy() / 96.0
    f["y"] = (10000.0
              + 2000.0 * np.sin(phase - 1.1)
              + 600.0 * np.sin(2 * phase)
              - 900.0 * (f["is_weekend"] | f["is_holiday"]).to_numpy()
              + rng.normal(0, 50, len(f)))
    return f


# ── design ──────────────────────────────────────────────────────────────

def test_fourier_terms_are_orthonormal_over_a_whole_period():
    p = np.arange(96) / 96.0
    T = fourier_terms(p, 3, "d")
    G = T.to_numpy().T @ T.to_numpy() / 96.0
    assert np.allclose(G, 0.5 * np.eye(6), atol=1e-12)


def test_fourier_terms_are_periodic_in_the_phase():
    a = fourier_terms(np.array([0.0, 0.25]), 2, "d")
    b = fourier_terms(np.array([1.0, 1.25]), 2, "d")
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_a_phase_with_a_hole_is_rejected_rather_than_transformed():
    with pytest.raises(ValueError):
        fourier_terms(np.array([0.0, np.nan]), 2, "d")


def test_the_design_uses_only_the_row_s_own_calendar_position():
    """Shuffling the rows must permute the design identically: any dependence
    on neighbouring rows would be a hidden temporal feature."""
    f = frame(days=20)
    X = design_matrix(f)
    perm = np.random.default_rng(1).permutation(len(f))
    Xp = design_matrix(f.iloc[perm].reset_index(drop=True))
    assert np.allclose(X.to_numpy()[perm], Xp.to_numpy())


def test_the_design_never_reads_the_target():
    f = frame(days=20)
    X = design_matrix(f)
    g = f.copy()
    g["y"] = g["y"] * 3.0 + 1000.0
    assert np.allclose(X.to_numpy(), design_matrix(g).to_numpy())


def test_holidays_are_classified_with_the_weekend_not_the_weekday():
    f = frame("2023-07-17 00:00Z", days=10)                 # 21 July is a Friday
    cls = day_class(f)
    hol = f["civil_date"] == pd.Timestamp("2023-07-21")
    assert (cls[hol] == "non-working").all()
    assert (f.loc[hol, "dow"] == 4).all()                   # still a Friday


def test_the_daily_block_is_split_by_day_class():
    f = frame(days=20)
    X = design_matrix(f)
    non = (day_class(f) == "non-working").to_numpy()
    assert (X.loc[non, "d_sin1_wk"] == 0).all()
    assert (X.loc[~non, "d_sin1_nw"] == 0).all()


def test_an_unsplit_design_has_fewer_columns_and_still_fits():
    f = frame(days=30)
    wide, narrow = design_matrix(f, SPEC), design_matrix(f, Spec(split_daily_by_day_class=False))
    assert narrow.shape[1] < wide.shape[1]
    fit_ols(narrow, f["y"])


def test_least_squares_recovers_a_known_harmonic_signal():
    f = frame(days=60, seed=2)
    X = design_matrix(f)
    beta, n = fit_ols(X, f["y"])
    resid = f["y"].to_numpy() - X.to_numpy() @ beta
    assert n == len(f)
    assert np.std(resid) < 80.0                             # noise sd is 50


def test_a_rank_deficient_design_still_returns_a_solution():
    f = frame(days=30)
    X = design_matrix(f)
    X["duplicate"] = X["d_sin1_wk"]
    beta, _ = fit_ols(X, f["y"])
    assert np.isfinite(beta).all()


def test_too_few_rows_for_the_parameters_fails_loudly():
    f = frame(days=30).head(5)
    with pytest.raises(ValueError):
        fit_ols(design_matrix(f), f["y"])


def test_rows_with_a_missing_target_are_excluded_from_the_fit():
    f = frame(days=40)
    f.loc[f.index[:200], "y"] = np.nan
    _, n = fit_ols(design_matrix(f), f["y"])
    assert n == len(f) - 200


# ── temporal integrity ──────────────────────────────────────────────────

def test_no_fit_window_reaches_its_own_target_day():
    f = frame(days=90)
    for window in (56, None):
        out = forecast_by_origin(f, window_days=window)
        got = f.assign(fit_end=out["fit_end"]).dropna(subset=["fit_end"])
        assert (got["fit_end"] < got["datetime"]).all(), f"window={window} leaks"


def test_the_fit_window_ends_at_or_before_the_stated_origin():
    f = frame(days=90)
    out = forecast_by_origin(f, window_days=56)
    got = out.dropna(subset=["fit_end"])
    assert (got["fit_end"] <= got["origin"]).all()
    assert_no_lookahead(got["fit_end"], got["origin"].max())


def test_the_origin_is_six_in_the_evening_on_the_previous_civil_day():
    f = frame(days=40)
    out = forecast_by_origin(f, window_days=56)
    got = f.assign(origin=out["origin"]).dropna(subset=["origin"])
    local = got["origin"].dt.tz_convert("Europe/Brussels")
    assert (local.dt.hour == 18).all()
    assert ((got["civil_date"] - local.dt.normalize().dt.tz_localize(None))
            == pd.Timedelta(days=1)).all()


def test_shifting_the_targets_forward_cannot_improve_the_fit_window():
    """A sign error in the origin would show up as a fit window that ends after
    the target begins; this pins the direction."""
    f = frame(days=60)
    out = forecast_by_origin(f, window_days=56)
    got = f.assign(end=out["fit_end"], start=out["fit_start"]).dropna(subset=["end"])
    assert (got["start"] < got["end"]).all()
    assert (got["end"] < got["datetime"].groupby(got["civil_date"]).transform("min")).all()


def test_a_rolling_window_never_exceeds_its_stated_length():
    f = frame(days=120)
    out = forecast_by_origin(f, window_days=28, min_history_days=21)
    got = out.dropna(subset=["fit_start"])
    assert len(got) > 0
    span = (got["fit_end"] - got["fit_start"]).dt.total_seconds() / 86400.0
    assert span.max() <= 28.0


def test_an_expanding_window_grows_and_a_rolling_one_does_not():
    f = frame(days=120)
    exp = forecast_by_origin(f, window_days=None).dropna(subset=["n_fit"])
    roll = forecast_by_origin(f, window_days=28,
                              min_history_days=21).dropna(subset=["n_fit"])
    assert exp["n_fit"].iloc[-1] > exp["n_fit"].iloc[0]
    assert roll["n_fit"].max() <= 28 * 96           # bounded by the stated window
    assert exp["n_fit"].max() > roll["n_fit"].max()  # expanding is not bounded


def test_days_without_enough_history_get_no_forecast():
    """History is measured to the origin, not to midnight, so the boundary is
    derived from the data rather than assumed to fall on a round day."""
    f = frame(days=60)
    out = forecast_by_origin(f, window_days=56, min_history_days=28)
    first = f["datetime"].min()
    origin = out.loc[out["available"], "origin"]
    assert ((origin - first).dt.total_seconds() / 86400.0 >= 28.0).all()
    avail_days = f.loc[out["available"], "civil_date"].unique()
    blocked = f.loc[~out["available"], "civil_date"].unique()
    assert len(avail_days) > 0 and len(blocked) > 0
    assert max(blocked) < min(avail_days)                 # one clean boundary


def test_a_minimum_history_longer_than_the_window_is_refused():
    """Otherwise every day silently goes unforecast and the run looks empty
    rather than wrong."""
    f = frame(days=60)
    with pytest.raises(ValueError, match="never span"):
        forecast_by_origin(f, window_days=28, min_history_days=28)


def test_a_window_shorter_than_a_week_is_refused():
    f = frame(days=40)
    with pytest.raises(ValueError):
        forecast_by_origin(f, window_days=3)


# ── DST and missing data ────────────────────────────────────────────────

def test_the_spring_forward_day_gets_a_forecast_for_each_of_its_92_slots():
    f = frame("2023-02-01 00:00Z", days=90)
    out = forecast_by_origin(f, window_days=56)
    day = f["civil_date"] == pd.Timestamp("2023-03-26")
    assert int(day.sum()) == 92
    assert out.loc[day, "available"].all()


def test_the_fall_back_day_gets_a_forecast_for_each_of_its_100_slots():
    f = frame("2023-09-01 00:00Z", days=80)
    out = forecast_by_origin(f, window_days=56)
    day = f["civil_date"] == pd.Timestamp("2023-10-29")
    assert int(day.sum()) == 100
    assert out.loc[day, "available"].all()


def test_the_repeated_hour_receives_the_same_forecast_twice():
    """Both passes share a quarter-of-day, and the model sees only clock
    position, so identical predictions are the correct behaviour — worth
    pinning, because an implementation keyed on row position would not."""
    f = frame("2023-09-01 00:00Z", days=80)
    out = forecast_by_origin(f, window_days=56)
    day = (f["civil_date"] == pd.Timestamp("2023-10-29")) & (f["qod"] == 8)
    v = out.loc[day, "yhat"].to_numpy()
    assert len(v) == 2 and np.isclose(v[0], v[1])


def test_a_missing_target_does_not_stop_a_forecast_being_produced():
    f = frame(days=90)
    hole = f.index[(f["civil_date"] == pd.Timestamp("2023-07-01")) & (f["qod"] == 40)]
    f.loc[hole, "y"] = np.nan
    out = forecast_by_origin(f, window_days=56)
    assert out.loc[hole, "available"].all()


def test_missing_history_shrinks_the_fit_but_not_the_forecast():
    f = frame(days=90)
    gap = f.index[(f["civil_date"] == pd.Timestamp("2023-06-10"))]
    f.loc[gap, "y"] = np.nan
    out = forecast_by_origin(f, window_days=56)
    later = f["civil_date"] == pd.Timestamp("2023-07-10")
    assert out.loc[later, "available"].all()
    assert out.loc[later, "n_fit"].iloc[0] <= 56 * 96 - len(gap)


def test_the_spec_describes_itself_in_periods_not_indices():
    text = SPEC.describe()
    for expected in ("24h", "12h", "8h", "6h", "168h", "least squares"):
        assert expected in text
