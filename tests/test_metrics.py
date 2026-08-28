"""Scoring behaviour: the pinball loss, and keeping the two samples apart."""

import numpy as np
import pandas as pd
import pytest

from electricity_load_forecasting.metrics import (
    coverage, interval_width, pinball, score, score_point, score_prob)


def test_pinball_against_hand_computed_values():
    """alpha=0.1: under- and over-prediction weighted 0.1 and 0.9."""
    assert pinball([10.0], [8.0], 0.10) == pytest.approx(0.2)
    assert pinball([10.0], [12.0], 0.10) == pytest.approx(1.8)
    assert pinball([10.0], [10.0], 0.10) == pytest.approx(0.0)


def test_pinball_at_the_median_is_half_the_absolute_error():
    y, q = np.array([1.0, 5.0, 9.0]), np.array([4.0, 4.0, 4.0])
    assert pinball(y, q, 0.5) == pytest.approx(0.5 * np.mean(np.abs(y - q)))


def test_pinball_is_minimised_at_the_true_quantile():
    """The defining property of a proper scoring rule for quantiles."""
    s = np.random.default_rng(0).normal(0.0, 1.0, 200_000)
    truth = float(np.quantile(s, 0.10))
    grid = np.linspace(truth - 0.5, truth + 0.5, 41)
    best = grid[int(np.argmin([pinball(s, np.full_like(s, g), 0.10) for g in grid]))]
    assert best == pytest.approx(truth, abs=0.05)


def test_pinball_is_non_negative_and_scales_with_the_units():
    rng = np.random.default_rng(1)
    y, q = rng.normal(size=500), rng.normal(size=500)
    assert pinball(y, q, 0.9) >= 0.0
    assert pinball(3.0 * y, 3.0 * q, 0.9) == pytest.approx(3.0 * pinball(y, q, 0.9))


def test_pinball_rejects_an_alpha_outside_the_unit_interval():
    with pytest.raises(ValueError):
        pinball([1.0], [1.0], 1.0)


def test_coverage_and_width_behave_as_defined():
    assert coverage([1.0, 5.0, 9.0], [0.0] * 3, [6.0] * 3) == pytest.approx(2 / 3)
    assert coverage([1.0], [1.0], [1.0]) == 1.0            # bounds are inclusive
    assert interval_width([0.0, 1.0], [3.0, 6.0]) == pytest.approx(4.0)


def test_a_wider_band_never_lowers_coverage():
    """Coverage alone is gameable — the reason sharpness is always reported."""
    y = np.random.default_rng(2).normal(size=2000)
    assert coverage(y, -2.0, 2.0) >= coverage(y, -1.0, 1.0)


def _frame(n=1000, seed=3):
    rng = np.random.default_rng(seed)
    y = 100.0 + rng.normal(0.0, 10.0, n)
    e = rng.normal(0.0, 4.0, n)
    return pd.DataFrame({"y": y, "yhat_da": y - e,
                         "q10_da": y - e - 8.0, "q90_da": y - e + 8.0})


def test_point_and_probabilistic_samples_stay_separate():
    """A missing confidence bound must not shrink the population behind MAE."""
    d = _frame()
    d.loc[:99, ["q10_da", "q90_da"]] = np.nan
    s = score(d)
    assert s["n_point"] == 1000
    assert s["n_prob"] == 900
    assert s["MAE"] == pytest.approx(score_point(d)["MAE"])


def test_a_missing_actual_is_dropped_from_both_blocks():
    d = _frame()
    d.loc[:49, "y"] = np.nan
    s = score(d)
    assert s["n_point"] == 950 and s["n_prob"] == 950


def test_pin_mean_is_the_average_of_the_two_quantile_losses():
    s = score_prob(_frame())
    assert s["pin_mean"] == pytest.approx(0.5 * (s["pin_lo"] + s["pin_hi"]))


def test_metrics_are_empty_but_typed_when_nothing_survives():
    d = _frame(10).assign(y=np.nan)
    s = score(d)
    assert s["n_point"] == 0 and s["n_prob"] == 0
    assert np.isnan(s["MAE"]) and np.isnan(s["cov"])


def test_bias_and_rmse_recover_a_known_offset():
    d = pd.DataFrame({"y": [10.0, 12.0], "yhat_da": [9.0, 11.0]})
    s = score_point(d)
    assert s["bias"] == pytest.approx(1.0)
    assert s["RMSE"] == pytest.approx(1.0)
