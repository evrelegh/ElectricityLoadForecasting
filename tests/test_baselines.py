"""Persistence baselines: civil-clock matching, DST policy, temporal integrity.

The failure these tests exist to prevent is a baseline implemented as a fixed
row shift. That is correct on 363 days of the year and quietly wrong on two,
which is exactly the kind of defect that survives eyeballing a plot.
"""

import numpy as np
import pandas as pd
import pytest

from electricity_load_forecasting.baselines import (
    civil_occurrence, daily_origin, origin_violations, seasonal_naive)
from electricity_load_forecasting.calendar import civil_features
from electricity_load_forecasting.validation import LookaheadError, assert_no_lookahead


def frame(start, days, seed=0, freq="15min"):
    """A quarter-hourly frame on the UTC axis with civil-time features."""
    idx = pd.Series(pd.date_range(start, periods=days * 96 + 4, freq=freq, tz="UTC"))
    f = pd.concat([pd.DataFrame({"datetime": idx}), civil_features(idx)], axis=1)
    f["y"] = np.arange(len(f), dtype=float) + 1000.0
    return f


# ── civil-clock matching ────────────────────────────────────────────────

def test_forecast_reuses_the_same_clock_time_one_day_earlier():
    f = frame("2023-06-10 00:00Z", 5)
    out = seasonal_naive(f, 1)
    got = f.assign(yhat=out["yhat"], src=out["source_time"]).dropna(subset=["yhat"])
    src = got.merge(f[["datetime", "civil_date", "qod"]], left_on="src",
                    right_on="datetime", suffixes=("", "_s"))
    assert (src["qod"] == src["qod_s"]).all()
    assert (src["civil_date"] - src["civil_date_s"] == pd.Timedelta(days=1)).all()


def test_the_source_value_is_the_observation_itself_not_a_neighbour():
    f = frame("2023-06-10 00:00Z", 3)
    out = seasonal_naive(f, 1)
    lookup = f.set_index("datetime")["y"]
    got = out.dropna(subset=["yhat"])
    assert np.allclose(got["yhat"], lookup.loc[got["source_time"]].to_numpy())


def test_a_lag_of_seven_days_preserves_the_weekday():
    f = frame("2023-06-01 00:00Z", 20)
    out = seasonal_naive(f, 7)
    got = f.assign(src=out["source_time"]).dropna(subset=["src"])
    src_dow = got["src"].dt.tz_convert("Europe/Brussels").dt.dayofweek
    assert (got["dow"].to_numpy() == src_dow.to_numpy()).all()


def test_a_row_shift_and_a_civil_match_disagree_across_the_transition():
    """The whole reason for the civil key: on an ordinary week they coincide,
    across a DST day a 96-row shift reuses the wrong clock time."""
    f = frame("2023-03-24 00:00Z", 5)
    civil = seasonal_naive(f, 1)["yhat"].to_numpy()
    rowshift = f["y"].shift(96).to_numpy()
    after = f["civil_date"] >= pd.Timestamp("2023-03-27")
    differ = ~np.isclose(civil[after], rowshift[after], equal_nan=True)
    assert differ.any(), "civil matching must differ from a row shift after the transition"


# ── DST policy ──────────────────────────────────────────────────────────

def test_spring_forward_gap_leaves_the_forecast_unavailable():
    """02:00-02:45 does not exist on 26 March, so the next day's 02:00-02:45
    has no counterpart. Unavailable is the answer; interpolation is not."""
    f = frame("2023-03-25 00:00Z", 4)
    out = seasonal_naive(f, 1)
    tgt = f["civil_date"] == pd.Timestamp("2023-03-27")
    missing = out.loc[tgt & f["qod"].between(8, 11), "available"]
    assert len(missing) == 4 and not missing.any()
    assert out.loc[tgt & (f["qod"] == 12), "available"].all()


def test_fall_back_repeats_a_quarter_and_both_passes_get_a_forecast():
    f = frame("2023-10-27 00:00Z", 4)
    out = seasonal_naive(f, 1)
    tgt = f["civil_date"] == pd.Timestamp("2023-10-29")
    rep = out.loc[tgt & f["qod"].between(8, 11)]
    assert len(rep) == 8                                  # each qod twice
    assert rep["available"].all()


def test_both_passes_of_a_repeated_hour_reuse_the_single_source_observation():
    """The source day has one 02:00; the policy reuses it for both passes."""
    f = frame("2023-10-27 00:00Z", 4)
    out = seasonal_naive(f, 1)
    tgt = (f["civil_date"] == pd.Timestamp("2023-10-29")) & (f["qod"] == 8)
    src = out.loc[tgt, "source_time"].tolist()
    assert len(src) == 2 and src[0] == src[1]


def test_occurrence_is_zero_except_inside_the_repeated_hour():
    f = frame("2023-10-28 00:00Z", 3)
    occ = civil_occurrence(f)
    assert set(occ.unique()) == {0, 1}
    assert int((occ == 1).sum()) == 4


def test_dst_days_are_not_dropped_from_the_forecast():
    """Simplifying by discarding the transition days would pass many tests;
    it must not pass this one."""
    f = frame("2023-10-27 00:00Z", 4)
    out = seasonal_naive(f, 1)
    dst = f["civil_date"] == pd.Timestamp("2023-10-29")
    assert out.loc[dst, "available"].sum() == 100


# ── missing data ────────────────────────────────────────────────────────

def test_a_missing_source_observation_makes_the_forecast_unavailable():
    f = frame("2023-06-10 00:00Z", 4)
    hole = f.index[(f["civil_date"] == pd.Timestamp("2023-06-11")) & (f["qod"] == 40)]
    f.loc[hole, "y"] = np.nan
    out = seasonal_naive(f, 1)
    tgt = (f["civil_date"] == pd.Timestamp("2023-06-12")) & (f["qod"] == 40)
    assert not out.loc[tgt, "available"].any()
    assert out["yhat"].notna().sum() == len(out) - out["yhat"].isna().sum()


def test_a_missing_target_does_not_affect_forecast_availability():
    """Availability is a property of the source; presence is a property of the
    target. Conflating them would silently shrink the evaluation sample."""
    f = frame("2023-06-10 00:00Z", 4)
    hole = f.index[(f["civil_date"] == pd.Timestamp("2023-06-12")) & (f["qod"] == 40)]
    f.loc[hole, "y"] = np.nan
    out = seasonal_naive(f, 1)
    assert out.loc[hole, "available"].all()


def test_no_forecast_exists_before_the_history_begins():
    f = frame("2023-06-10 00:00Z", 4)
    out = seasonal_naive(f, 1)
    first = f["civil_date"] == f["civil_date"].min()
    assert not out.loc[first, "available"].any()


# ── temporal integrity ──────────────────────────────────────────────────

def test_every_source_observation_precedes_its_target():
    f = frame("2023-06-01 00:00Z", 14)
    for lag in (1, 2, 7):
        out = seasonal_naive(f, lag)
        got = f.assign(src=out["source_time"]).dropna(subset=["src"])
        assert (got["src"] < got["datetime"]).all(), f"lag {lag} reaches forward"


def test_a_negative_or_zero_lag_is_rejected():
    f = frame("2023-06-10 00:00Z", 3)
    for bad in (0, -1, -7):
        with pytest.raises(ValueError):
            seasonal_naive(f, bad)


def test_a_forward_shift_is_caught_by_the_origin_guard():
    """The wrong-sign mistake this increment is most exposed to: if a future
    observation were used, the guard must fail rather than the metrics improve."""
    f = frame("2023-06-01 00:00Z", 10)
    origin = daily_origin(f["civil_date"], 12, 1)
    forward = f["datetime"] + pd.Timedelta(days=1)          # deliberately wrong
    assert len(origin_violations(forward, origin)) > 0
    with pytest.raises(LookaheadError):
        assert_no_lookahead(forward, origin.max())


def test_a_weekly_baseline_respects_a_noon_origin_on_the_previous_day():
    f = frame("2023-06-01 00:00Z", 21)
    out = seasonal_naive(f, 7)
    origin = daily_origin(f["civil_date"], 12, 1)
    assert origin_violations(out["source_time"], origin).empty


def test_a_daily_baseline_violates_an_evening_origin_only_after_that_hour():
    """At an 18:00 origin only the evening of the source day is inadmissible,
    so the violation is bounded by six hours rather than twelve."""
    f = frame("2023-06-01 00:00Z", 14)
    out = seasonal_naive(f, 1)
    v = origin_violations(out["source_time"], daily_origin(f["civil_date"], 18, 1))
    assert not v.empty
    assert v["late_by"].max() <= pd.Timedelta(hours=6)


def test_a_later_origin_admits_strictly_more_of_the_previous_day():
    """Monotonicity: moving the origin later can only reduce violations."""
    f = frame("2023-06-01 00:00Z", 14)
    src = seasonal_naive(f, 1)["source_time"]
    counts = [len(origin_violations(src, daily_origin(f["civil_date"], h, 1)))
              for h in (8.75, 12, 18)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] > 0


def test_a_fractional_origin_hour_lands_on_the_quarter():
    o = daily_origin(pd.Series([pd.Timestamp("2023-06-15")]), 8.75, 1)
    assert o.iloc[0] == pd.Timestamp("2023-06-14 06:45", tz="UTC")   # UTC+2 in summer


def test_a_daily_baseline_violates_a_noon_origin_by_construction():
    """Not a defect in the code: D-1 after 12:00 postdates a 12:00 D-1 origin.
    Recording it as a test keeps the conflict visible instead of forgotten."""
    f = frame("2023-06-01 00:00Z", 14)
    out = seasonal_naive(f, 1)
    origin = daily_origin(f["civil_date"], 12, 1)
    v = origin_violations(out["source_time"], origin)
    assert not v.empty
    assert v["late_by"].max() <= pd.Timedelta(hours=12)


def test_the_origin_is_local_noon_and_moves_with_the_zone():
    """Belgium is UTC+1 in winter and UTC+2 in summer; a fixed UTC origin would
    drift by an hour across the year."""
    o = daily_origin(pd.Series([pd.Timestamp("2023-01-15"), pd.Timestamp("2023-07-15")]),
                     12, 1)
    assert o.iloc[0] == pd.Timestamp("2023-01-14 11:00", tz="UTC")
    assert o.iloc[1] == pd.Timestamp("2023-07-14 10:00", tz="UTC")


def test_origin_violations_names_the_offenders_rather_than_counting_them():
    f = frame("2023-06-01 00:00Z", 5)
    origin = daily_origin(f["civil_date"], 12, 1)
    v = origin_violations(f["datetime"], origin)
    assert {"source_time", "origin", "late_by"} <= set(v.columns)
    assert (v["late_by"] > pd.Timedelta(0)).all()


# ── contract ────────────────────────────────────────────────────────────

def test_a_frame_without_the_civil_key_is_rejected():
    f = frame("2023-06-10 00:00Z", 3).drop(columns=["qod"])
    with pytest.raises(KeyError):
        seasonal_naive(f, 1)


def test_the_result_is_aligned_to_the_input_index():
    f = frame("2023-06-10 00:00Z", 3)
    f.index = f.index + 500
    out = seasonal_naive(f, 1)
    assert out.index.equals(f.index)
