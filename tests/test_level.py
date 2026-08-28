"""Recent-level correction: leakage, shape preservation and DST behaviour."""

import numpy as np
import pandas as pd
import pytest

from electricity_load_forecasting.calendar import civil_features
from electricity_load_forecasting.level import recent_residual_level_correction


def frame(start="2023-05-01 00:00Z", days=12):
    idx = pd.Series(pd.date_range(start, periods=days * 96, freq="15min", tz="UTC"))
    f = pd.concat([pd.DataFrame({"datetime": idx}), civil_features(idx)], axis=1)
    phase = 2 * np.pi * f["qod"].to_numpy() / 96.0
    base = 10_000.0 + 1_500.0 * np.sin(phase)
    f["base"] = base
    f["y"] = base + 300.0
    return f


def corrected(f, **kwargs):
    return recent_residual_level_correction(
        f, f["base"], window_hours=24, min_rows=48, **kwargs)


def test_a_constant_recent_bias_is_recovered_exactly():
    f = frame(days=12)
    out = corrected(f)
    got = out.dropna(subset=["yhat"])
    assert len(got) > 0
    assert np.allclose(got["level_offset"], 300.0)


def test_the_correction_preserves_intraday_shape_exactly():
    f = frame(days=12)
    out = corrected(f)
    got = out["available"]
    delta = out.loc[got, "yhat"] - f.loc[got, "base"]
    by_day = delta.groupby(f.loc[got, "civil_date"])
    assert (by_day.max() - by_day.min()).abs().max() < 1e-12


def test_no_recent_residual_used_by_a_target_postdates_its_origin():
    f = frame(days=12)
    out = corrected(f).dropna(subset=["level_end"])
    assert (out["level_end"] <= out["origin"]).all()
    assert ((out["origin"] - out["level_start"]) <= pd.Timedelta(hours=24)).all()


def test_future_actuals_cannot_change_an_already_issued_correction():
    f = frame(days=12)
    target = pd.Timestamp("2023-05-08")
    a = corrected(f, targets=[target])
    origin = a.loc[a["available"], "origin"].iloc[0]

    g = f.copy()
    g.loc[g["datetime"] > origin, "y"] += 50_000.0
    b = corrected(g, targets=[target])

    rows = f["civil_date"] == target
    assert np.allclose(a.loc[rows, "yhat"], b.loc[rows, "yhat"], equal_nan=True)
    assert np.allclose(a.loc[rows, "level_offset"],
                       b.loc[rows, "level_offset"], equal_nan=True)


def test_missing_target_actual_does_not_change_forecast_availability():
    f = frame(days=12)
    target = pd.Timestamp("2023-05-08")
    rows = f.index[f["civil_date"] == target]
    f.loc[rows, "y"] = np.nan
    out = corrected(f, targets=[target])
    assert out.loc[rows, "available"].all()


def test_insufficient_recent_residuals_block_the_correction_loudly_by_availability():
    f = frame(days=12)
    target = pd.Timestamp("2023-05-08")
    probe = corrected(f, targets=[target])
    origin = probe.loc[probe["available"], "origin"].iloc[0]
    recent = (f["datetime"] <= origin) & (f["datetime"] > origin - pd.Timedelta(hours=24))
    # Keep only 47 finite realised forecast errors in the recent window.
    keep = f.index[recent][-47:]
    f.loc[f.index[recent].difference(keep), "y"] = np.nan
    out = corrected(f, targets=[target])
    rows = f["civil_date"] == target
    assert not out.loc[rows, "available"].any()


def test_missing_base_forecasts_in_history_are_not_treated_as_zero_error():
    f = frame(days=12)
    target = pd.Timestamp("2023-05-08")
    probe = corrected(f, targets=[target])
    origin = probe.loc[probe["available"], "origin"].iloc[0]
    recent = (f["datetime"] <= origin) & (f["datetime"] > origin - pd.Timedelta(hours=24))
    base = f["base"].copy()
    base.loc[recent] = np.nan
    out = recent_residual_level_correction(
        f, base, targets=[target], window_hours=24, min_rows=48)
    rows = f["civil_date"] == target
    assert not out.loc[rows, "available"].any()


def test_spring_forward_target_keeps_all_92_base_slots():
    f = frame("2023-03-20 00:00Z", days=12)
    target = pd.Timestamp("2023-03-26")
    out = corrected(f, targets=[target])
    rows = f["civil_date"] == target
    assert int(rows.sum()) == 92
    assert out.loc[rows, "available"].all()


def test_fall_back_target_keeps_all_100_base_slots():
    f = frame("2023-10-23 00:00Z", days=12)
    target = pd.Timestamp("2023-10-29")
    out = corrected(f, targets=[target])
    rows = f["civil_date"] == target
    assert int(rows.sum()) == 100
    assert out.loc[rows, "available"].all()
    assert out.loc[rows, "level_offset"].nunique() == 1


def test_invalid_window_and_minimum_are_rejected():
    f = frame()
    with pytest.raises(ValueError):
        recent_residual_level_correction(f, f["base"], window_hours=0)
    with pytest.raises(ValueError):
        recent_residual_level_correction(f, f["base"], min_rows=0)

def test_misaligned_series_index_is_rejected():
    frame = pd.DataFrame({
        "civil_date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
        "datetime": pd.to_datetime(
            ["2023-01-01T00:00:00Z", "2023-01-01T00:15:00Z"],
            utc=True,
        ),
        "y": [100.0, 101.0],
    })

    base = pd.Series(
        [99.0, 100.0],
        index=[10, 11],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="base_forecast index must match frame.index exactly",
    ):
        recent_residual_level_correction(
            frame,
            base,
            targets=[pd.Timestamp("2023-01-01")],
        )

