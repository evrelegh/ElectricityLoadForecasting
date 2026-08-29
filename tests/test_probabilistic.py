import numpy as np
import pandas as pd
import pytest

from electricity_load_forecasting.probabilistic import conditional_empirical_intervals

TZ = "Europe/Brussels"


def toy_frame(start="2023-01-01", days=70):
    t = pd.date_range(start, periods=days * 96, freq="15min", tz="UTC")
    civ = t.tz_convert(TZ)
    return pd.DataFrame({
        "datetime": t,
        "civil_date": civ.normalize().tz_localize(None),
        "hour": civ.hour,
        "y": 1000.0 + np.arange(len(t), dtype=float) * 0.0,
    })


def test_known_constant_residual_shifts_both_quantiles():
    f = toy_frame(days=70)
    point = pd.Series(f["y"] - 25.0, index=f.index)
    target = pd.Timestamp("2023-03-05")
    out = conditional_empirical_intervals(
        f, point, targets=[target], window_days=56, min_rows=4
    )
    rows = f["civil_date"].eq(target)
    assert out.loc[rows, "available"].all()
    assert np.allclose(out.loc[rows, "q_lo"], f.loc[rows, "y"])
    assert np.allclose(out.loc[rows, "q_hi"], f.loc[rows, "y"])


def test_post_origin_actual_perturbation_has_no_effect():
    f = toy_frame(days=70)
    point = pd.Series(f["y"] - 10.0, index=f.index)
    target = pd.Timestamp("2023-03-05")
    a = conditional_empirical_intervals(f, point, targets=[target], min_rows=4)
    origin = a.loc[a["available"], "origin"].iloc[0]
    g = f.copy()
    g.loc[g["datetime"] > origin, "y"] += 1_000_000.0
    b = conditional_empirical_intervals(g, point, targets=[target], min_rows=4)
    rows = f["civil_date"].eq(target)
    assert np.allclose(a.loc[rows, "q_lo"], b.loc[rows, "q_lo"], equal_nan=True)
    assert np.allclose(a.loc[rows, "q_hi"], b.loc[rows, "q_hi"], equal_nan=True)


def test_positive_control_pre_origin_shift_moves_quantiles_exactly():
    f = toy_frame(days=70)
    point = pd.Series(f["y"] - 10.0, index=f.index)
    target = pd.Timestamp("2023-03-05")
    a = conditional_empirical_intervals(f, point, targets=[target], min_rows=4)
    origin = a.loc[a["available"], "origin"].iloc[0]
    start = origin - pd.Timedelta(days=56)
    g = f.copy()
    legitimate = (g["datetime"] <= origin) & (g["datetime"] > start)
    g.loc[legitimate, "y"] += 500.0
    b = conditional_empirical_intervals(g, point, targets=[target], min_rows=4)
    rows = f["civil_date"].eq(target)
    assert np.allclose(b.loc[rows, "q_lo"] - a.loc[rows, "q_lo"], 500.0)
    assert np.allclose(b.loc[rows, "q_hi"] - a.loc[rows, "q_hi"], 500.0)


def test_civil_hour_conditioning_is_respected():
    f = toy_frame(days=70)
    residual = f["hour"].astype(float) * 100.0
    point = pd.Series(f["y"] - residual, index=f.index)
    target = pd.Timestamp("2023-03-05")
    out = conditional_empirical_intervals(f, point, targets=[target], min_rows=4)
    rows = f["civil_date"].eq(target)
    expected = f.loc[rows, "y"].to_numpy()
    assert np.allclose(out.loc[rows, "q_lo"], expected)
    assert np.allclose(out.loc[rows, "q_hi"], expected)


def test_insufficient_history_blocks_interval():
    f = toy_frame(days=10)
    point = pd.Series(f["y"] - 10.0, index=f.index)
    target = pd.Timestamp("2023-01-10")
    out = conditional_empirical_intervals(f, point, targets=[target], min_rows=999)
    assert not out["available"].any()


def test_missing_target_actual_does_not_block_interval():
    f = toy_frame(days=70)
    point = pd.Series(f["y"] - 10.0, index=f.index)
    target = pd.Timestamp("2023-03-05")
    f.loc[f["civil_date"].eq(target), "y"] = np.nan
    out = conditional_empirical_intervals(f, point, targets=[target], min_rows=4)
    assert out.loc[f["civil_date"].eq(target), "available"].all()


def test_quantiles_never_cross():
    f = toy_frame(days=70)
    point = pd.Series(f["y"] - np.sin(np.arange(len(f))), index=f.index)
    out = conditional_empirical_intervals(
        f, point, targets=[pd.Timestamp("2023-03-05")], min_rows=4
    )
    used = out["available"]
    assert (out.loc[used, "q_lo"] <= out.loc[used, "q_hi"]).all()


def test_history_end_never_exceeds_origin():
    f = toy_frame(days=70)
    point = pd.Series(f["y"] - 10.0, index=f.index)
    out = conditional_empirical_intervals(
        f, point, targets=[pd.Timestamp("2023-03-05")], min_rows=4
    )
    used = out["available"]
    assert (out.loc[used, "history_end"] <= out.loc[used, "origin"]).all()


def test_misaligned_series_index_is_rejected():
    f = toy_frame(days=2)
    point = pd.Series(np.zeros(len(f)), index=np.arange(1000, 1000 + len(f)))
    with pytest.raises(ValueError, match="point_forecast index must match frame.index exactly"):
        conditional_empirical_intervals(f, point, min_rows=1)
