import numpy as np
import pandas as pd

from electricity_load_forecasting.level import recent_residual_level_correction


def test_positive_control_legitimate_recent_errors_move_offset_exactly():
    t = pd.date_range("2023-01-01", periods=12 * 96, freq="15min", tz="UTC")
    civ = t.tz_convert("Europe/Brussels")
    f = pd.DataFrame({
        "datetime": t,
        "civil_date": civ.normalize().tz_localize(None),
        "qod": civ.hour * 4 + civ.minute // 15,
        "y": 1000.0,
    })
    base = pd.Series(990.0, index=f.index)
    target = pd.Timestamp("2023-01-10")

    a = recent_residual_level_correction(
        f, base, targets=[target], window_hours=24, min_rows=48
    )
    origin = a.loc[a["available"], "origin"].iloc[0]

    g = f.copy()
    legitimate = (g["datetime"] <= origin) & (g["datetime"] > origin - pd.Timedelta(hours=24))
    g.loc[legitimate, "y"] += 500.0

    b = recent_residual_level_correction(
        g, base, targets=[target], window_hours=24, min_rows=48
    )

    rows = f["civil_date"].eq(target)
    assert np.allclose(b.loc[rows, "level_offset"] - a.loc[rows, "level_offset"], 500.0)
    assert np.allclose(b.loc[rows, "yhat"] - a.loc[rows, "yhat"], 500.0)
