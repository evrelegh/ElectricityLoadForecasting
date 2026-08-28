"""Civil-time behaviour: the DST day length and the holiday calendar."""

import pandas as pd
import pytest

from electricity_load_forecasting.calendar import (
    be_holidays, civil_features, easter, expected_slots)


@pytest.mark.parametrize("year,expected", [
    (2022, "2022-04-17"), (2023, "2023-04-09"), (2024, "2024-03-31"),
    (2025, "2025-04-20"), (2026, "2026-04-05"), (2038, "2038-04-25"),
])
def test_easter_matches_known_dates(year, expected):
    assert easter(year) == pd.Timestamp(expected)


@pytest.mark.parametrize("day,slots", [
    ("2023-03-26", 92),    # clocks forward, an hour is lost
    ("2023-10-29", 100),   # clocks back, an hour is repeated
    ("2023-06-15", 96),    # ordinary day
    ("2023-01-01", 96),    # the year boundary is ordinary in civil time
    ("2024-03-31", 92),
    ("2024-10-27", 100),
])
def test_expected_slots_follows_the_zone(day, slots):
    assert expected_slots(day) == slots


def test_civil_year_has_exactly_365_times_96_slots():
    """The two DST days cancel, so a non-leap civil year is 35040 quarters —
    the invariant the notebook's data contract relies on."""
    days = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    assert sum(expected_slots(d) for d in days) == 365 * 96


def test_holidays_are_ten_per_year_and_include_moveable_feasts():
    h = be_holidays([2023])
    assert len(h) == 10
    for d in ["2023-01-01", "2023-04-10", "2023-05-18", "2023-05-29",
              "2023-07-21", "2023-11-11", "2023-12-25"]:
        assert pd.Timestamp(d) in h


def test_holidays_track_easter_across_years():
    """Moveable feasts must move: a fixed offset table would fail this."""
    for y in (2023, 2024, 2025):
        assert easter(y) + pd.Timedelta(days=1) in be_holidays([y])


def test_civil_features_use_the_local_clock_not_utc():
    """Belgium is UTC+1 in winter, so 23:00Z is already the next civil day."""
    idx = pd.Series(pd.to_datetime(["2022-12-31 23:00Z", "2023-06-15 12:00Z"]))
    f = civil_features(idx)
    assert f.loc[0, "civil_date"] == pd.Timestamp("2023-01-01")
    assert f.loc[0, "qod"] == 0
    assert bool(f.loc[0, "is_holiday"])                 # New Year, in civil time
    assert f.loc[1, "hour"] == 14                       # UTC+2 in summer


def test_fallback_day_repeats_a_quarter_of_day_but_not_a_utc_instant():
    idx = pd.Series(pd.date_range("2023-10-29 00:00Z", periods=100, freq="15min"))
    f = civil_features(idx)
    assert idx.is_unique
    assert (f["qod"].value_counts() == 2).sum() == 4     # the repeated hour
    assert f["qod"].between(0, 95).all()


def test_complete_civil_days_keeps_whole_days_and_reports_the_rest():
    """Warm-up history and the analysis window use the same rule, so it lives
    in one place: a partial day at the edge is dropped, not trimmed."""
    from electricity_load_forecasting.calendar import complete_civil_days
    idx = pd.Series(pd.date_range("2023-03-25 00:00Z", periods=96 * 4, freq="15min"))
    f = pd.concat([pd.DataFrame({"datetime": idx}), civil_features(idx)], axis=1)
    kept, dropped = complete_civil_days(f, "2023-03-25", "2023-03-28")
    assert list(dropped.index) == [pd.Timestamp("2023-03-25")]     # starts at 01:00 civil
    assert kept.groupby("civil_date").size().to_dict() == {
        pd.Timestamp("2023-03-26"): 92,                            # DST day kept whole
        pd.Timestamp("2023-03-27"): 96,
        pd.Timestamp("2023-03-28"): 96,
    }


def test_complete_civil_days_does_not_discard_the_transition_days():
    from electricity_load_forecasting.calendar import complete_civil_days
    idx = pd.Series(pd.date_range("2023-10-28 00:00Z", periods=96 * 3, freq="15min"))
    f = pd.concat([pd.DataFrame({"datetime": idx}), civil_features(idx)], axis=1)
    kept, _ = complete_civil_days(f, "2023-10-29", "2023-10-29")
    assert len(kept) == 100
