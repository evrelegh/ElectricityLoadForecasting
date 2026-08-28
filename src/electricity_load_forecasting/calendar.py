# ══════════════════════════════════════════════════════════════════════════
# calendar · Belgian civil time, DST-aware day length, public holidays v0.3.0
# provides ▸ easter · be_holidays · expected_slots · civil_features
#            complete_civil_days
# requires ▸ pandas
# ══════════════════════════════════════════════════════════════════════════
"""Civil-time primitives.

Electricity demand follows the human clock, so calendar features are built in
Europe/Brussels while the computational axis stays UTC. The number of quarter
hours in a civil day is derived from the zone rather than assumed, so the DST
transitions need no special case and a future change to the EU clock rules
does not silently invalidate the series.
"""

from __future__ import annotations

import pandas as pd

TZ_BE = "Europe/Brussels"
FREQ_QH = "15min"


def easter(year: int) -> pd.Timestamp:
    """Gregorian Easter Sunday — anonymous algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = (h + m - 7 * n + 114) % 31 + 1
    return pd.Timestamp(year=year, month=month, day=day)


def be_holidays(years) -> pd.DatetimeIndex:
    """The ten Belgian public holidays per year — fixed plus Easter-derived.

    Regional and sector holidays are out of scope: only the federal days that
    visibly move national load are included.
    """
    out: list[pd.Timestamp] = []
    for y in years:
        e = easter(int(y))
        out += [
            pd.Timestamp(f"{y}-01-01"),          # New Year
            e + pd.Timedelta(days=1),            # Easter Monday
            pd.Timestamp(f"{y}-05-01"),          # Labour Day
            e + pd.Timedelta(days=39),           # Ascension
            e + pd.Timedelta(days=50),           # Whit Monday
            pd.Timestamp(f"{y}-07-21"),          # National Day
            pd.Timestamp(f"{y}-08-15"),          # Assumption
            pd.Timestamp(f"{y}-11-01"),          # All Saints
            pd.Timestamp(f"{y}-11-11"),          # Armistice
            pd.Timestamp(f"{y}-12-25"),          # Christmas
        ]
    return pd.DatetimeIndex(sorted(out))


def expected_slots(day, tz: str = TZ_BE, freq: str = FREQ_QH) -> int:
    """Number of `freq` slots in the civil day `day`, taken from the zone.

    Returns 96 on an ordinary quarter-hour day, 92 when the clock springs
    forward and 100 when it falls back. Local midnight is never ambiguous in
    Europe/Brussels, so the two localisations are safe.
    """
    day = pd.Timestamp(day).normalize()
    a = pd.Timestamp(day, tz=tz)
    b = pd.Timestamp(day + pd.Timedelta(days=1), tz=tz)
    return int((b - a) / pd.Timedelta(freq))


def civil_features(utc_index: pd.Series, tz: str = TZ_BE) -> pd.DataFrame:
    """Calendar features on the civil clock, from a tz-aware UTC index.

    `qod` is the quarter of the civil day, 0..95. On the fall-back day two
    distinct UTC instants share a qod; that is a property of civil time, not
    a defect, and the UTC axis remains the unique key.
    """
    civ = pd.to_datetime(utc_index, utc=True).dt.tz_convert(tz)
    f = pd.DataFrame(index=getattr(utc_index, "index", None))
    f["civil_date"] = civ.dt.normalize().dt.tz_localize(None)
    f["qod"] = civ.dt.hour * 4 + civ.dt.minute // 15
    f["hour"] = civ.dt.hour
    f["dow"] = civ.dt.dayofweek
    f["month"] = civ.dt.month
    f["is_weekend"] = f["dow"] >= 5
    f["is_holiday"] = f["civil_date"].isin(be_holidays(sorted(set(f["civil_date"].dt.year))))
    return f


def complete_civil_days(frame: pd.DataFrame, start, end, tz: str = TZ_BE,
                        freq: str = FREQ_QH, date_col: str = "civil_date"):
    """Restrict `frame` to whole civil days inside [start, end].

    A civil day is kept only when it holds exactly the number of slots the zone
    implies. Partial days at the window edge are dropped rather than trimmed:
    a lag across one would silently reuse the wrong clock time. Returns the
    restricted frame and the dropped day lengths, so the caller can report
    what went rather than discover it later.
    """
    f = frame[frame[date_col].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    per_day = f.groupby(date_col).size()
    exp = per_day.index.to_series().map(lambda d: expected_slots(d, tz, freq))
    incomplete = per_day[per_day != exp]
    f = f[~f[date_col].isin(incomplete.index)].reset_index(drop=True)
    return f, incomplete
