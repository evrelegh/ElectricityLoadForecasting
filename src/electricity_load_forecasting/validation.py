# ══════════════════════════════════════════════════════════════════════════
# validation · temporal-integrity guards                            v0.2.0
# provides ▸ LookaheadError · assert_no_lookahead
# requires ▸ pandas
# ══════════════════════════════════════════════════════════════════════════
"""Guards against using information that did not exist at the forecast origin.

Nothing in the project forecasts yet. The guard is written first so that when
the rolling-origin framework arrives it is checked by construction rather than
by inspection: every feature timestamp must be at or before the origin, and
every target timestamp strictly after it.
"""

from __future__ import annotations

import pandas as pd


class LookaheadError(AssertionError):
    """A feature or target violates the forecast origin."""


def assert_no_lookahead(feature_times, origin, target_times=None) -> None:
    """Raise unless features precede the origin and targets follow it."""
    origin = pd.Timestamp(origin)
    ft = pd.DatetimeIndex(pd.to_datetime(feature_times))
    late = ft[ft > origin]
    if len(late):
        raise LookaheadError(
            f"{len(late)} feature timestamp(s) after origin {origin}; "
            f"first offender {late[0]}")

    if target_times is not None:
        tt = pd.DatetimeIndex(pd.to_datetime(target_times))
        early = tt[tt <= origin]
        if len(early):
            raise LookaheadError(
                f"{len(early)} target timestamp(s) at or before origin {origin}; "
                f"first offender {early[0]}")
