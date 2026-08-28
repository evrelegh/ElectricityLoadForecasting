"""The temporal-integrity guard, written before the forecasting code it guards."""

import pandas as pd
import pytest

from electricity_load_forecasting.validation import LookaheadError, assert_no_lookahead

ORIGIN = pd.Timestamp("2023-06-15 12:00")


def test_features_before_the_origin_and_targets_after_it_pass():
    assert_no_lookahead(
        pd.date_range("2023-06-14 12:00", ORIGIN, freq="15min"),
        ORIGIN,
        pd.date_range("2023-06-16 00:00", periods=96, freq="15min"))


def test_a_feature_after_the_origin_is_rejected():
    with pytest.raises(LookaheadError, match="after origin"):
        assert_no_lookahead(pd.date_range(ORIGIN, periods=3, freq="15min")[1:], ORIGIN)


def test_a_feature_exactly_at_the_origin_is_allowed():
    assert_no_lookahead([ORIGIN], ORIGIN)


def test_a_target_at_or_before_the_origin_is_rejected():
    with pytest.raises(LookaheadError, match="before origin"):
        assert_no_lookahead([ORIGIN], ORIGIN, [ORIGIN])
