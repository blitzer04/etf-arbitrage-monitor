"""Tests for exact session alignment without canonical-data mutation."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from etf_monitor.prices.schema import PriceValidationError
from etf_monitor.signals.panel import build_session_panel
from tests.prices.helpers import canonical_frame


def _prices(*frames: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"], kind="mergesort")
        .reset_index(drop=True)
    )


def test_default_panel_uses_global_methodology_start() -> None:
    prices = canonical_frame(day="2018-01-02")

    panel = build_session_panel(prices, "SPY", date(2018, 1, 5))

    assert panel.index.name == "xnys_position"
    assert panel["session"].tolist() == list(
        pd.to_datetime(["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"])
    )
    assert panel["row_present"].tolist() == [True, False, False, False]
    assert panel["row_present"].dtype == np.dtype("bool")
    assert panel["adjusted_close"].dtype == np.dtype("float64")
    assert panel["adjusted_close"].iloc[0] == 100.5


def test_panel_distinguishes_absent_rows_from_present_null_adjusted_close() -> None:
    prices = _prices(
        canonical_frame(day="2018-01-02", adjusted_close=None),
        canonical_frame(day="2018-01-04"),
    )
    before = prices.copy(deep=True)

    panel = build_session_panel(
        prices, "SPY", date(2018, 1, 5), grid_start=date(2018, 1, 2)
    )

    assert panel["row_present"].tolist() == [True, False, True, False]
    assert np.isnan(panel["adjusted_close"].iloc[0])
    assert np.isnan(panel["adjusted_close"].iloc[1])
    assert panel["adjusted_close"].iloc[2] == 100.5
    pd.testing.assert_frame_equal(prices, before)


@pytest.mark.parametrize(
    ("day", "grid_start", "target", "message"),
    [
        ("2018-01-02", date(2018, 1, 3), date(2018, 1, 5), "outside"),
        ("2018-01-05", date(2018, 1, 2), date(2018, 1, 4), "outside"),
    ],
)
def test_canonical_rows_outside_grid_are_rejected(
    day: str, grid_start: date, target: date, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_session_panel(
            canonical_frame(day=day), "SPY", target, grid_start=grid_start
        )


def test_non_session_canonical_date_is_rejected_not_remapped() -> None:
    prices = canonical_frame(day="2018-01-06")

    with pytest.raises(PriceValidationError, match="XNYS sessions"):
        build_session_panel(
            prices,
            "SPY",
            date(2018, 1, 8),
            grid_start=date(2018, 1, 2),
        )


def test_grid_and_target_arguments_are_validated() -> None:
    prices = canonical_frame(day="2018-01-02")

    with pytest.raises(ValueError, match="on or before"):
        build_session_panel(
            prices,
            "SPY",
            date(2018, 1, 5),
            grid_start=date(2018, 1, 8),
        )
    with pytest.raises(ValueError, match="not an XNYS session"):
        build_session_panel(
            prices,
            "SPY",
            date(2018, 1, 6),
            grid_start=date(2018, 1, 2),
        )
    with pytest.raises(TypeError, match="grid_start"):
        build_session_panel(
            prices,
            "SPY",
            date(2018, 1, 5),
            grid_start="2018-01-02",  # type: ignore[arg-type]
        )
