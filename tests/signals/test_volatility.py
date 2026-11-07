"""Exact-window annualized Volatility formula and missingness tests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
import pytest

from etf_monitor.signals.volatility import calculate_volatility
from etf_monitor.xnys import xnys_sessions
from tests.signals.helpers import price_frame


def _sessions(count: int) -> pd.DatetimeIndex:
    return xnys_sessions(date(2018, 1, 2), date(2025, 12, 31))[:count]


def _calculate(
    sessions: pd.DatetimeIndex,
    adjusted: Sequence[float | None],
    *,
    missing_rows: set[int] | None = None,
    return_count: int = 21,
    reference_window: int = 756,
    minimum_reference: int = 1,
) -> pd.DataFrame:
    return calculate_volatility(
        price_frame("SPY", sessions, adjusted, missing_rows=missing_rows),
        "SPY",
        sessions[-1],
        grid_start=sessions[0],
        return_count=return_count,
        reference_window_sessions=reference_window,
        minimum_reference_observations=minimum_reference,
    )


def test_k3_hand_example_uses_sample_std_and_sqrt_252() -> None:
    sessions = _sessions(4)
    result = _calculate(
        sessions,
        [100.0, 110.0, 99.0, 108.9],
        return_count=3,
    )

    # The three adjacent returns are log(1.1), log(0.9), log(1.1).
    returns = np.array([np.log(1.1), np.log(0.9), np.log(1.1)])
    mean_return = (returns[0] + returns[1] + returns[2]) / 3.0
    squared_deviations = sum((value - mean_return) ** 2 for value in returns)
    sample_std = math.sqrt(squared_deviations / 2.0)
    expected = math.sqrt(252.0) * sample_std

    assert result["status"].iloc[2] == "insufficient_history"
    assert result["status"].iloc[3] == "eligible"
    assert result["window_start_session"].iloc[3] == sessions[0]
    assert result["window_end_session"].iloc[3] == sessions[3]
    assert result["raw_annualized_vol"].iloc[3] == pytest.approx(
        expected, rel=1e-15, abs=0.0
    )
    assert result["display_vol_pct"].iloc[3] == pytest.approx(
        100.0 * expected, rel=1e-15, abs=0.0
    )


def test_k21_uses_ddof_one_denominator_twenty() -> None:
    sessions = _sessions(22)
    adjusted = [100.0]
    for position in range(21):
        daily_return = 0.01 if position < 20 else 0.02
        adjusted.append(adjusted[-1] * math.exp(daily_return))
    result = _calculate(sessions, adjusted)

    log_returns = np.log(
        np.asarray(adjusted[1:], dtype=np.float64)
        / np.asarray(adjusted[:-1], dtype=np.float64)
    )
    mean_return = sum(float(value) for value in log_returns) / 21.0
    squared_deviations = sum((float(value) - mean_return) ** 2 for value in log_returns)
    expected = math.sqrt(252.0) * math.sqrt(squared_deviations / 20.0)

    assert result["raw_annualized_vol"].iloc[-1] == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


def test_constant_log_returns_produce_valid_exact_zero_volatility() -> None:
    sessions = _sessions(40)
    adjusted: list[float | None] = [2.0**position for position in range(len(sessions))]
    result = _calculate(sessions, adjusted)

    assert result["status"].iloc[21] == "eligible"
    assert result["raw_annualized_vol"].iloc[21] == 0.0
    assert result["display_vol_pct"].iloc[21] == 0.0


def test_one_missing_middle_price_invalidates_exactly_22_windows() -> None:
    sessions = _sessions(70)
    adjusted: list[float | None] = [2.0**position for position in range(len(sessions))]
    adjusted[30] = None
    result = _calculate(sessions, adjusted)

    expected_missing = set(range(30, 52))
    actual_missing = {
        position
        for position, status in enumerate(result["status"])
        if status == "missing_adjusted_close_in_window"
    }
    assert actual_missing == expected_missing
    assert result["window_missing_adjclose_count"].iloc[30:52].gt(0).all()
    assert result["status"].iloc[29] == "eligible"
    assert result["status"].iloc[52] == "eligible"


def test_missing_row_invalidates_exactly_the_windows_containing_it() -> None:
    sessions = _sessions(70)
    adjusted = [2.0**position for position in range(len(sessions))]
    result = _calculate(sessions, adjusted, missing_rows={30})

    expected_missing = set(range(30, 52))
    actual_missing = {
        position
        for position, status in enumerate(result["status"])
        if status == "missing_row_in_window"
    }
    assert actual_missing == expected_missing
    assert result["window_missing_row_count"].iloc[30:52].gt(0).all()


def test_first_raw_normalized_and_first_use_boundaries() -> None:
    sessions = _sessions(275)
    adjusted = [100.0 + position for position in range(len(sessions))]
    result = _calculate(
        sessions,
        adjusted,
        reference_window=756,
        minimum_reference=252,
    )

    assert result["status"].iloc[20] == "insufficient_history"
    assert result["status"].iloc[21] == "eligible"
    assert result["normalization_status"].iloc[272] == "insufficient_reference"
    assert result["normalization_status"].iloc[273] == "available"
    assert result["first_use_session"].iloc[273] == sessions[274]
