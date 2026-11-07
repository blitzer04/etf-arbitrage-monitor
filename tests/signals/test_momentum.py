"""Hand-checkable adjusted-close Momentum endpoint and timing tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from etf_monitor.signals.momentum import calculate_momentum
from etf_monitor.xnys import xnys_sessions
from tests.signals.helpers import price_frame


def _sessions(count: int) -> pd.DatetimeIndex:
    return xnys_sessions(date(2018, 1, 2), date(2025, 12, 31))[:count]


def test_tiny_horizon_uses_exact_adjusted_endpoints_and_midrank_history() -> None:
    sessions = _sessions(10)
    adjusted = [100.0, 101.0, 103.0, 106.0, 110.0, 115.0, 121.0, 128.0, 136.0, 145.0]
    result = calculate_momentum(
        price_frame("SPY", sessions, adjusted),
        "SPY",
        sessions[-1],
        grid_start=sessions[0],
        horizon_sessions=3,
        reference_window_sessions=5,
        minimum_reference_observations=2,
    )

    assert result["status"].iloc[2] == "insufficient_history"
    assert result["status"].iloc[3] == "eligible"
    assert result["start_session"].iloc[3] == sessions[0]
    assert result["signal_date"].iloc[3] == sessions[3]
    assert result["start_adjusted_close"].iloc[3] == 100.0
    assert result["end_adjusted_close"].iloc[3] == 106.0
    assert result["raw_log_return"].iloc[3] == float(np.log(106.0 / 100.0))
    assert result["simple_return_pct"].iloc[3] == 100.0 * (106.0 / 100.0 - 1.0)
    assert result["normalization_status"].iloc[4] == "insufficient_reference"
    assert result["normalization_status"].iloc[5] == "available"
    assert result["first_use_session"].iloc[3] == sessions[4]


def test_interior_missing_rows_and_prices_are_diagnostics_only() -> None:
    sessions = _sessions(8)
    adjusted = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0]
    complete = calculate_momentum(
        price_frame("SPY", sessions, adjusted),
        "SPY",
        sessions[-1],
        grid_start=sessions[0],
        horizon_sessions=3,
        reference_window_sessions=5,
        minimum_reference_observations=1,
    )
    incomplete = calculate_momentum(
        price_frame(
            "SPY",
            sessions,
            [100.0, 102.0, 104.0, 106.0, None, 110.0, 112.0, 114.0],
            missing_rows={5},
        ),
        "SPY",
        sessions[-1],
        grid_start=sessions[0],
        horizon_sessions=3,
        reference_window_sessions=5,
        minimum_reference_observations=1,
    )

    # At t=6, only positions 4 and 5 are interior; their missing states do not
    # alter the exact 106 -> 112 endpoint return.
    assert complete["raw_log_return"].iloc[6] == float(np.log(112.0 / 106.0))
    assert incomplete["raw_log_return"].iloc[6] == complete["raw_log_return"].iloc[6]
    assert incomplete["interior_missing_adjclose_count"].iloc[6] == 1
    assert incomplete["interior_missing_row_count"].iloc[6] == 1


@pytest.mark.parametrize(
    ("missing_rows", "null_adjusted", "position", "expected_status"),
    [
        ({0, 3}, set(), 3, "missing_end_row"),
        ({0}, {3}, 3, "missing_end_adjusted_close"),
        ({0}, set(), 3, "missing_start_row"),
        (set(), {0}, 3, "missing_start_adjusted_close"),
    ],
)
def test_endpoint_status_precedence_never_shifts_positions(
    missing_rows: set[int],
    null_adjusted: set[int],
    position: int,
    expected_status: str,
) -> None:
    sessions = _sessions(5)
    adjusted: list[float | None] = [
        None if index in null_adjusted else 100.0 + index for index in range(5)
    ]
    result = calculate_momentum(
        price_frame("SPY", sessions, adjusted, missing_rows=missing_rows),
        "SPY",
        sessions[-1],
        grid_start=sessions[0],
        horizon_sessions=3,
        reference_window_sessions=5,
        minimum_reference_observations=1,
    )

    assert result["status"].iloc[position] == expected_status
    assert np.isnan(result["raw_log_return"].iloc[position])
    assert result["signal_date"].iloc[position] == sessions[position]


def test_full_horizon_252_boundary_and_first_normalized_use_dates() -> None:
    sessions = _sessions(506)
    adjusted = [100.0] * len(sessions)
    adjusted[252] = 110.0
    result = calculate_momentum(
        price_frame("SPY", sessions, adjusted),
        "SPY",
        sessions[-1],
        grid_start=sessions[0],
    )

    assert result["status"].iloc[251] == "insufficient_history"
    assert result["status"].iloc[252] == "eligible"
    assert result["raw_log_return"].iloc[252] == float(np.log(110.0 / 100.0))
    assert result["simple_return_pct"].iloc[252] == 100.0 * (110.0 / 100.0 - 1.0)
    assert result["normalization_status"].iloc[503] == "insufficient_reference"
    assert result["normalization_status"].iloc[504] == "available"
    assert result["signal_date"].iloc[252] == sessions[252]
    assert result["signal_date"].iloc[504] == sessions[504]
    assert result["first_use_session"].iloc[504] == sessions[505]


def test_end_failure_takes_precedence_over_start_failure() -> None:
    sessions = _sessions(5)
    result = calculate_momentum(
        price_frame(
            "SPY",
            sessions,
            [100.0, 101.0, 102.0, None, 104.0],
            missing_rows={0, 1},
        ),
        "SPY",
        sessions[-1],
        grid_start=sessions[0],
        horizon_sessions=3,
        reference_window_sessions=5,
        minimum_reference_observations=1,
    )

    # The exact end has a present row but null adjusted_close; start position
    # zero is absent. End eligibility is checked first.
    assert result["status"].iloc[3] == "missing_end_adjusted_close"
    assert result["status"].iloc[4] == "missing_start_row"


def test_insufficient_history_precedes_a_missing_end_row() -> None:
    sessions = _sessions(4)
    result = calculate_momentum(
        price_frame(
            "SPY",
            sessions,
            [100.0, 101.0, 102.0, 103.0],
            missing_rows={2},
        ),
        "SPY",
        sessions[-1],
        grid_start=sessions[0],
        horizon_sessions=3,
        reference_window_sessions=5,
        minimum_reference_observations=1,
    )

    assert result["status"].iloc[2] == "insufficient_history"
