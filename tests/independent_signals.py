"""Deliberately independent loop-based references for signal regression tests.

This module uses Python dictionaries and scalar math only for the calculations;
it does not import production signal code.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd


def _reference_percentiles(
    raw_values: list[float | None],
    sessions: pd.DatetimeIndex,
    *,
    reference_window_sessions: int,
    minimum_reference_observations: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for position, current in enumerate(raw_values):
        first_reference_position = max(0, position - reference_window_sessions + 1)
        reference_positions: list[int] = []
        reference_values: list[float] = []
        for reference_position in range(first_reference_position, position):
            reference_value = raw_values[reference_position]
            if reference_value is not None:
                reference_positions.append(reference_position)
                reference_values.append(reference_value)
        count = len(reference_positions)
        first_session = (
            sessions[reference_positions[0]] if reference_positions else None
        )
        last_session = (
            sessions[reference_positions[-1]] if reference_positions else None
        )
        if current is None:
            status = "raw_ineligible"
            percentile = None
        elif count < minimum_reference_observations:
            status = "insufficient_reference"
            percentile = None
        else:
            lower_count = sum(value < current for value in reference_values)
            equal_count = sum(value == current for value in reference_values)
            percentile = 100.0 * (lower_count + 0.5 * equal_count) / count
            status = "available"
        normalized.append(
            {
                "percentile": percentile,
                "reference_count": count,
                "reference_first_session": first_session,
                "reference_last_session": last_session,
                "normalization_status": status,
            }
        )
    return normalized


def calculate_independent_signals(
    prices: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    tickers: Sequence[str],
    *,
    momentum_horizon_sessions: int = 252,
    volatility_return_count: int = 21,
    volatility_ddof: int = 1,
    annualization_sessions: int = 252,
    reference_window_sessions: int = 756,
    minimum_reference_observations: int = 252,
) -> tuple[
    dict[tuple[str, pd.Timestamp], dict[str, Any]],
    dict[tuple[str, pd.Timestamp], dict[str, Any]],
]:
    """Calculate Momentum and Volatility with scalar loops and keyed prices."""
    prices_by_ticker: dict[str, dict[pd.Timestamp, float | None]] = {
        ticker: {} for ticker in tickers
    }
    for row in prices.to_dict(orient="records"):
        ticker = str(row["ticker"])
        session = pd.Timestamp(row["date"])
        if ticker not in prices_by_ticker:
            continue
        row_adjusted_close = row["adjusted_close"]
        adjusted_close = (
            None if pd.isna(row_adjusted_close) else float(row_adjusted_close)
        )
        prices_by_ticker[ticker][session] = adjusted_close

    momentum_results: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
    volatility_results: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
    for ticker in tickers:
        ticker_prices = prices_by_ticker[ticker]
        row_present = [session in ticker_prices for session in sessions]
        adjusted_closes = [ticker_prices.get(session) for session in sessions]
        valid_adjusted = [
            value is not None and math.isfinite(value) and value > 0.0
            for value in adjusted_closes
        ]

        momentum_raw: list[float | None] = []
        momentum_rows: list[dict[str, Any]] = []
        for position, _session in enumerate(sessions):
            start_position = position - momentum_horizon_sessions
            start_session = sessions[start_position] if start_position >= 0 else None
            start_value = (
                adjusted_closes[start_position] if start_position >= 0 else None
            )
            end_value = adjusted_closes[position]
            if start_position < 0:
                status = "insufficient_history"
                raw_value = None
                simple_return_pct = None
                missing_rows = 0
                missing_adjusted = 0
            else:
                missing_rows = sum(
                    not row_present[interior_position]
                    for interior_position in range(start_position + 1, position)
                )
                missing_adjusted = sum(
                    row_present[interior_position]
                    and not valid_adjusted[interior_position]
                    for interior_position in range(start_position + 1, position)
                )
                if not row_present[position]:
                    status = "missing_end_row"
                    raw_value = None
                    simple_return_pct = None
                elif not valid_adjusted[position]:
                    status = "missing_end_adjusted_close"
                    raw_value = None
                    simple_return_pct = None
                elif not row_present[start_position]:
                    status = "missing_start_row"
                    raw_value = None
                    simple_return_pct = None
                elif not valid_adjusted[start_position]:
                    status = "missing_start_adjusted_close"
                    raw_value = None
                    simple_return_pct = None
                else:
                    assert start_value is not None and end_value is not None
                    ratio = end_value / start_value
                    raw_value = math.log(ratio)
                    simple_return_pct = 100.0 * (ratio - 1.0)
                    status = "eligible"
            momentum_raw.append(raw_value)
            momentum_rows.append(
                {
                    "status": status,
                    "start_session": start_session,
                    "start_adjusted_close": start_value,
                    "end_adjusted_close": end_value,
                    "raw_log_return": raw_value,
                    "simple_return_pct": simple_return_pct,
                    "interior_missing_row_count": missing_rows,
                    "interior_missing_adjclose_count": missing_adjusted,
                }
            )

        momentum_normalized = _reference_percentiles(
            momentum_raw,
            sessions,
            reference_window_sessions=reference_window_sessions,
            minimum_reference_observations=minimum_reference_observations,
        )

        volatility_raw: list[float | None] = []
        volatility_rows: list[dict[str, Any]] = []
        for position, session in enumerate(sessions):
            start_position = position - volatility_return_count
            if start_position < 0:
                status = "insufficient_history"
                raw_value = None
                missing_rows = 0
                missing_adjusted = 0
                start_session = None
                end_session = None
            else:
                start_session = sessions[start_position]
                end_session = session
                missing_rows = sum(
                    not row_present[window_position]
                    for window_position in range(start_position, position + 1)
                )
                missing_adjusted = sum(
                    row_present[window_position] and not valid_adjusted[window_position]
                    for window_position in range(start_position, position + 1)
                )
                if missing_rows:
                    status = "missing_row_in_window"
                    raw_value = None
                elif missing_adjusted:
                    status = "missing_adjusted_close_in_window"
                    raw_value = None
                else:
                    returns: list[float] = []
                    for window_position in range(start_position + 1, position + 1):
                        previous = adjusted_closes[window_position - 1]
                        current = adjusted_closes[window_position]
                        assert previous is not None and current is not None
                        returns.append(math.log(current / previous))
                    mean_return = sum(returns) / volatility_return_count
                    sum_squared_deviations = sum(
                        (value - mean_return) ** 2 for value in returns
                    )
                    sample_variance = sum_squared_deviations / (
                        volatility_return_count - volatility_ddof
                    )
                    raw_value = math.sqrt(annualization_sessions) * math.sqrt(
                        sample_variance
                    )
                    status = "eligible"
            volatility_raw.append(raw_value)
            volatility_rows.append(
                {
                    "status": status,
                    "window_start_session": start_session,
                    "window_end_session": end_session,
                    "raw_annualized_vol": raw_value,
                    "display_vol_pct": (
                        None if raw_value is None else 100.0 * raw_value
                    ),
                    "window_missing_row_count": missing_rows,
                    "window_missing_adjclose_count": missing_adjusted,
                }
            )

        volatility_normalized = _reference_percentiles(
            volatility_raw,
            sessions,
            reference_window_sessions=reference_window_sessions,
            minimum_reference_observations=minimum_reference_observations,
        )
        for position, session in enumerate(sessions):
            key = (ticker, pd.Timestamp(session))
            momentum_results[key] = (
                momentum_rows[position] | momentum_normalized[position]
            )
            volatility_results[key] = (
                volatility_rows[position] | volatility_normalized[position]
            )
    return momentum_results, volatility_results
