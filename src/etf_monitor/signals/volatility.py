"""Standalone 21-return annualized adjusted-close Volatility diagnostic."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from etf_monitor.methodology import load_methodology
from etf_monitor.signals.panel import build_session_panel
from etf_monitor.signals.percentile import normalize_midrank_percentiles
from etf_monitor.xnys import next_session


def _positive_integer(value: int | None, default: int, name: str) -> int:
    selected = default if value is None else value
    if type(selected) is not int or selected <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return selected


def calculate_volatility(
    prices: pd.DataFrame,
    ticker: str,
    target: date | pd.Timestamp,
    *,
    grid_start: date | pd.Timestamp | None = None,
    return_count: int | None = None,
    ddof: int | None = None,
    annualization_sessions: int | None = None,
    reference_window_sessions: int | None = None,
    minimum_reference_observations: int | None = None,
) -> pd.DataFrame:
    """Calculate exact-window annualized Volatility for one ETF.

    Each value uses exactly ``return_count`` adjacent log returns, requiring
    one additional exact adjusted-close session price. Window overrides are
    intended for small unit tests; evaluation code must use the methodology
    defaults.

    Args:
        prices: Canonical price frame restricted to the panel's target range.
        ticker: Configured ETF to calculate.
        target: Inclusive target XNYS session.
        grid_start: Optional test-only session-grid start; see
            :func:`build_session_panel`.
        return_count: Optional test override for the number of adjacent
            returns.
        ddof: Optional test override for the sample-standard-deviation
            denominator adjustment.
        annualization_sessions: Optional test override for annualization.
        reference_window_sessions: Optional test override for percentile
            history length.
        minimum_reference_observations: Optional test override for the
            percentile reference minimum.

    Returns:
        One row per panel session with raw, normalized, window, timing, and
        missingness diagnostics.
    """
    methodology = load_methodology()
    returns_required = _positive_integer(
        return_count,
        methodology.volatility_return_count,
        "return_count",
    )
    selected_ddof = methodology.volatility_ddof if ddof is None else ddof
    if type(selected_ddof) is not int or not 0 <= selected_ddof < returns_required:
        raise ValueError("ddof must be an integer from zero to return_count - 1.")
    annualization = _positive_integer(
        annualization_sessions,
        methodology.annualization_sessions,
        "annualization_sessions",
    )
    reference_window = _positive_integer(
        reference_window_sessions,
        methodology.reference_window_sessions,
        "reference_window_sessions",
    )
    reference_minimum = _positive_integer(
        minimum_reference_observations,
        methodology.minimum_reference_observations,
        "minimum_reference_observations",
    )

    panel = build_session_panel(prices, ticker, target, grid_start=grid_start)
    sessions = pd.DatetimeIndex(panel["session"])
    row_present = panel["row_present"].to_numpy(dtype=np.bool_)
    adjusted_close = panel["adjusted_close"].to_numpy(dtype=np.float64)
    valid_adjusted_close = np.isfinite(adjusted_close) & (adjusted_close > 0.0)
    window_size = returns_required + 1
    windows = (
        sliding_window_view(adjusted_close, window_size)
        if len(adjusted_close) >= window_size
        else np.empty((0, window_size), dtype=np.float64)
    )
    raw_annualized_vol = np.full(len(panel), np.nan, dtype=np.float64)
    display_vol_pct = np.full(len(panel), np.nan, dtype=np.float64)
    window_start_sessions: list[pd.Timestamp | None] = [None] * len(panel)
    window_end_sessions: list[pd.Timestamp | None] = [None] * len(panel)
    window_missing_row_count = np.zeros(len(panel), dtype=np.int64)
    window_missing_adjclose_count = np.zeros(len(panel), dtype=np.int64)
    statuses: list[str] = []

    for position in range(len(panel)):
        if position < returns_required:
            statuses.append("insufficient_history")
            continue

        start_position = position - returns_required
        window_start_sessions[position] = sessions[start_position]
        window_end_sessions[position] = sessions[position]
        window_slice = slice(start_position, position + 1)
        missing_rows = int(np.count_nonzero(~row_present[window_slice]))
        missing_adjusted = int(
            np.count_nonzero(
                row_present[window_slice] & ~valid_adjusted_close[window_slice]
            )
        )
        window_missing_row_count[position] = missing_rows
        window_missing_adjclose_count[position] = missing_adjusted
        if missing_rows:
            statuses.append("missing_row_in_window")
        elif missing_adjusted:
            statuses.append("missing_adjusted_close_in_window")
        else:
            window_offset = position - returns_required
            price_window = windows[window_offset]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                log_returns = np.log(price_window[1:] / price_window[:-1])
            sample_std = float(np.std(log_returns, ddof=selected_ddof))
            raw_value = math.sqrt(annualization) * sample_std
            raw_annualized_vol[position] = raw_value
            display_vol_pct[position] = 100.0 * raw_value
            statuses.append("eligible")

    first_use_sessions = list(sessions[1:]) + [next_session(sessions[-1])]
    normalized = normalize_midrank_percentiles(
        raw_annualized_vol,
        sessions,
        [status == "eligible" for status in statuses],
        reference_window_sessions=reference_window,
        minimum_reference_observations=reference_minimum,
    )
    result = pd.DataFrame(
        {
            "ticker": pd.Series([ticker] * len(panel), dtype="string"),
            "signal_date": sessions,
            "first_use_session": pd.to_datetime(first_use_sessions),
            "window_start_session": pd.Series(
                window_start_sessions, dtype="datetime64[ns]"
            ),
            "window_end_session": pd.Series(
                window_end_sessions, dtype="datetime64[ns]"
            ),
            "raw_annualized_vol": raw_annualized_vol,
            "display_vol_pct": display_vol_pct,
            "window_missing_row_count": window_missing_row_count,
            "window_missing_adjclose_count": window_missing_adjclose_count,
            "status": pd.Series(statuses, dtype="string"),
            "percentile": normalized["percentile"].to_numpy(dtype=np.float64),
            "reference_count": normalized["reference_count"].to_numpy(dtype=np.int64),
            "reference_first_session": normalized["reference_first_session"].to_numpy(),
            "reference_last_session": normalized["reference_last_session"].to_numpy(),
            "normalization_status": pd.Series(
                normalized["normalization_status"], dtype="string"
            ),
        }
    )
    for column in (
        "signal_date",
        "first_use_session",
        "window_start_session",
        "window_end_session",
        "reference_first_session",
        "reference_last_session",
    ):
        result[column] = pd.to_datetime(result[column])
    return result.sort_values(["ticker", "signal_date"], kind="mergesort").reset_index(
        drop=True
    )
