"""Standalone 252-session adjusted-close Momentum diagnostic."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from etf_monitor.methodology import load_methodology
from etf_monitor.signals.panel import build_session_panel
from etf_monitor.signals.percentile import normalize_midrank_percentiles
from etf_monitor.xnys import next_session


def _positive_integer(value: int | None, default: int, name: str) -> int:
    selected = default if value is None else value
    if type(selected) is not int or selected <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return selected


def calculate_momentum(
    prices: pd.DataFrame,
    ticker: str,
    target: date | pd.Timestamp,
    *,
    grid_start: date | pd.Timestamp | None = None,
    horizon_sessions: int | None = None,
    reference_window_sessions: int | None = None,
    minimum_reference_observations: int | None = None,
) -> pd.DataFrame:
    """Calculate endpoint-only adjusted-close Momentum for one ETF.

    The raw value is ``float(np.log(A_t / A_(t-H)))``. Only the exact endpoints
    determine eligibility; missing interior rows and adjusted closes are
    diagnostic counts. Optional window overrides are intended for small unit
    tests. Evaluation code must use the methodology defaults.

    Args:
        prices: Canonical price frame restricted to the panel's target range.
        ticker: Configured ETF to calculate.
        target: Inclusive target XNYS session.
        grid_start: Optional test-only session-grid start; see
            :func:`build_session_panel`.
        horizon_sessions: Optional test override for the endpoint distance.
        reference_window_sessions: Optional test override for percentile
            history length.
        minimum_reference_observations: Optional test override for the
            percentile reference minimum.

    Returns:
        One row per panel session with raw, normalized, endpoint, timing, and
        interior-missingness diagnostics.
    """
    methodology = load_methodology()
    horizon = _positive_integer(
        horizon_sessions,
        methodology.momentum_horizon_sessions,
        "horizon_sessions",
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
    raw_log_return = np.full(len(panel), np.nan, dtype=np.float64)
    simple_return_pct = np.full(len(panel), np.nan, dtype=np.float64)
    start_sessions: list[pd.Timestamp | None] = [None] * len(panel)
    start_adjusted_close = np.full(len(panel), np.nan, dtype=np.float64)
    interior_missing_row_count = np.zeros(len(panel), dtype=np.int64)
    interior_missing_adjclose_count = np.zeros(len(panel), dtype=np.int64)
    statuses: list[str] = []

    for position in range(len(panel)):
        if position < horizon:
            statuses.append("insufficient_history")
            continue

        start_position = position - horizon
        start_sessions[position] = sessions[start_position]
        start_adjusted_close[position] = adjusted_close[start_position]
        interior = slice(start_position + 1, position)
        interior_missing_row_count[position] = int(
            np.count_nonzero(~row_present[interior])
        )
        interior_missing_adjclose_count[position] = int(
            np.count_nonzero(row_present[interior] & ~valid_adjusted_close[interior])
        )

        if not row_present[position]:
            statuses.append("missing_end_row")
        elif not valid_adjusted_close[position]:
            statuses.append("missing_end_adjusted_close")
        elif not row_present[start_position]:
            statuses.append("missing_start_row")
        elif not valid_adjusted_close[start_position]:
            statuses.append("missing_start_adjusted_close")
        else:
            ratio = adjusted_close[position] / adjusted_close[start_position]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                raw_log_return[position] = float(np.log(ratio))
                simple_return_pct[position] = 100.0 * (ratio - 1.0)
            statuses.append("eligible")

    first_use_sessions = list(sessions[1:]) + [next_session(sessions[-1])]
    normalized = normalize_midrank_percentiles(
        raw_log_return,
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
            "start_session": pd.Series(start_sessions, dtype="datetime64[ns]"),
            "start_adjusted_close": start_adjusted_close,
            "end_adjusted_close": adjusted_close,
            "raw_log_return": raw_log_return,
            "simple_return_pct": simple_return_pct,
            "interior_missing_row_count": interior_missing_row_count,
            "interior_missing_adjclose_count": interior_missing_adjclose_count,
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
    result["signal_date"] = pd.to_datetime(result["signal_date"])
    result["first_use_session"] = pd.to_datetime(result["first_use_session"])
    result["start_session"] = pd.to_datetime(result["start_session"])
    result["reference_first_session"] = pd.to_datetime(
        result["reference_first_session"]
    )
    result["reference_last_session"] = pd.to_datetime(result["reference_last_session"])
    return result.sort_values(["ticker", "signal_date"], kind="mergesort").reset_index(
        drop=True
    )
