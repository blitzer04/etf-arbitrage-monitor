"""Per-ETF trailing midrank percentile normalization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from etf_monitor.methodology import load_methodology


def normalize_midrank_percentiles(
    raw_values: Sequence[float] | np.ndarray | pd.Series,
    sessions: pd.DatetimeIndex,
    eligible: Sequence[bool] | np.ndarray | pd.Series,
    *,
    reference_window_sessions: int | None = None,
    minimum_reference_observations: int | None = None,
) -> pd.DataFrame:
    """Normalize one ETF's eligible raw observations against prior sessions.

    The reference window is inclusive of the current signal session, while
    the current raw observation is excluded. Ties use exact float equality.

    Args:
        raw_values: Raw signal values in session order; ineligible entries may
            be NaN.
        sessions: Unique, increasing, timezone-naive XNYS session labels.
        eligible: Boolean raw-eligibility mask in the same position order.
        reference_window_sessions: Optional test override; defaults to
            methodology configuration.
        minimum_reference_observations: Optional test override; defaults to
            methodology configuration.

    Returns:
        Columns ``percentile``, ``reference_count``,
        ``reference_first_session``, ``reference_last_session``, and
        ``normalization_status`` in input order.

    Raises:
        ValueError: If inputs are misaligned, unsorted, or use invalid window
            parameters.
    """
    methodology = load_methodology()
    window = (
        methodology.reference_window_sessions
        if reference_window_sessions is None
        else reference_window_sessions
    )
    minimum = (
        methodology.minimum_reference_observations
        if minimum_reference_observations is None
        else minimum_reference_observations
    )
    if type(window) is not int or window <= 0:
        raise ValueError("reference_window_sessions must be a positive integer.")
    if type(minimum) is not int or minimum <= 0:
        raise ValueError("minimum_reference_observations must be a positive integer.")

    values = np.asarray(raw_values, dtype=np.float64)
    session_index = sessions
    eligibility = np.asarray(eligible, dtype=np.bool_)
    if not (len(values) == len(session_index) == len(eligibility)):
        raise ValueError("raw_values, sessions, and eligible must have equal lengths.")
    if session_index.tz is not None or not session_index.is_monotonic_increasing:
        raise ValueError("sessions must be increasing and timezone-naive.")
    if session_index.has_duplicates:
        raise ValueError("sessions must be unique.")
    if np.any(eligibility & ~np.isfinite(values)):
        raise ValueError("Eligible raw values must be finite.")

    percentiles = np.full(len(values), np.nan, dtype=np.float64)
    reference_counts = np.zeros(len(values), dtype=np.int64)
    first_sessions: list[pd.Timestamp | None] = [None] * len(values)
    last_sessions: list[pd.Timestamp | None] = [None] * len(values)
    statuses: list[str] = []

    for position, current_value in enumerate(values):
        first_reference_position = max(0, position - window + 1)
        reference_positions = [
            reference_position
            for reference_position in range(first_reference_position, position)
            if eligibility[reference_position]
        ]
        reference_values = values[reference_positions]
        count = len(reference_positions)
        reference_counts[position] = count
        if reference_positions:
            first_sessions[position] = session_index[reference_positions[0]]
            last_sessions[position] = session_index[reference_positions[-1]]

        if not eligibility[position]:
            statuses.append("raw_ineligible")
        elif count < minimum:
            statuses.append("insufficient_reference")
        else:
            less_count = int(np.count_nonzero(reference_values < current_value))
            equal_count = int(np.count_nonzero(reference_values == current_value))
            percentiles[position] = 100.0 * (less_count + 0.5 * equal_count) / count
            statuses.append("available")

    return pd.DataFrame(
        {
            "percentile": percentiles,
            "reference_count": reference_counts,
            "reference_first_session": pd.Series(
                first_sessions, dtype="datetime64[ns]"
            ),
            "reference_last_session": pd.Series(last_sessions, dtype="datetime64[ns]"),
            "normalization_status": pd.Series(statuses, dtype="string"),
        }
    )
