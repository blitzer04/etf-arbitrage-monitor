"""Pinned XNYS session labels and scheduled-close target resolution."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import exchange_calendars
import pandas as pd
from exchange_calendars import ExchangeCalendar, get_calendar

from etf_monitor.methodology import load_methodology

_CALENDAR_NAME = "XNYS"
_CALENDAR = get_calendar(_CALENDAR_NAME)


def assert_calendar_pin() -> None:
    """Require the installed exchange-calendars version approved by methodology.

    Raises:
        RuntimeError: If the installed package differs from the locked version.
    """
    expected_version = load_methodology().exchange_calendars_version
    actual_version = exchange_calendars.__version__
    if actual_version != expected_version:
        raise RuntimeError(
            f"exchange-calendars version {actual_version} does not match "
            f"approved pin {expected_version}."
        )


def _calendar() -> ExchangeCalendar:
    """Return the process-wide pinned XNYS calendar."""
    assert_calendar_pin()
    return _CALENDAR


def _session_label(value: date | datetime | pd.Timestamp) -> pd.Timestamp:
    """Normalize a session-like value to a timezone-naive session label."""
    label = pd.Timestamp(value)
    if label.tzinfo is not None:
        label = label.tz_localize(None)
    return label.normalize()


def _naive_session_labels(labels: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return XNYS labels as timezone-naive midnight timestamps."""
    if labels.tz is not None:
        return labels.tz_localize(None)
    return labels


def xnys_sessions(start: date, end_inclusive: date) -> pd.DatetimeIndex:
    """Return XNYS session labels within an inclusive date range.

    Args:
        start: First calendar date to include.
        end_inclusive: Last calendar date to include.

    Returns:
        Timezone-naive midnight labels for regular XNYS sessions.

    Raises:
        ValueError: If the requested date range is reversed.
    """
    if end_inclusive < start:
        raise ValueError("end_inclusive must be on or after start.")
    sessions = _calendar().sessions_in_range(
        pd.Timestamp(start),
        pd.Timestamp(end_inclusive),
    )
    return _naive_session_labels(pd.DatetimeIndex(sessions))


def is_session(session_date: date) -> bool:
    """Return whether a calendar date is an XNYS regular session.

    Args:
        session_date: Calendar date to test.

    Returns:
        True only for a scheduled XNYS session.
    """
    return bool(_calendar().is_session(_session_label(session_date)))


def session_offset(session: date | datetime | pd.Timestamp, n: int) -> pd.Timestamp:
    """Move by exactly ``n`` XNYS session positions.

    Args:
        session: XNYS session label used as the starting position.
        n: Signed number of sessions to move.

    Returns:
        Timezone-naive XNYS session label at the offset position.

    Raises:
        ValueError: If the starting date is not a session.
        IndexError: If the requested position exceeds calendar bounds.
    """
    calendar = _calendar()
    label = _session_label(session)
    if not calendar.is_session(label):
        raise ValueError(f"{label.date()} is not an XNYS session.")
    try:
        result = calendar.session_offset(label, n)
    except (IndexError, ValueError) as error:
        raise IndexError(
            f"Session offset {n} from {label.date()} exceeds calendar bounds."
        ) from error
    return pd.Timestamp(result).tz_localize(None)


def next_session(session: date | datetime | pd.Timestamp) -> pd.Timestamp:
    """Return the next XNYS session after a session label.

    Args:
        session: XNYS session label.

    Returns:
        Timezone-naive label of the next session.
    """
    return session_offset(session, 1)


def resolve_target_session(instant: datetime) -> pd.Timestamp:
    """Resolve the latest session whose scheduled close is no later than instant.

    Args:
        instant: Timezone-aware observation instant.

    Returns:
        Timezone-naive label of the latest eligible XNYS session.

    Raises:
        ValueError: If ``instant`` is naive or predates available calendar data.
    """
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("instant must be timezone-aware.")
    utc_instant = pd.Timestamp(instant).tz_convert("UTC")
    calendar = _calendar()
    end_date = utc_instant.date()
    start_date = end_date - timedelta(days=10)
    labels = calendar.sessions_in_range(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date),
    )
    eligible = [
        label for label in labels if calendar.session_close(label) <= utc_instant
    ]
    if not eligible:
        raise ValueError("No XNYS session close is available at or before instant.")
    return pd.Timestamp(eligible[-1]).tz_localize(None)
