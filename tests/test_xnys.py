"""Hand-checked tests for pinned XNYS sessions and scheduled closes."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from etf_monitor.xnys import (
    assert_calendar_pin,
    is_session,
    next_session,
    resolve_target_session,
    session_offset,
    xnys_sessions,
)


def test_calendar_pin_matches_approved_package_version() -> None:
    assert_calendar_pin()


def test_weekends_holidays_and_early_close_resolution() -> None:
    assert not is_session(date(2024, 7, 4))
    assert is_session(date(2024, 7, 3))
    assert resolve_target_session(
        datetime.fromisoformat("2024-07-03T16:59:59+00:00")
    ) == (pd.Timestamp("2024-07-02"))
    assert resolve_target_session(
        datetime.fromisoformat("2024-07-03T17:00:00+00:00")
    ) == (pd.Timestamp("2024-07-03"))


def test_black_friday_early_close_is_scheduled_at_1800_utc() -> None:
    before_close = datetime.fromisoformat("2024-11-29T17:59:59+00:00")
    at_close = datetime.fromisoformat("2024-11-29T18:00:00+00:00")

    assert resolve_target_session(before_close) == pd.Timestamp("2024-11-27")
    assert resolve_target_session(at_close) == pd.Timestamp("2024-11-29")


@pytest.mark.parametrize(
    ("instant", "expected"),
    [
        ("2024-03-11T19:59:59+00:00", "2024-03-08"),
        ("2024-03-11T20:00:00+00:00", "2024-03-11"),
        ("2024-03-08T20:59:59+00:00", "2024-03-07"),
        ("2024-03-08T21:00:00+00:00", "2024-03-08"),
    ],
)
def test_scheduled_close_resolution_observes_dst(instant: str, expected: str) -> None:
    assert resolve_target_session(datetime.fromisoformat(instant)) == pd.Timestamp(
        expected
    )


def test_saturday_instant_resolves_to_previous_friday() -> None:
    instant = datetime.fromisoformat("2024-07-06T16:00:00+00:00")

    assert resolve_target_session(instant) == pd.Timestamp("2024-07-05")


def test_utc_date_shift_does_not_change_exchange_local_session_date() -> None:
    instant = datetime.fromisoformat("2024-07-09T02:00:00+00:00")

    assert resolve_target_session(instant) == pd.Timestamp("2024-07-08")


def test_naive_instant_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_target_session(datetime(2024, 7, 3, 17))


def test_session_offsets_are_exact_xnys_positions() -> None:
    sessions = xnys_sessions(date(2024, 1, 2), date(2025, 12, 31))

    assert next_session(pd.Timestamp("2024-01-02")) == sessions[1]
    assert session_offset(sessions[0], 252) == sessions[252]
    assert len(sessions[:253]) == 253
    with pytest.raises(ValueError, match="not an XNYS session"):
        session_offset(date(2024, 7, 4), 1)


def test_xnys_session_range_regression_constants() -> None:
    sessions = xnys_sessions(date(2018, 1, 1), date(2025, 12, 31))

    assert sessions[0] == pd.Timestamp("2018-01-02")
    assert sessions[-1] == pd.Timestamp("2025-12-31")
    assert len(sessions) == 2011


def test_reversed_session_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="on or after"):
        xnys_sessions(date(2024, 1, 3), date(2024, 1, 2))
