"""Offline SSGA NAV-history parser tests on fully fictional workbooks."""

from __future__ import annotations

from datetime import date

import pytest

from etf_monitor.providers.ssga_nav import NavPayloadError, parse_ssga_nav_history
from tests.fixtures.ssga_nav import build_ssga_workbook

START = date(2024, 1, 1)


def _reason(raw: bytes, ticker: str = "SPY") -> str:
    with pytest.raises(NavPayloadError) as caught:
        parse_ssga_nav_history(raw, ticker, START)
    return caught.value.reason_code


def test_parses_rows_ascending_and_ignores_footer_with_link_cells() -> None:
    parsed = parse_ssga_nav_history(build_ssga_workbook(), "SPY", START)

    assert parsed.fund_name == "FICTIONAL Test Fund"
    assert [row.date for row in parsed.observations] == [
        date(2024, 7, 2),
        date(2024, 7, 3),
    ]
    assert parsed.observations[1].nav == 101.5
    assert parsed.observations[1].shares_outstanding == 2000
    assert isinstance(parsed.observations[1].shares_outstanding, int)
    assert parsed.observations[1].total_net_assets == 203000.0
    assert parsed.rows_before_history_start == 0


def test_dash_values_stay_missing_not_zero() -> None:
    raw = build_ssga_workbook(rows=(("02-Jul-2024", 100.0, "-", "-"),))

    observation = parse_ssga_nav_history(raw, "SPY", START).observations[0]

    assert observation.shares_outstanding is None
    assert observation.total_net_assets is None


def test_rows_before_history_start_are_counted_not_validated() -> None:
    raw = build_ssga_workbook(
        rows=(
            ("02-Jan-2024", 100.0, 1000.0, 100000.0),
            ("29-Dec-2023", 99.0, 1000.5, 99049.5),
        )
    )

    parsed = parse_ssga_nav_history(raw, "SPY", START)

    assert [row.date for row in parsed.observations] == [date(2024, 1, 2)]
    assert parsed.rows_before_history_start == 1


def test_fractional_shares_in_window_are_rejected() -> None:
    raw = build_ssga_workbook(rows=(("02-Jan-2024", 100.0, 1000.5, 100050.0),))
    assert _reason(raw) == "invalid_shares_outstanding"


def test_shares_at_two_pow_53_are_exact_and_above_are_rejected() -> None:
    exact = build_ssga_workbook(rows=(("02-Jan-2024", 100.0, float(2**53), 1.0),))
    parsed = parse_ssga_nav_history(exact, "SPY", START)
    assert parsed.observations[0].shares_outstanding == 2**53

    above = build_ssga_workbook(rows=(("02-Jan-2024", 100.0, float(2**54), 1.0),))
    assert _reason(above) == "invalid_shares_outstanding"


def test_identity_mismatch_is_rejected_before_rows() -> None:
    assert _reason(build_ssga_workbook(workbook_ticker="DIA")) == "identity_mismatch"


def test_blank_fund_name_is_rejected() -> None:
    assert _reason(build_ssga_workbook(fund_name=" ")) == "identity"


def test_header_change_is_rejected() -> None:
    raw = build_ssga_workbook(header=("Date", "NAV", "Shares", "Total Net Assets"))
    assert _reason(raw) == "header_mismatch"


@pytest.mark.parametrize("nav", [0.0, -1.0, "-", "100.0", True, None])
def test_invalid_nav_is_rejected(nav: object) -> None:
    raw = build_ssga_workbook(rows=(("02-Jul-2024", nav, 1.0, 1.0),))
    assert _reason(raw) in {"invalid_nav", "malformed_row"}


def test_negative_net_assets_are_rejected() -> None:
    raw = build_ssga_workbook(rows=(("02-Jul-2024", 1.0, 1.0, -5.0),))
    assert _reason(raw) == "invalid_total_net_assets"


def test_duplicate_dates_are_rejected() -> None:
    raw = build_ssga_workbook(
        rows=(("02-Jul-2024", 1.0, 1.0, 1.0), ("02-Jul-2024", 1.0, 1.0, 1.0))
    )
    assert _reason(raw) == "duplicate_date"


def test_ascending_publication_order_is_rejected() -> None:
    raw = build_ssga_workbook(
        rows=(("02-Jul-2024", 1.0, 1.0, 1.0), ("03-Jul-2024", 1.0, 1.0, 1.0))
    )
    assert _reason(raw) == "unordered_dates"


@pytest.mark.parametrize("label", ["31-Feb-2024", "2024-07-02", "02-JUL-2024"])
def test_invalid_date_labels_are_rejected(label: str) -> None:
    raw = build_ssga_workbook(rows=((label, 1.0, 1.0, 1.0),))
    assert _reason(raw) == "malformed_row"


def test_dated_row_after_footer_is_rejected() -> None:
    raw = build_ssga_workbook(extra_rows_after_footer=(("01-Jul-2024", 1.0, 1.0, 1.0),))
    assert _reason(raw) == "malformed_row"


def test_non_workbook_bytes_and_extra_sheets_are_rejected() -> None:
    assert _reason(b"<html>not a workbook</html>") == "not_xlsx"
    assert _reason(build_ssga_workbook(extra_sheet=True)) == "workbook_shape"


def test_history_entirely_before_start_is_rejected() -> None:
    raw = build_ssga_workbook(rows=(("29-Dec-2023", 1.0, 1.0, 1.0),))
    assert _reason(raw) == "empty_history"
