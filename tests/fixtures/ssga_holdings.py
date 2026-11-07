"""Fully fictional SSGA-shaped daily holdings workbooks for offline tests."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO

from openpyxl import Workbook

from etf_monitor.providers.ssga_holdings import EQUITY_HEADER, FIXED_INCOME_HEADER

EQUITY_LINES: tuple[tuple[object, ...], ...] = (
    ("FICTIONAL ALPHA CORP", "AAA", "000000001", "0000001", 60.0, "-", 100.0, "USD"),
    ("FICTIONAL BETA CL B", "BBB.B", "000000002", "0000002", 30.0, "-", 50.0, "USD"),
    ("FICTIONAL SECTOR FUT DEC26", "IXXZ6", "ADI000001", "-", 0.01, "-", 7.0, "USD"),
    ("FICTIONAL RIGHT", "-", "000CVR001", "-", 0.5, "-", 9.0, "USD"),
    ("US DOLLAR", "-", "999USDZ92", "-", 9.49, "-", 1_000.0, "USD"),
)
FIXED_INCOME_LINES: tuple[tuple[object, ...], ...] = (
    (
        "FICTIONAL BOND 1",
        "US0000000001",
        "-",
        60.0,
        5.0,
        600.0,
        600.0,
        "USD",
        "01/15/2030",
    ),
    (
        "FICTIONAL BOND 2",
        "US0000000002",
        "B000002",
        40.5,
        0.0,
        400.0,
        405.0,
        "USD",
        "-",
    ),
    ("US DOLLAR", "999USDZ92", "-", -0.5, 0.0, -5.0, -5.0, "USD", "-"),
)


def build_holdings_workbook(
    ticker: str = "SPY",
    lines: Sequence[tuple[object, ...]] = EQUITY_LINES,
    *,
    header: tuple[str, ...] = EQUITY_HEADER,
    as_of: object = "As of 03-Jul-2024",
    workbook_ticker: object | None = None,
    blank_fourth_row: bool = True,
    extra_rows_after_footer: Sequence[tuple[object, ...]] = (),
) -> bytes:
    """Build workbook bytes in the layout SSGA was observed to publish."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "holdings"
    sheet.append(["Fund Name:", "FICTIONAL Test Fund"])
    sheet.append(
        ["Ticker Symbol:", ticker if workbook_ticker is None else workbook_ticker]
    )
    sheet.append(["Holdings:", as_of])
    sheet.append([] if blank_fourth_row else ["unexpected"])
    sheet.append(list(header))
    for line in lines:
        sheet.append(list(line))
    sheet.append([])
    sheet.append(["FICTIONAL footer disclaimer."])
    sheet.append(
        ["FICTIONAL footer link", None, None, None, None, None, None, None, None, "x"]
    )
    for extra in extra_rows_after_footer:
        sheet.append(list(extra))
    stream = BytesIO()
    workbook.save(stream)
    return stream.getvalue()


def build_fixed_income_workbook(ticker: str = "BIL") -> bytes:
    """Build a fixed-income-layout workbook."""
    return build_holdings_workbook(
        ticker, FIXED_INCOME_LINES, header=FIXED_INCOME_HEADER
    )
