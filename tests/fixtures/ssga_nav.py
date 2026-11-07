"""Fully fictional SSGA-shaped NAV-history workbooks for offline tests."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO

from openpyxl import Workbook

FOOTER = (
    "FICTIONAL footer disclaimer text.",
    "The FICTIONAL prospectus is available by clicking",
)


def build_ssga_workbook(
    ticker: str = "SPY",
    rows: Sequence[tuple[object, object, object, object]] = (
        ("03-Jul-2024", 101.5, 2000.0, 203000.0),
        ("02-Jul-2024", 100.0, 1900.0, 190000.0),
    ),
    *,
    fund_name: object = "FICTIONAL Test Fund",
    workbook_ticker: object | None = None,
    header: tuple[object, ...] = (
        "Date",
        "NAV",
        "Shares Outstanding",
        "Total Net Assets",
    ),
    footer: Sequence[str] = FOOTER,
    footer_link_text: str | None = "FICTIONAL link",
    extra_rows_after_footer: Sequence[tuple[object, ...]] = (),
    extra_sheet: bool = False,
) -> bytes:
    """Build workbook bytes in the layout SSGA was observed to publish."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "navhist"
    sheet.append(["Fund Name:", fund_name])
    sheet.append(
        ["Ticker Symbol:", ticker if workbook_ticker is None else workbook_ticker]
    )
    sheet.append([])
    sheet.append(list(header))
    for row in rows:
        sheet.append(list(row))
    sheet.append([])
    for position, text in enumerate(footer):
        values: list[object] = [text]
        if position == len(footer) - 1 and footer_link_text is not None:
            values += [None, None, None, None, None, footer_link_text]
        sheet.append(values)
    for extra in extra_rows_after_footer:
        sheet.append(list(extra))
    if extra_sheet:
        workbook.create_sheet("other")
    stream = BytesIO()
    workbook.save(stream)
    return stream.getvalue()
