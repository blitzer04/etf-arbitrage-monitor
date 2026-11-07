"""Offline parser for SSGA daily NAV-history workbooks.

The workbook layout was observed on 2026-11-03 for 25 SPDR funds; it is not a
documented contract, so every structural assumption below is validated and a
violation rejects the whole ticker response instead of being repaired.
"""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from datetime import date
from io import BytesIO

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

SSGA_NAV_SOURCE = "ssga_navhist_xlsx"
NAV_HISTORY_HEADER = ("Date", "NAV", "Shares Outstanding", "Total Net Assets")
# SSGA writes "-" where it publishes no value; it is preserved as missing.
MISSING_VALUE_TOKEN = "-"
_MAX_EXACT_FLOAT_INTEGER = 2**53
_MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


class NavPayloadError(ValueError):
    """Indicate that an issuer NAV-history payload violates its observed layout."""

    def __init__(self, message: str, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class NavObservation:
    """One published end-of-day NAV row for one fund."""

    date: date
    nav: float
    shares_outstanding: int | None
    total_net_assets: float | None


@dataclass(frozen=True, slots=True)
class ParsedNavHistory:
    """Validated NAV-history rows in ascending date order."""

    ticker: str
    fund_name: str
    observations: tuple[NavObservation, ...]
    rows_before_history_start: int


def _trimmed(row: tuple[object, ...]) -> tuple[object, ...]:
    end = len(row)
    while end and row[end - 1] is None:
        end -= 1
    return row[:end]


def _parse_date(value: object) -> date | None:
    """Parse SSGA's ``DD-Mon-YYYY`` label strictly; return None if it is not one."""
    if not isinstance(value, str) or len(value) != 11:
        return None
    day_text, month_text, year_text = value[:2], value[3:6], value[7:]
    if value[2] != "-" or value[6] != "-" or month_text not in _MONTHS:
        return None
    if not (day_text.isdigit() and year_text.isdigit()):
        return None
    try:
        return date(int(year_text), _MONTHS[month_text], int(day_text))
    except ValueError:
        return None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    if not math.isfinite(converted) or converted != value:
        return None
    return converted


def _nav_value(value: object, label: str) -> float:
    number = _finite_number(value)
    if number is None or number <= 0:
        raise NavPayloadError(
            f"NAV on {label} must be a positive finite number.", "invalid_nav"
        )
    return number


def _shares_value(value: object, label: str) -> int | None:
    if value == MISSING_VALUE_TOKEN:
        return None
    number = _finite_number(value)
    if (
        number is None
        or number < 0
        or not number.is_integer()
        or number > _MAX_EXACT_FLOAT_INTEGER
    ):
        raise NavPayloadError(
            f"Shares outstanding on {label} must be a non-negative integer "
            f"no larger than 2**53 or {MISSING_VALUE_TOKEN!r}.",
            "invalid_shares_outstanding",
        )
    return int(number)


def _net_assets_value(value: object, label: str) -> float | None:
    if value == MISSING_VALUE_TOKEN:
        return None
    number = _finite_number(value)
    if number is None or number < 0:
        raise NavPayloadError(
            f"Total net assets on {label} must be a non-negative finite number "
            f"or {MISSING_VALUE_TOKEN!r}.",
            "invalid_total_net_assets",
        )
    return number


def _read_rows(raw: bytes) -> list[tuple[object, ...]]:
    try:
        workbook = load_workbook(BytesIO(raw), read_only=True, data_only=True)
    except (zipfile.BadZipFile, InvalidFileException, KeyError, OSError) as error:
        raise NavPayloadError(
            f"Response is not a readable xlsx workbook: {error}", "not_xlsx"
        ) from error
    try:
        if len(workbook.worksheets) != 1:
            raise NavPayloadError(
                "Workbook must contain exactly one worksheet.", "workbook_shape"
            )
        return [
            _trimmed(tuple(row))
            for row in workbook.worksheets[0].iter_rows(values_only=True)
        ]
    finally:
        workbook.close()


def parse_ssga_nav_history(
    raw: bytes, ticker: str, history_start: date
) -> ParsedNavHistory:
    """Parse and validate one SSGA NAV-history workbook without network access.

    Rows dated before ``history_start`` are outside the methodology window:
    their date labels still count toward the order and duplicate checks, but
    their values are neither validated nor returned (only counted). For
    example BIL publishes fractional shares outstanding in December 2017.

    Args:
        raw: Exact workbook bytes returned by the issuer.
        ticker: Requested ETF ticker; the workbook must identify the same fund.
        history_start: First date whose values are validated and returned.

    Returns:
        Validated in-window rows sorted by ascending date. ``-`` cells become
        ``None``.

    Raises:
        NavPayloadError: If the workbook, identity, header, rows, or date order
            violate the observed layout.
    """
    rows = _read_rows(raw)
    if len(rows) < 2 or len(rows[0]) != 2 or rows[0][0] != "Fund Name:":
        raise NavPayloadError("First row must be 'Fund Name:' and a name.", "identity")
    fund_name = rows[0][1]
    if not isinstance(fund_name, str) or not fund_name.strip():
        raise NavPayloadError("Fund name must be a non-empty string.", "identity")
    if rows[1] != ("Ticker Symbol:", ticker):
        raise NavPayloadError(
            f"Workbook ticker row {rows[1]!r} does not identify {ticker}.",
            "identity_mismatch",
        )

    header_positions = [
        position for position, row in enumerate(rows) if row == NAV_HISTORY_HEADER
    ]
    if len(header_positions) != 1:
        raise NavPayloadError(
            f"Expected exactly one header row {NAV_HISTORY_HEADER!r}.",
            "header_mismatch",
        )
    if any(row for row in rows[2 : header_positions[0]]):
        raise NavPayloadError(
            "Unexpected content between identity and header rows.", "workbook_shape"
        )

    observations: list[NavObservation] = []
    published_dates: list[date] = []
    footer_started = False
    for row in rows[header_positions[0] + 1 :]:
        if not row:
            continue
        observation_date = _parse_date(row[0])
        if observation_date is None:
            # Footer disclaimer rows carry text in column A (and sometimes
            # link text beyond column D) but nothing in the data columns.
            if isinstance(row[0], str) and all(
                cell is None for cell in row[1 : len(NAV_HISTORY_HEADER)]
            ):
                footer_started = True
                continue
            raise NavPayloadError(f"Unrecognized row {row[:4]!r}.", "malformed_row")
        if footer_started:
            raise NavPayloadError(
                f"Dated row {row[0]!r} appears after footer text.", "malformed_row"
            )
        if len(row) != len(NAV_HISTORY_HEADER):
            raise NavPayloadError(
                f"Row {row[0]!r} must have exactly four cells.", "malformed_row"
            )
        published_dates.append(observation_date)
        if observation_date < history_start:
            continue
        label = str(row[0])
        observations.append(
            NavObservation(
                date=observation_date,
                nav=_nav_value(row[1], label),
                shares_outstanding=_shares_value(row[2], label),
                total_net_assets=_net_assets_value(row[3], label),
            )
        )

    if not observations:
        raise NavPayloadError(
            f"Workbook contains no NAV rows on or after {history_start}.",
            "empty_history",
        )
    if len(set(published_dates)) != len(published_dates):
        raise NavPayloadError("Workbook repeats a date.", "duplicate_date")
    if any(
        later <= earlier
        for later, earlier in zip(published_dates, published_dates[1:], strict=False)
    ):
        raise NavPayloadError(
            "Workbook dates must be strictly descending as published.",
            "unordered_dates",
        )
    return ParsedNavHistory(
        ticker=ticker,
        fund_name=fund_name,
        observations=tuple(reversed(observations)),
        rows_before_history_start=len(published_dates) - len(observations),
    )
