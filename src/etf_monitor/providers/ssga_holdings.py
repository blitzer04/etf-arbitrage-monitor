"""Offline parser for SSGA daily fund-holdings workbooks.

The layout was observed on 2026-11-03 for 25 SPDR funds. It is not a
documented contract, so every structural assumption is validated and a
violation rejects the whole ticker response instead of being repaired.

SSGA serves only the latest snapshot (``Holdings: As of DD-Mon-YYYY``); there
is no dated archive, so holdings history exists only from the first capture.

Two layouts exist. Equity funds list ``Shares Held`` per line; cash, money
market and currency lines carry ``-`` as ticker and their amount (which can
be negative) in ``Shares Held``. Fixed-income funds list ``Par Value`` and the
issuer's ``Market Value`` instead, and some add an ``ISIN`` column.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

from etf_monitor.providers.ssga_nav import (
    MISSING_VALUE_TOKEN,
    NavPayloadError,
    _finite_number,
    _parse_date,
    _read_rows,
)

SSGA_HOLDINGS_SOURCE = "ssga_holdings_daily_xlsx"
EQUITY_HEADER = (
    "Name",
    "Ticker",
    "Identifier",
    "SEDOL",
    "Weight",
    "Sector",
    "Shares Held",
    "Local Currency",
)
FIXED_INCOME_HEADER = (
    "Name",
    "Identifier",
    "SEDOL",
    "Weight",
    "Coupon",
    "Par Value",
    "Market Value",
    "Local Currency",
    "Maturity",
)
FIXED_INCOME_ISIN_HEADER = (*FIXED_INCOME_HEADER, "ISIN")
LAYOUTS = {
    EQUITY_HEADER: "equity",
    FIXED_INCOME_HEADER: "fixed_income",
    FIXED_INCOME_ISIN_HEADER: "fixed_income",
}
_AS_OF_PREFIX = "As of "


class HoldingsPayloadError(NavPayloadError):
    """Indicate that a holdings workbook violates its observed layout."""


@dataclass(frozen=True, slots=True)
class HoldingRow:
    """One published holdings line, with ``-`` cells kept as ``None``."""

    name: str
    security_ticker: str | None
    identifier: str
    sedol: str | None
    isin: str | None
    weight_pct: float
    sector: str | None
    quantity: float | None
    par_value: float | None
    market_value: float | None
    coupon: float | None
    maturity: date | None
    local_currency: str | None


@dataclass(frozen=True, slots=True)
class ParsedHoldings:
    """Validated holdings lines for one fund in published order."""

    ticker: str
    fund_name: str
    as_of: date
    layout: str
    holdings: tuple[HoldingRow, ...]


def _text(value: object, column: str, line: int, *, optional: bool) -> str | None:
    if optional and value == MISSING_VALUE_TOKEN:
        return None
    if not isinstance(value, str) or not value.strip() or value == MISSING_VALUE_TOKEN:
        raise HoldingsPayloadError(
            f"Line {line} column {column!r} must be non-empty text"
            + (f" or {MISSING_VALUE_TOKEN!r}." if optional else "."),
            "invalid_text",
        )
    return value


def _number(value: object, column: str, line: int, *, optional: bool) -> float | None:
    if optional and value == MISSING_VALUE_TOKEN:
        return None
    number = _finite_number(value)
    if number is None:
        raise HoldingsPayloadError(
            f"Line {line} column {column!r} must be a finite number.",
            "invalid_number",
        )
    return number


def _currency(value: object, line: int) -> str | None:
    text = _text(value, "Local Currency", line, optional=True)
    if text is not None and (
        len(text) != 3 or not text.isalpha() or not text.isupper()
    ):
        raise HoldingsPayloadError(
            f"Line {line} currency {text!r} is not a three-letter code.",
            "invalid_currency",
        )
    return text


def _maturity(value: object, line: int) -> date | None:
    text = _text(value, "Maturity", line, optional=True)
    if text is None:
        return None
    parts = text.split("/")
    if len(parts) != 3 or [len(part) for part in parts] != [2, 2, 4]:
        raise HoldingsPayloadError(
            f"Line {line} maturity {text!r} is not MM/DD/YYYY.", "invalid_maturity"
        )
    try:
        return date(int(parts[2]), int(parts[0]), int(parts[1]))
    except ValueError as error:
        raise HoldingsPayloadError(
            f"Line {line} maturity {text!r} is not a calendar date.",
            "invalid_maturity",
        ) from error


def _row(cells: dict[str, object], layout: str, line: int) -> HoldingRow:
    name = _text(cells["Name"], "Name", line, optional=False)
    assert name is not None
    identifier = _text(cells["Identifier"], "Identifier", line, optional=False)
    assert identifier is not None
    weight = _number(cells["Weight"], "Weight", line, optional=False)
    assert weight is not None
    sedol = _text(cells["SEDOL"], "SEDOL", line, optional=True)
    currency = _currency(cells["Local Currency"], line)
    if layout == "equity":
        return HoldingRow(
            name=name,
            security_ticker=_text(cells["Ticker"], "Ticker", line, optional=True),
            identifier=identifier,
            sedol=sedol,
            isin=None,
            weight_pct=weight,
            sector=_text(cells["Sector"], "Sector", line, optional=True),
            quantity=_number(cells["Shares Held"], "Shares Held", line, optional=False),
            par_value=None,
            market_value=None,
            coupon=None,
            maturity=None,
            local_currency=currency,
        )
    return HoldingRow(
        name=name,
        security_ticker=None,
        identifier=identifier,
        sedol=sedol,
        isin=_text(cells["ISIN"], "ISIN", line, optional=True)
        if "ISIN" in cells
        else None,
        weight_pct=weight,
        sector=None,
        quantity=None,
        par_value=_number(cells["Par Value"], "Par Value", line, optional=False),
        market_value=_number(
            cells["Market Value"], "Market Value", line, optional=False
        ),
        coupon=_number(cells["Coupon"], "Coupon", line, optional=True),
        maturity=_maturity(cells["Maturity"], line),
        local_currency=currency,
    )


def parse_ssga_holdings(raw: bytes, ticker: str) -> ParsedHoldings:
    """Parse and validate one SSGA daily holdings workbook without network access.

    Args:
        raw: Exact workbook bytes returned by the issuer.
        ticker: Requested ETF ticker; the workbook must identify the same fund.

    Returns:
        The fund identity, the ``As of`` date, the layout name (``equity`` or
        ``fixed_income``) and every holdings line in published order.

    Raises:
        HoldingsPayloadError: If the workbook, identity, date, header, or any
            line violates the observed layout.
    """
    try:
        rows = _read_rows(raw)
    except NavPayloadError as error:
        raise HoldingsPayloadError(str(error), error.reason_code) from error
    if len(rows) < 5 or len(rows[0]) != 2 or rows[0][0] != "Fund Name:":
        raise HoldingsPayloadError(
            "First row must be 'Fund Name:' and a name.", "identity"
        )
    fund_name = rows[0][1]
    if not isinstance(fund_name, str) or not fund_name.strip():
        raise HoldingsPayloadError("Fund name must be a non-empty string.", "identity")
    if rows[1] != ("Ticker Symbol:", ticker):
        raise HoldingsPayloadError(
            f"Workbook ticker row {rows[1]!r} does not identify {ticker}.",
            "identity_mismatch",
        )
    as_of_cell = rows[2][1] if len(rows[2]) == 2 and rows[2][0] == "Holdings:" else None
    as_of = (
        _parse_date(as_of_cell[len(_AS_OF_PREFIX) :])
        if isinstance(as_of_cell, str) and as_of_cell.startswith(_AS_OF_PREFIX)
        else None
    )
    if as_of is None:
        raise HoldingsPayloadError(
            f"Third row {rows[2]!r} must be 'Holdings:' and 'As of DD-Mon-YYYY'.",
            "as_of",
        )
    if rows[3]:
        raise HoldingsPayloadError("Fourth row must be empty.", "workbook_shape")
    header = cast(tuple[str, ...], rows[4])
    if header not in LAYOUTS:
        raise HoldingsPayloadError(
            f"Unrecognized holdings header {header!r}.", "header_mismatch"
        )
    layout = LAYOUTS[header]

    holdings: list[HoldingRow] = []
    footer_started = False
    for line, row in enumerate(rows[5:], start=6):
        if not row:
            continue
        if isinstance(row[0], str) and all(
            cell is None for cell in row[1 : len(header)]
        ):
            footer_started = True
            continue
        if footer_started:
            raise HoldingsPayloadError(
                f"Line {line} has data after footer text.", "malformed_row"
            )
        if len(row) > len(header):
            raise HoldingsPayloadError(
                f"Line {line} has more cells than the header.", "malformed_row"
            )
        padded = row + (None,) * (len(header) - len(row))
        holdings.append(_row(dict(zip(header, padded, strict=True)), layout, line))
    if not holdings:
        raise HoldingsPayloadError("Workbook contains no holdings lines.", "empty")
    return ParsedHoldings(
        ticker=ticker,
        fund_name=fund_name,
        as_of=as_of,
        layout=layout,
        holdings=tuple(holdings),
    )
