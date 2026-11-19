"""Yahoo cash-dividend events and their contribution to a holdings basket."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from etf_monitor.arbitrage.basket import classify_lines
from etf_monitor.arbitrage.splits import SplitEvent, split_factors
from etf_monitor.holdings.symbols import yahoo_symbol
from etf_monitor.xnys import is_session

_NEW_YORK = ZoneInfo("America/New_York")


class DividendPayloadError(ValueError):
    """Indicate that a Yahoo chart payload's dividend events are unusable."""


@dataclass(frozen=True, slots=True)
class DividendEvent:
    """Per-share cash effective at an XNYS session open.

    Raw Yahoo amounts are split-adjusted; ``as_traded_dividends`` restores the
    amount to its ex-date share basis.
    """

    ticker: str
    ex_date: date
    amount: float


def _reject_constant(token: str) -> None:
    raise DividendPayloadError(f"non-standard JSON numeric constant {token!r}")


def _positive_amount(value: object, ticker: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DividendPayloadError(
            f"{ticker} dividend amount must be a finite positive number, got {value!r}."
        )
    try:
        amount = float(value)
    except (OverflowError, ValueError) as error:
        raise DividendPayloadError(
            f"{ticker} dividend amount must be a finite positive number, got {value!r}."
        ) from error
    if not math.isfinite(amount) or amount <= 0:
        raise DividendPayloadError(
            f"{ticker} dividend amount must be a finite positive number, got {value!r}."
        )
    if isinstance(value, int) and int(amount) != value:
        raise DividendPayloadError(
            f"{ticker} dividend amount cannot be represented exactly as a float, "
            f"got {value!r}."
        )
    return amount


def dividend_events_from_chart(raw: bytes, ticker: str) -> tuple[DividendEvent, ...]:
    """Extract validated cash dividends from a raw Yahoo chart response.

    Yahoo dividend epochs are converted to America/New_York calendar dates.
    Repeated records for a session collapse only when their amounts agree.

    Args:
        raw: Exact Yahoo chart response bytes, requested with dividend events.
        ticker: Expected ``meta.symbol`` and event symbol.

    Returns:
        Events sorted by ex-date; empty when the payload contains no dividends.

    Raises:
        DividendPayloadError: On malformed JSON, identity mismatch, invalid
            dates or amounts, non-session dates, or conflicting duplicates.
    """
    try:
        payload = json.loads(raw, parse_constant=_reject_constant)
        result = payload["chart"]["result"][0]
        symbol = result["meta"]["symbol"]
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        KeyError,
        IndexError,
        TypeError,
    ) as error:
        raise DividendPayloadError(
            f"{ticker} chart payload is malformed: {error}."
        ) from error
    if symbol != ticker:
        raise DividendPayloadError(f"Chart payload is for {symbol!r}, not {ticker!r}.")

    chart_events = result.get("events") or {}
    if not isinstance(chart_events, dict):
        raise DividendPayloadError(f"{ticker} chart events must be an object.")
    dividends = chart_events.get("dividends") or {}
    if not isinstance(dividends, dict):
        raise DividendPayloadError(f"{ticker} dividend events must be an object.")

    by_date: dict[date, float] = {}
    for entry in dividends.values():
        if (
            not isinstance(entry, dict)
            or isinstance(entry.get("date"), bool)
            or not isinstance(entry.get("date"), int)
        ):
            raise DividendPayloadError(f"{ticker} dividend event has no integer date.")
        try:
            ex_date = (
                datetime.fromtimestamp(entry["date"], tz=UTC)
                .astimezone(_NEW_YORK)
                .date()
            )
        except (OverflowError, OSError, ValueError) as error:
            raise DividendPayloadError(
                f"{ticker} dividend event has an invalid epoch date."
            ) from error
        if not is_session(ex_date):
            raise DividendPayloadError(
                f"{ticker} dividend on non-session date {ex_date}."
            )
        amount = _positive_amount(entry.get("amount"), ticker)
        previous = by_date.get(ex_date)
        if previous is not None and previous != amount:
            raise DividendPayloadError(
                f"{ticker} has conflicting dividends on {ex_date}."
            )
        by_date[ex_date] = amount

    return tuple(
        DividendEvent(ticker, ex_date, by_date[ex_date]) for ex_date in sorted(by_date)
    )


def as_traded_dividends(
    dividends: Sequence[DividendEvent],
    splits: Sequence[SplitEvent],
) -> tuple[DividendEvent, ...]:
    """Convert Yahoo dividend amounts to shares as traded on each ex-date.

    Yahoo adjusts historical dividend amounts for splits after the ex-date.
    Applying the later-split factor from the same component vintage restores
    each event's per-share amount to its ex-date basis.

    Args:
        dividends: Validated dividends from one component vintage.
        splits: Split events from that same component vintage.

    Returns:
        Events with amounts expressed in as-traded shares, preserving order.
    """
    if not dividends:
        return ()
    factors = split_factors(
        pd.Series([event.ticker for event in dividends], dtype="string"),
        pd.Series(pd.to_datetime([event.ex_date for event in dividends])),
        splits,
    )
    result: list[DividendEvent] = []
    for event, factor in zip(dividends, factors, strict=True):
        amount = event.amount * float(factor)
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError(
                f"{event.ticker} as-traded dividend amount is not finite and positive."
            )
        result.append(DividendEvent(event.ticker, event.ex_date, amount))
    return tuple(result)


def ex_dividend_value(
    holdings: pd.DataFrame,
    dividends: Sequence[DividendEvent],
    after: pd.Timestamp,
    through: pd.Timestamp,
) -> pd.Series:
    """Value ex-dividend receivables for each fund's priced holdings lines.

    ``dividends`` must have amounts in as-traded units. Only events with
    ``after < ex_date <= through`` are included.

    Args:
        holdings: Canonical holdings rows for one or more funds.
        dividends: Validated, as-traded dividend events keyed to Yahoo symbols.
        after: Exclusive lower valuation-date bound.
        through: Inclusive upper valuation-date bound.

    Returns:
        A series indexed by fund ticker. Funds with priced lines and no
        eligible events receive zero; funds with no priced lines are omitted.
    """
    lines = holdings.loc[classify_lines(holdings).eq("priced").to_numpy()].copy()
    lines["symbol"] = lines["security_ticker"].map(
        lambda value: yahoo_symbol(str(value))
    )
    lines = lines.loc[lines["symbol"].notna()]
    result: dict[str, float] = {}
    eligible = [
        event
        for event in dividends
        if pd.Timestamp(after).date() < event.ex_date <= pd.Timestamp(through).date()
    ]
    events_by_symbol: dict[str, list[DividendEvent]] = {}
    for event in eligible:
        events_by_symbol.setdefault(event.ticker, []).append(event)

    for ticker, fund in lines.groupby("ticker", sort=True):
        total = 0.0
        for symbol, symbol_lines in fund.groupby("symbol", sort=True):
            events = events_by_symbol.get(str(symbol), [])
            if not events:
                continue
            amount = np.array([event.amount for event in events], dtype=np.float64)
            quantity = float(symbol_lines["quantity"].sum())
            total += float((quantity * amount).sum())
        result[str(ticker)] = total
    return pd.Series(result, name="ex_dividend_value_usd", dtype="float64")
