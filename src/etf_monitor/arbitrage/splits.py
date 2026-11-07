"""Share-split events from Yahoo chart payloads, for issuer-NAV comparisons.

Yahoo's ``close`` is split-adjusted: closes before a ``numerator:denominator``
split are divided by ``numerator / denominator``. SSGA's NAV-history files
are not restated, so pre-split NAV and shares outstanding stay in pre-split
units. Comparing them needs the as-traded close
``close * split_factor``, where ``split_factor`` is the product of
``numerator / denominator`` over every split whose session is after t.

A split session's share-count change mixes the split with creations and
redemptions, so the flow proxy is not computed for that session.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from etf_monitor.xnys import is_session

_NEW_YORK = ZoneInfo("America/New_York")


class SplitPayloadError(ValueError):
    """Indicate that a Yahoo chart payload's split events are unusable."""


@dataclass(frozen=True, slots=True)
class SplitEvent:
    """One split effective at the open of XNYS session ``date``."""

    ticker: str
    date: date
    numerator: int
    denominator: int

    @property
    def ratio(self) -> float:
        """Return new shares per old share."""
        return self.numerator / self.denominator


def _positive_integer(value: object, field: str, ticker: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
        or value != int(value)
    ):
        raise SplitPayloadError(
            f"{ticker} split {field} must be a positive integer, got {value!r}."
        )
    return int(value)


def _reject_constant(token: str) -> None:
    raise SplitPayloadError(f"non-standard JSON numeric constant {token!r}")


def split_events_from_chart(raw: bytes, ticker: str) -> tuple[SplitEvent, ...]:
    """Extract validated split events from one raw Yahoo chart response.

    Args:
        raw: Exact Yahoo chart response bytes, requested with split events.
        ticker: Expected ``meta.symbol``.

    Returns:
        Split events sorted by session; empty when the payload has none.

    Raises:
        SplitPayloadError: On malformed JSON, identity mismatch, non-integer
            ratios, non-session dates, or duplicate split sessions.
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
        raise SplitPayloadError(
            f"{ticker} chart payload is malformed: {error}."
        ) from error
    if symbol != ticker:
        raise SplitPayloadError(f"Chart payload is for {symbol!r}, not {ticker!r}.")
    splits = (result.get("events") or {}).get("splits") or {}
    if not isinstance(splits, dict):
        raise SplitPayloadError(f"{ticker} split events must be an object.")
    events: dict[date, SplitEvent] = {}
    for entry in splits.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("date"), int):
            raise SplitPayloadError(f"{ticker} split event has no integer date.")
        session = (
            datetime.fromtimestamp(entry["date"], tz=UTC).astimezone(_NEW_YORK).date()
        )
        if not is_session(session):
            raise SplitPayloadError(f"{ticker} split on non-session date {session}.")
        if session in events:
            raise SplitPayloadError(f"{ticker} has two splits on {session}.")
        events[session] = SplitEvent(
            ticker,
            session,
            _positive_integer(entry.get("numerator"), "numerator", ticker),
            _positive_integer(entry.get("denominator"), "denominator", ticker),
        )
    return tuple(events[day] for day in sorted(events))


def load_split_events(raw_dir: Path, tickers: Iterable[str]) -> tuple[SplitEvent, ...]:
    """Read ``<TICKER>.json`` for every ticker from one raw price batch.

    Raises:
        FileNotFoundError: If a ticker's raw chart file is absent.
        SplitPayloadError: If any payload's split events are invalid.
    """
    events: list[SplitEvent] = []
    for ticker in sorted(set(tickers)):
        events.extend(
            split_events_from_chart((raw_dir / f"{ticker}.json").read_bytes(), ticker)
        )
    return tuple(events)


def split_factors(
    tickers: pd.Series, dates: pd.Series, events: Sequence[SplitEvent]
) -> np.ndarray:
    """Return, per row, the product of split ratios after that row's date."""
    factors = np.ones(len(dates), dtype=np.float64)
    ticker_values = tickers.astype("string").to_numpy()
    date_values = pd.to_datetime(dates).to_numpy()
    for event in events:
        after = (ticker_values == event.ticker) & (
            date_values < np.datetime64(event.date, "ns")
        )
        factors[after] *= event.ratio
    return factors
