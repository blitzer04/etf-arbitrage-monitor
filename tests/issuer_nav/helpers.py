"""Fully fictional builders for issuer NAV tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pandas as pd

from etf_monitor.issuer_nav.schema import FICTIONAL_NAV_SOURCE
from etf_monitor.universe import Etf, Universe

RETRIEVED_AT = datetime(2024, 7, 9, 1, 0, tzinfo=UTC)


def small_universe(*tickers: str) -> Universe:
    """Return an in-memory universe of fictional entries for the given tickers."""
    return Universe(
        version=1,
        etfs=tuple(Etf(ticker, f"FICTIONAL {ticker}", "Test") for ticker in tickers),
        sha256="FICTIONAL" + "".join(tickers),
    )


def nav_frame(
    rows: Sequence[tuple[str, str, float, int | None]],
    *,
    source: str = FICTIONAL_NAV_SOURCE,
) -> pd.DataFrame:
    """Build canonical NAV rows from (ticker, date, nav, shares) tuples."""
    return pd.DataFrame(
        {
            "ticker": pd.array([row[0] for row in rows], dtype="string"),
            "date": pd.to_datetime([row[1] for row in rows]).astype("datetime64[ns]"),
            "nav": pd.Series([row[2] for row in rows], dtype="float64"),
            "shares_outstanding": pd.array([row[3] for row in rows], dtype="Int64"),
            "total_net_assets": pd.Series(
                [float("nan") if row[3] is None else row[2] * row[3] for row in rows],
                dtype="float64",
            ),
            "retrieved_at": pd.Series(
                [pd.Timestamp(RETRIEVED_AT)] * len(rows), dtype="datetime64[ns, UTC]"
            ),
            "source": pd.array([source] * len(rows), dtype="string"),
        }
    )


class MappingNavTransport:
    """Return prebuilt fictional workbook bytes and record requests."""

    def __init__(self, responses: Mapping[str, bytes | Exception]) -> None:
        self.responses = dict(responses)
        self.calls: list[str] = []

    def fetch(self, ticker: str) -> bytes:
        self.calls.append(ticker)
        response = self.responses[ticker]
        if isinstance(response, Exception):
            raise response
        return response
