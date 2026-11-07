"""Shared fully fictional price-pipeline test builders."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from etf_monitor.prices.acquire import BatchResult, fetch_price_batch
from etf_monitor.prices.schema import CANONICAL_COLUMNS, PRICE_SOURCE
from etf_monitor.universe import load_universe
from tests.fixtures.yahoo_chart import build_yahoo_chart_payload

DEFAULT_START = date(2024, 7, 1)
DEFAULT_END = date(2024, 7, 10)
DEFAULT_RETRIEVED_AT = datetime(2024, 7, 10, 16, 0, tzinfo=UTC)


class MappingTransport:
    """Return prebuilt fictional chart bodies and record exact requests."""

    def __init__(
        self,
        responses: Mapping[str, bytes | Exception] | None = None,
    ) -> None:
        self.responses = dict(responses or {})
        self.calls: list[tuple[str, int, int]] = []

    def fetch(self, symbol: str, period1: int, period2: int) -> bytes:
        self.calls.append((symbol, period1, period2))
        response = self.responses.get(
            symbol,
            json.dumps(build_yahoo_chart_payload(ticker=symbol, dates=())).encode(
                "utf-8"
            ),
        )
        if isinstance(response, Exception):
            raise response
        return response


def fictional_payload(
    ticker: str = "SPY",
    dates: tuple[str, ...] = ("2024-07-02",),
    *,
    close: float | int = 101.0,
    adjusted_close: float | int | None = 100.5,
    volume: int | None = 1_000_000,
    **kwargs: object,
) -> bytes:
    """Encode one deterministic fictional Yahoo-shaped payload."""
    opens = tuple(100.0 + index for index in range(len(dates)))
    closes = tuple(close + index for index in range(len(dates)))
    values: dict[str, tuple[float | int | None, ...]] = {
        "open": opens,
        "high": tuple(
            max(102.0 + index, open_value, close_value)
            for index, (open_value, close_value) in enumerate(
                zip(opens, closes, strict=True)
            )
        ),
        "low": tuple(
            min(99.0 + index, open_value, close_value)
            for index, (open_value, close_value) in enumerate(
                zip(opens, closes, strict=True)
            )
        ),
        "close": closes,
        "volume": tuple(
            (1_000_000 if volume is None else volume) + index
            for index in range(len(dates))
        ),
    }
    if volume is None:
        kwargs["null_quote_positions"] = {"volume": tuple(range(len(dates)))}
    if adjusted_close is None:
        kwargs["null_adjusted_positions"] = tuple(range(len(dates)))
    payload = build_yahoo_chart_payload(
        ticker=ticker,
        dates=dates,
        quote_values=values,
        adjusted_values=tuple(
            adjusted_close + index
            for index in range(len(dates))
            if adjusted_close is not None
        )
        if adjusted_close is not None
        else None,
        **kwargs,  # type: ignore[arg-type]
    )
    return json.dumps(payload).encode("utf-8")


def make_batch(
    responses: Mapping[str, bytes | Exception] | None = None,
    *,
    start: date = DEFAULT_START,
    end_exclusive: date = DEFAULT_END,
    retrieved_at: datetime = DEFAULT_RETRIEVED_AT,
    batch_instant: datetime | None = None,
    raw_dir: Path | None = None,
) -> BatchResult:
    """Acquire one fake batch; unlisted tickers return genuine empty fixtures."""
    clock_calls = 0

    def clock() -> datetime:
        nonlocal clock_calls
        clock_calls += 1
        if batch_instant is not None and clock_calls == 1:
            return batch_instant
        return retrieved_at

    return fetch_price_batch(
        load_universe(),
        start,
        end_exclusive,
        MappingTransport(responses),
        clock,
        raw_dir=raw_dir,
    )


def canonical_frame(
    *,
    ticker: str = "SPY",
    day: str = "2024-07-02",
    open_price: float = 100.0,
    high: float = 102.0,
    low: float = 99.0,
    close: float = 101.0,
    adjusted_close: float | None = 100.5,
    volume: int | None = 1_000_000,
    retrieved_at: datetime = DEFAULT_RETRIEVED_AT,
    source: str = PRICE_SOURCE,
) -> pd.DataFrame:
    """Build one fictional row using exact canonical pandas dtypes."""
    return pd.DataFrame(
        {
            "ticker": pd.Series([ticker], dtype="string"),
            "date": pd.to_datetime([day]),
            "open": pd.Series([open_price], dtype="float64"),
            "high": pd.Series([high], dtype="float64"),
            "low": pd.Series([low], dtype="float64"),
            "close": pd.Series([close], dtype="float64"),
            "adjusted_close": pd.Series(
                [float("nan") if adjusted_close is None else adjusted_close],
                dtype="float64",
            ),
            "volume": pd.Series(pd.array([volume], dtype="Int64")),
            "retrieved_at": pd.Series(
                [pd.Timestamp(retrieved_at)], dtype="datetime64[ns, UTC]"
            ),
            "source": pd.Series([source], dtype="string"),
        },
        columns=CANONICAL_COLUMNS,
    )
