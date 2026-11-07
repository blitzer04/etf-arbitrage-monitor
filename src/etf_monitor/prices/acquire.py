"""Sequential per-ETF Yahoo chart acquisition with deterministic statuses."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from etf_monitor.prices.schema import (
    CANONICAL_COLUMNS,
    PRICE_SOURCE,
    PriceValidationError,
    canonicalize_frame,
    empty_canonical_frame,
    validate_canonical,
)
from etf_monitor.providers.transport import ChartTransport
from etf_monitor.providers.yahoo_chart import (
    ChartPayloadError,
    ParsedChart,
    parse_chart_payload,
)
from etf_monitor.universe import Universe
from etf_monitor.xnys import xnys_sessions

_NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class TickerStatus:
    """Record the request window, provider outcome, and returned coverage."""

    ticker: str
    status: Literal["success", "empty", "failed"]
    reason: str | None
    detail: str | None
    request_start: date
    request_end_exclusive: date
    retrieved_at: pd.Timestamp | None
    returned_dates: tuple[date, ...]
    row_count: int
    first_date: date | None
    last_date: date | None
    null_row_dates: tuple[date, ...]
    adjclose_block_absent: bool
    trailing_missing_sessions: tuple[date, ...]
    raw_sha256: str | None


@dataclass(frozen=True, slots=True)
class BatchResult:
    """One immutable multi-ticker acquisition result."""

    batch_id: str
    request_start: date
    request_end_exclusive: date
    statuses: tuple[TickerStatus, ...]
    rows: pd.DataFrame


def _period_epoch(value: date) -> int:
    """Convert a local New York midnight to Unix epoch seconds."""
    return int(datetime.combine(value, time.min, tzinfo=_NEW_YORK).timestamp())


def _utc_timestamp(value: datetime) -> pd.Timestamp:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock() must return a timezone-aware UTC datetime.")
    if value.utcoffset() != timedelta(0):
        raise ValueError("clock() must return a UTC datetime.")
    return pd.Timestamp(value).tz_convert("UTC")


def _session_dates(start: date, end_exclusive: date) -> tuple[date, ...]:
    if start >= end_exclusive:
        raise ValueError("start must be earlier than end_exclusive.")
    return tuple(
        timestamp.date()
        for timestamp in xnys_sessions(start, end_exclusive - timedelta(days=1))
    )


def _save_raw_response(
    raw_dir: Path | None, batch_id: str, ticker: str, raw: bytes
) -> None:
    if raw_dir is None:
        return
    destination = raw_dir / batch_id / f"{ticker}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)


def _failed_status(
    ticker: str,
    start: date,
    end_exclusive: date,
    reason: str,
    retrieved_at: pd.Timestamp | None,
    *,
    detail: str | None = None,
    raw_sha256: str | None = None,
    null_row_dates: tuple[date, ...] = (),
    adjclose_block_absent: bool = False,
) -> TickerStatus:
    return TickerStatus(
        ticker=ticker,
        status="failed",
        reason=reason,
        detail=detail,
        request_start=start,
        request_end_exclusive=end_exclusive,
        retrieved_at=retrieved_at,
        returned_dates=(),
        row_count=0,
        first_date=None,
        last_date=None,
        null_row_dates=null_row_dates,
        adjclose_block_absent=adjclose_block_absent,
        trailing_missing_sessions=(),
        raw_sha256=raw_sha256,
    )


def _frame_for_chart(
    ticker: str, chart: ParsedChart, retrieved_at: pd.Timestamp
) -> pd.DataFrame:
    if not chart.observations:
        return empty_canonical_frame()
    data: dict[str, object] = {
        "ticker": pd.Series([ticker] * len(chart.observations), dtype="string"),
        "date": pd.to_datetime(
            [observation.date for observation in chart.observations]
        ),
    }
    for field in ("open", "high", "low", "close", "adjusted_close"):
        data[field] = pd.Series(
            [
                np.nan
                if getattr(observation, field) is None
                else getattr(observation, field)
                for observation in chart.observations
            ],
            dtype="float64",
        )
    data["volume"] = pd.Series(
        pd.array(
            [observation.volume for observation in chart.observations],
            dtype="Int64",
        )
    )
    data["retrieved_at"] = pd.Series(
        [retrieved_at] * len(chart.observations), dtype="datetime64[ns, UTC]"
    )
    data["source"] = pd.Series([PRICE_SOURCE] * len(chart.observations), dtype="string")
    frame = pd.DataFrame(data, columns=CANONICAL_COLUMNS)
    return canonicalize_frame(frame)


def _status_from_chart(
    ticker: str,
    start: date,
    end_exclusive: date,
    retrieved_at: pd.Timestamp,
    chart: ParsedChart,
    raw_sha256: str,
    frame: pd.DataFrame,
    sessions: tuple[date, ...],
) -> TickerStatus:
    if chart.empty:
        return TickerStatus(
            ticker=ticker,
            status="empty",
            reason="empty_response",
            detail=None,
            request_start=start,
            request_end_exclusive=end_exclusive,
            retrieved_at=retrieved_at,
            returned_dates=(),
            row_count=0,
            first_date=None,
            last_date=None,
            null_row_dates=(),
            adjclose_block_absent=chart.adjclose_block_absent,
            trailing_missing_sessions=sessions,
            raw_sha256=raw_sha256,
        )

    last_date = chart.returned_dates[-1] if chart.returned_dates else None
    trailing_sessions = tuple(
        session for session in sessions if last_date is None or session > last_date
    )
    return TickerStatus(
        ticker=ticker,
        status="success",
        reason=None,
        detail=None,
        request_start=start,
        request_end_exclusive=end_exclusive,
        retrieved_at=retrieved_at,
        returned_dates=chart.returned_dates,
        row_count=len(frame),
        first_date=chart.returned_dates[0] if chart.returned_dates else None,
        last_date=last_date,
        null_row_dates=chart.null_row_dates,
        adjclose_block_absent=chart.adjclose_block_absent,
        trailing_missing_sessions=trailing_sessions,
        raw_sha256=raw_sha256,
    )


def fetch_price_batch(
    universe: Universe,
    start: date,
    end_exclusive: date,
    transport: ChartTransport,
    clock: Callable[[], datetime],
    raw_dir: Path | None = None,
) -> BatchResult:
    """Fetch and validate one chart response for each universe ETF in order.

    A failed ticker never discards successful rows from other tickers. Requests
    are sequential, use local New York midnight bounds, and are not retried.

    Args:
        universe: Validated, ordered ETF universe.
        start: Inclusive request start date.
        end_exclusive: Exclusive request end date.
        transport: Injected chart transport; production and tests use separate
            implementations.
        clock: Injected retrieval-time source, required to return UTC.
        raw_dir: Optional local destination for exact response bytes.

    Returns:
        Per-ticker outcomes and canonical rows from successful tickers only.
    """
    sessions = _session_dates(start, end_exclusive)
    batch_instant = _utc_timestamp(clock())
    batch_id = batch_instant.strftime("%Y%m%dT%H%M%S%fZ")
    period1 = _period_epoch(start)
    period2 = _period_epoch(end_exclusive)
    statuses: list[TickerStatus] = []
    successful_frames: list[pd.DataFrame] = []

    for etf in universe.etfs:
        try:
            raw = transport.fetch(etf.ticker, period1, period2)
        except Exception as error:
            statuses.append(
                _failed_status(
                    etf.ticker,
                    start,
                    end_exclusive,
                    "transport_error",
                    None,
                    detail=str(error),
                )
            )
            continue

        retrieved_at = _utc_timestamp(clock())
        if not isinstance(raw, bytes):
            statuses.append(
                _failed_status(
                    etf.ticker,
                    start,
                    end_exclusive,
                    "transport_error",
                    retrieved_at,
                    detail="transport returned a non-bytes response",
                )
            )
            continue
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            _save_raw_response(raw_dir, batch_id, etf.ticker, raw)
        except OSError as error:
            statuses.append(
                _failed_status(
                    etf.ticker,
                    start,
                    end_exclusive,
                    "raw_write_error",
                    retrieved_at,
                    detail=str(error),
                    raw_sha256=raw_sha256,
                )
            )
            continue
        try:
            chart = parse_chart_payload(raw, etf.ticker, start, end_exclusive)
        except ChartPayloadError as error:
            statuses.append(
                _failed_status(
                    etf.ticker,
                    start,
                    end_exclusive,
                    error.reason_code,
                    retrieved_at,
                    detail=str(error),
                    raw_sha256=raw_sha256,
                )
            )
            continue

        try:
            frame = _frame_for_chart(etf.ticker, chart, retrieved_at)
            if not frame.empty:
                validate_canonical(frame, universe)
        except (PriceValidationError, ValueError, TypeError, OverflowError) as error:
            statuses.append(
                _failed_status(
                    etf.ticker,
                    start,
                    end_exclusive,
                    "invalid_observation",
                    retrieved_at,
                    detail=str(error),
                    raw_sha256=raw_sha256,
                    null_row_dates=chart.null_row_dates,
                    adjclose_block_absent=chart.adjclose_block_absent,
                )
            )
            continue

        status = _status_from_chart(
            etf.ticker,
            start,
            end_exclusive,
            retrieved_at,
            chart,
            raw_sha256,
            frame,
            sessions,
        )
        statuses.append(status)
        if status.status == "success" and not frame.empty:
            successful_frames.append(frame)

    if successful_frames:
        combined = canonicalize_frame(pd.concat(successful_frames, ignore_index=True))
        validate_canonical(combined, universe)
    else:
        combined = empty_canonical_frame()
    return BatchResult(
        batch_id=batch_id,
        request_start=start,
        request_end_exclusive=end_exclusive,
        statuses=tuple(statuses),
        rows=combined,
    )
