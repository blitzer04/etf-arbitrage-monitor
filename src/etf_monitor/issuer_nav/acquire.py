"""Fetch one batch of issuer NAV-history files and validate each ticker."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from etf_monitor.issuer_nav.schema import (
    NavValidationError,
    empty_nav_frame,
    nav_frame_from_history,
    validate_nav_frame,
)
from etf_monitor.issuer_nav.sources import NavSource, nav_sources_for
from etf_monitor.providers.ssga_nav import (
    SSGA_NAV_SOURCE,
    NavPayloadError,
    parse_ssga_nav_history,
)
from etf_monitor.providers.transport import IssuerFileTransport, ProviderTransportError
from etf_monitor.universe import Universe
from etf_monitor.xnys import xnys_sessions

_NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class NavTickerStatus:
    """Outcome and data-quality counts for one ETF in a NAV batch.

    ``status`` is ``success``, ``failed`` (request or validation failed; no
    rows kept) or ``unavailable`` (no source configured; nothing requested).
    ``missing_sessions`` lists XNYS sessions between the first and last kept
    date that have no issuer row; they stay missing and are never filled.
    """

    ticker: str
    status: str
    reason_code: str | None
    detail: str | None
    source: str | None
    retrieved_at: datetime | None
    raw_sha256: str | None
    row_count: int
    first_date: date | None
    last_date: date | None
    rows_before_history_start: int
    missing_shares_outstanding: int
    missing_total_net_assets: int
    missing_sessions: tuple[date, ...]


@dataclass(frozen=True, slots=True)
class NavBatchResult:
    """Validated rows, statuses, and exact raw payloads for one batch."""

    batch_id: str
    started_at: datetime
    history_start: date
    universe_sha256: str
    statuses: tuple[NavTickerStatus, ...]
    frame: pd.DataFrame
    raw_payloads: Mapping[str, bytes]


def _utc_now(clock: Callable[[], datetime]) -> datetime:
    instant = clock()
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("clock() must return a timezone-aware UTC datetime.")
    return instant


def _failed(
    source: NavSource,
    reason_code: str,
    detail: str,
    retrieved_at: datetime | None = None,
    raw_sha256: str | None = None,
) -> NavTickerStatus:
    return NavTickerStatus(
        ticker=source.ticker,
        status="failed" if source.source else "unavailable",
        reason_code=reason_code,
        detail=detail,
        source=source.source,
        retrieved_at=retrieved_at,
        raw_sha256=raw_sha256,
        row_count=0,
        first_date=None,
        last_date=None,
        rows_before_history_start=0,
        missing_shares_outstanding=0,
        missing_total_net_assets=0,
        missing_sessions=(),
    )


def nav_rows_from_payload(
    raw: bytes,
    ticker: str,
    history_start: date,
    retrieved_at: datetime,
) -> tuple[pd.DataFrame, int]:
    """Parse, normalize, and validate one SSGA payload for one ticker.

    Args:
        raw: Exact workbook bytes.
        ticker: Requested ticker, which the workbook must identify.
        history_start: First methodology date kept.
        retrieved_at: UTC instant the bytes were received.

    Returns:
        Canonical NAV rows and the number of rows dated before
        ``history_start``.

    Raises:
        NavPayloadError: If the workbook layout or values are invalid.
        NavValidationError: If normalized rows violate the NAV contract,
            including any date after the New York calendar date of retrieval.
    """
    history = parse_ssga_nav_history(raw, ticker, history_start)
    frame = nav_frame_from_history(history, retrieved_at, SSGA_NAV_SOURCE)
    validate_nav_frame(frame, {ticker})
    retrieval_day = retrieved_at.astimezone(_NEW_YORK).date()
    if history.observations[-1].date > retrieval_day:
        raise NavValidationError(
            f"{ticker} reports NAV dated {history.observations[-1].date}, after "
            f"the retrieval date {retrieval_day} in New York."
        )
    return frame, history.rows_before_history_start


def fetch_nav_batch(
    universe: Universe,
    history_start: date,
    transport: IssuerFileTransport,
    clock: Callable[[], datetime],
    sources: Mapping[str, NavSource] | None = None,
) -> NavBatchResult:
    """Fetch, parse, and validate issuer NAV history for every universe ETF.

    One request per ETF with a configured source, in universe order, with no
    retries. A failure for one ticker never changes another ticker's rows.

    Args:
        universe: Validated universe whose ETFs are requested.
        history_start: First methodology date kept from each history.
        transport: Issuer file transport.
        clock: UTC clock called once at batch start and after each response.
        sources: Optional NAV source mapping; defaults to the packaged file.

    Returns:
        Combined validated rows, per-ticker statuses, and raw payload bytes.
    """
    started_at = _utc_now(clock)
    batch_id = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    statuses: list[NavTickerStatus] = []
    frames: list[pd.DataFrame] = []
    payloads: dict[str, bytes] = {}
    for source in nav_sources_for(universe, None if sources is None else dict(sources)):
        if source.source is None:
            statuses.append(
                _failed(source, "no_configured_source", source.reason or "")
            )
            continue
        try:
            raw = transport.fetch(source.ticker)
        except ProviderTransportError as error:
            statuses.append(_failed(source, "transport", str(error)))
            continue
        retrieved_at = _utc_now(clock)
        payloads[source.ticker] = raw
        digest = hashlib.sha256(raw).hexdigest()
        try:
            frame, earlier_rows = nav_rows_from_payload(
                raw, source.ticker, history_start, retrieved_at
            )
        except NavPayloadError as error:
            statuses.append(
                _failed(source, error.reason_code, str(error), retrieved_at, digest)
            )
            continue
        except NavValidationError as error:
            statuses.append(
                _failed(source, "nav_contract", str(error), retrieved_at, digest)
            )
            continue
        first, last = frame["date"].iloc[0].date(), frame["date"].iloc[-1].date()
        expected = xnys_sessions(first, last)
        missing = tuple(
            label.date()
            for label in expected.difference(pd.DatetimeIndex(frame["date"]))
        )
        statuses.append(
            NavTickerStatus(
                ticker=source.ticker,
                status="success",
                reason_code=None,
                detail=None,
                source=source.source,
                retrieved_at=retrieved_at,
                raw_sha256=digest,
                row_count=len(frame),
                first_date=first,
                last_date=last,
                rows_before_history_start=earlier_rows,
                missing_shares_outstanding=int(
                    frame["shares_outstanding"].isna().sum()
                ),
                missing_total_net_assets=int(frame["total_net_assets"].isna().sum()),
                missing_sessions=missing,
            )
        )
        frames.append(frame)

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"], kind="mergesort")
        .reset_index(drop=True)
        if frames
        else empty_nav_frame()
    )
    validate_nav_frame(combined, {etf.ticker for etf in universe.etfs})
    return NavBatchResult(
        batch_id=batch_id,
        started_at=started_at,
        history_start=history_start,
        universe_sha256=universe.sha256,
        statuses=tuple(statuses),
        frame=combined,
        raw_payloads=payloads,
    )
