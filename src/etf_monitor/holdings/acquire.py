"""Fetch one batch of issuer holdings snapshots and validate each ticker."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from etf_monitor.holdings.schema import (
    HoldingsValidationError,
    empty_holdings_frame,
    holdings_frame,
    validate_holdings_frame,
)
from etf_monitor.holdings.vintage import sha256_hex
from etf_monitor.issuer_nav.sources import NavSource, nav_sources_for
from etf_monitor.providers.ssga_holdings import (
    SSGA_HOLDINGS_SOURCE,
    HoldingsPayloadError,
    parse_ssga_holdings,
)
from etf_monitor.providers.ssga_nav import SSGA_NAV_SOURCE
from etf_monitor.providers.transport import IssuerFileTransport, ProviderTransportError
from etf_monitor.universe import Universe

_NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class HoldingsTickerStatus:
    """Outcome of one holdings request.

    ``status`` is ``success``, ``failed`` (request or validation failed; no
    rows kept, raw bytes still stored) or ``unavailable`` (the fund's issuer
    has no configured holdings source; nothing requested).
    """

    ticker: str
    status: str
    reason_code: str | None
    detail: str | None
    source: str | None
    retrieved_at: datetime | None
    raw_sha256: str | None
    as_of: date | None
    layout: str | None
    row_count: int


@dataclass(frozen=True, slots=True)
class HoldingsBatchResult:
    """Validated rows, statuses, and exact raw payloads for one batch."""

    batch_id: str
    started_at: datetime
    universe_sha256: str
    statuses: tuple[HoldingsTickerStatus, ...]
    frame: pd.DataFrame
    raw_payloads: Mapping[str, bytes]


def holdings_source(nav_source: NavSource) -> str | None:
    """Return the holdings source for a fund; SSGA funds publish both files."""
    return SSGA_HOLDINGS_SOURCE if nav_source.source == SSGA_NAV_SOURCE else None


def holdings_rows_from_payload(
    raw: bytes, ticker: str, retrieved_at: datetime
) -> pd.DataFrame:
    """Parse, normalize, and validate one SSGA holdings payload.

    Raises:
        HoldingsPayloadError: If the workbook layout or values are invalid.
        HoldingsValidationError: If rows violate the contract, including an
            ``As of`` date after the New York calendar date of retrieval.
    """
    parsed = parse_ssga_holdings(raw, ticker)
    retrieval_day = retrieved_at.astimezone(_NEW_YORK).date()
    if parsed.as_of > retrieval_day:
        raise HoldingsValidationError(
            f"{ticker} holdings are dated {parsed.as_of}, after the retrieval "
            f"date {retrieval_day} in New York."
        )
    frame = holdings_frame(parsed, retrieved_at, SSGA_HOLDINGS_SOURCE)
    validate_holdings_frame(frame, {ticker})
    return frame


def _utc_now(clock: Callable[[], datetime]) -> datetime:
    instant = clock()
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("clock() must return a timezone-aware UTC datetime.")
    return instant


def fetch_holdings_batch(
    universe: Universe,
    transport: IssuerFileTransport,
    clock: Callable[[], datetime],
    sources: Mapping[str, NavSource] | None = None,
) -> HoldingsBatchResult:
    """Fetch the latest holdings snapshot once per fund, in universe order.

    No retries. A failure for one ticker never changes another ticker's rows.
    """
    started_at = _utc_now(clock)
    statuses: list[HoldingsTickerStatus] = []
    frames: list[pd.DataFrame] = []
    payloads: dict[str, bytes] = {}
    for nav_source in nav_sources_for(
        universe, None if sources is None else dict(sources)
    ):
        ticker, source = nav_source.ticker, holdings_source(nav_source)

        def failed(
            reason: str,
            detail: str,
            retrieved_at: datetime | None = None,
            digest: str | None = None,
            *,
            ticker: str = ticker,
            source: str | None = source,
        ) -> HoldingsTickerStatus:
            return HoldingsTickerStatus(
                ticker=ticker,
                status="failed" if source else "unavailable",
                reason_code=reason,
                detail=detail,
                source=source,
                retrieved_at=retrieved_at,
                raw_sha256=digest,
                as_of=None,
                layout=None,
                row_count=0,
            )

        if source is None:
            statuses.append(
                failed("no_configured_source", nav_source.reason or "non-SSGA fund")
            )
            continue
        try:
            raw = transport.fetch(ticker)
        except ProviderTransportError as error:
            statuses.append(failed("transport", str(error)))
            continue
        retrieved_at = _utc_now(clock)
        payloads[ticker] = raw
        digest = sha256_hex(raw)
        try:
            frame = holdings_rows_from_payload(raw, ticker, retrieved_at)
        except HoldingsPayloadError as error:
            statuses.append(failed(error.reason_code, str(error), retrieved_at, digest))
            continue
        except HoldingsValidationError as error:
            statuses.append(
                failed("holdings_contract", str(error), retrieved_at, digest)
            )
            continue
        statuses.append(
            HoldingsTickerStatus(
                ticker=ticker,
                status="success",
                reason_code=None,
                detail=None,
                source=source,
                retrieved_at=retrieved_at,
                raw_sha256=digest,
                as_of=frame["as_of"].iloc[0].date(),
                layout=str(frame["layout"].iloc[0]),
                row_count=len(frame),
            )
        )
        frames.append(frame)
    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "position"], kind="mergesort")
        .reset_index(drop=True)
        if frames
        else empty_holdings_frame()
    )
    validate_holdings_frame(combined, {etf.ticker for etf in universe.etfs})
    return HoldingsBatchResult(
        batch_id=started_at.strftime("%Y%m%dT%H%M%S%fZ"),
        started_at=started_at,
        universe_sha256=universe.sha256,
        statuses=tuple(statuses),
        frame=combined,
        raw_payloads=payloads,
    )
