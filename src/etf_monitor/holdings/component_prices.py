"""Yahoo closes for the securities in issuer holdings, kept as raw vintages.

Only U.S.-listed common-stock style tickers are requested: SSGA writes share
classes with a dot (``BRK.B``) where Yahoo uses a hyphen (``BRK-B``). Index
futures, contingent value rights, earn-outs, and cash lines have no Yahoo
symbol and are never requested.

``close`` is Yahoo's split-adjusted, not dividend-adjusted, regular-session
close; split events from the same payload let callers recover the as-traded
close on a past session (see :mod:`etf_monitor.arbitrage.splits`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from etf_monitor.arbitrage.splits import (
    SplitEvent,
    SplitPayloadError,
    split_events_from_chart,
)
from etf_monitor.holdings.vintage import (
    VintageError,
    read_vintage_dir,
    sha256_hex,
    write_vintage_dir,
)
from etf_monitor.providers.transport import ChartTransport, ProviderTransportError
from etf_monitor.providers.yahoo_chart import ChartPayloadError, parse_chart_payload

COMPONENT_VINTAGE_FORMAT_VERSION = 1
COMPONENT_PRICE_COLUMNS = ("symbol", "date", "close")
_NEW_YORK = ZoneInfo("America/New_York")
_ISSUER_TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


def yahoo_symbol(security_ticker: str | None) -> str | None:
    """Map an issuer security ticker to a Yahoo symbol, or None if unmappable."""
    if security_ticker is None or _ISSUER_TICKER.fullmatch(security_ticker) is None:
        return None
    return security_ticker.replace(".", "-")


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    """Outcome of one component chart request (``success`` or ``failed``)."""

    symbol: str
    status: str
    reason_code: str | None
    detail: str | None
    retrieved_at: datetime | None
    raw_sha256: str | None
    row_count: int


@dataclass(frozen=True, slots=True)
class ComponentPriceBatch:
    """Closes, split events, statuses, and raw payloads for one batch."""

    batch_id: str
    started_at: datetime
    start: date
    end_exclusive: date
    statuses: tuple[ComponentStatus, ...]
    frame: pd.DataFrame
    splits: tuple[SplitEvent, ...]
    raw_payloads: Mapping[str, bytes]


def empty_component_frame() -> pd.DataFrame:
    """Return an empty component-price frame with canonical dtypes."""
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype="string"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "close": pd.Series(dtype="float64"),
        }
    )


def component_rows_from_payload(
    raw: bytes, symbol: str, start: date, end_exclusive: date
) -> tuple[pd.DataFrame, tuple[SplitEvent, ...]]:
    """Parse one raw chart payload into closes and split events.

    Sessions whose close is null are dropped, so they stay missing downstream.

    Raises:
        ChartPayloadError: If the payload violates the chart contract.
        SplitPayloadError: If its split events are invalid.
    """
    chart = parse_chart_payload(raw, symbol, start, end_exclusive)
    kept = [row for row in chart.observations if row.close is not None]
    frame = pd.DataFrame(
        {
            "symbol": pd.array([symbol] * len(kept), dtype="string"),
            "date": pd.to_datetime([row.date for row in kept]).astype("datetime64[ns]"),
            "close": np.array([row.close for row in kept], dtype=np.float64),
        }
    )
    return frame, split_events_from_chart(raw, symbol)


def _combine(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    if not frames:
        return empty_component_frame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol", "date"], kind="mergesort")
        .reset_index(drop=True)
    )


def _epoch(value: date) -> int:
    return int(datetime.combine(value, time(), tzinfo=_NEW_YORK).timestamp())


def fetch_component_prices(
    symbols: Iterable[str],
    start: date,
    end_exclusive: date,
    transport: ChartTransport,
    clock: Callable[[], datetime],
) -> ComponentPriceBatch:
    """Request one chart per unique symbol, sequentially and without retries."""
    if start >= end_exclusive:
        raise ValueError("start must be earlier than end_exclusive.")
    started_at = clock()
    if started_at.utcoffset() is None:
        raise ValueError("clock() must return a timezone-aware UTC datetime.")
    statuses: list[ComponentStatus] = []
    frames: list[pd.DataFrame] = []
    splits: list[SplitEvent] = []
    payloads: dict[str, bytes] = {}
    for symbol in sorted(set(symbols)):
        try:
            raw = transport.fetch(symbol, _epoch(start), _epoch(end_exclusive))
        except ProviderTransportError as error:
            statuses.append(
                ComponentStatus(
                    symbol, "failed", "transport", str(error), None, None, 0
                )
            )
            continue
        retrieved_at = clock()
        payloads[symbol] = raw
        digest = sha256_hex(raw)
        try:
            frame, events = component_rows_from_payload(
                raw, symbol, start, end_exclusive
            )
        except (ChartPayloadError, SplitPayloadError) as error:
            reason = getattr(error, "reason_code", "split_events")
            statuses.append(
                ComponentStatus(
                    symbol, "failed", reason, str(error), retrieved_at, digest, 0
                )
            )
            continue
        statuses.append(
            ComponentStatus(
                symbol, "success", None, None, retrieved_at, digest, len(frame)
            )
        )
        frames.append(frame)
        splits.extend(events)
    return ComponentPriceBatch(
        batch_id=started_at.strftime("%Y%m%dT%H%M%S%fZ"),
        started_at=started_at,
        start=start,
        end_exclusive=end_exclusive,
        statuses=tuple(statuses),
        frame=_combine(frames),
        splits=tuple(splits),
        raw_payloads=payloads,
    )


def write_component_vintage(batch: ComponentPriceBatch, root: Path) -> Path:
    """Persist raw chart payloads and statuses as a new verified vintage."""
    final = write_vintage_dir(
        root / f"components-{batch.batch_id}",
        {f"raw/{symbol}.json": raw for symbol, raw in batch.raw_payloads.items()},
        {
            "format_version": COMPONENT_VINTAGE_FORMAT_VERSION,
            "batch_id": batch.batch_id,
            "started_at": batch.started_at.isoformat(),
            "start": batch.start.isoformat(),
            "end_exclusive": batch.end_exclusive.isoformat(),
            "statuses": [
                {
                    "symbol": status.symbol,
                    "status": status.status,
                    "reason_code": status.reason_code,
                    "detail": status.detail,
                    "retrieved_at": None
                    if status.retrieved_at is None
                    else status.retrieved_at.isoformat(),
                    "raw_sha256": status.raw_sha256,
                    "row_count": status.row_count,
                }
                for status in batch.statuses
            ],
        },
    )
    load_component_vintage(final)
    return final


@dataclass(frozen=True, slots=True)
class ComponentVintage:
    """A verified component-price vintage re-derived from raw payloads."""

    path: Path
    manifest: dict[str, object]
    frame: pd.DataFrame
    splits: tuple[SplitEvent, ...]
    failed_symbols: tuple[str, ...]


def load_component_vintage(path: Path) -> ComponentVintage:
    """Load a component vintage, verifying digests and re-parsing every payload.

    Raises:
        VintageError: If files are altered or re-parsed rows disagree with the
            manifest.
    """
    manifest, payloads = read_vintage_dir(path, COMPONENT_VINTAGE_FORMAT_VERSION)
    start = date.fromisoformat(str(manifest["start"]))
    end_exclusive = date.fromisoformat(str(manifest["end_exclusive"]))
    statuses = manifest.get("statuses")
    if not isinstance(statuses, list):
        raise VintageError("Manifest must list statuses.")
    frames: list[pd.DataFrame] = []
    splits: list[SplitEvent] = []
    failed: list[str] = []
    for status in statuses:
        if status["status"] != "success":
            failed.append(status["symbol"])
            continue
        frame, events = component_rows_from_payload(
            payloads[f"raw/{status['symbol']}.json"],
            status["symbol"],
            start,
            end_exclusive,
        )
        if len(frame) != status["row_count"]:
            raise VintageError(f"{status['symbol']} differs from its manifest status.")
        frames.append(frame)
        splits.extend(events)
    return ComponentVintage(
        path=path,
        manifest=manifest,
        frame=_combine(frames),
        splits=tuple(splits),
        failed_symbols=tuple(failed),
    )
