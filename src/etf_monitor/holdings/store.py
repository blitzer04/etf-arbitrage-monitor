"""Immutable holdings vintages: raw issuer workbooks plus a manifest.

Canonical rows are not stored separately: :func:`load_holdings_vintage`
re-derives them from the stored raw bytes, so the rows can never disagree
with the published files. Each capture is a new vintage; a fund's holdings
history is the sequence of its vintages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from etf_monitor.holdings.acquire import (
    HoldingsBatchResult,
    HoldingsTickerStatus,
    holdings_rows_from_payload,
)
from etf_monitor.holdings.schema import empty_holdings_frame, validate_holdings_frame
from etf_monitor.holdings.vintage import (
    VintageError,
    read_vintage_dir,
    write_vintage_dir,
)
from etf_monitor.universe import Universe

HOLDINGS_VINTAGE_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class HoldingsVintage:
    """A verified holdings vintage loaded from disk."""

    path: Path
    manifest: dict[str, object]
    frame: pd.DataFrame


def _status_record(status: HoldingsTickerStatus) -> dict[str, object]:
    return {
        "ticker": status.ticker,
        "status": status.status,
        "reason_code": status.reason_code,
        "detail": status.detail,
        "source": status.source,
        "retrieved_at": None
        if status.retrieved_at is None
        else status.retrieved_at.isoformat(),
        "raw_sha256": status.raw_sha256,
        "as_of": None if status.as_of is None else status.as_of.isoformat(),
        "layout": status.layout,
        "row_count": status.row_count,
    }


def write_holdings_vintage(
    batch: HoldingsBatchResult, root: Path, universe: Universe
) -> Path:
    """Persist one holdings batch as a new vintage and verify it by reloading.

    Raises:
        VintageError: If the batch does not match the universe or the vintage
            already exists.
    """
    if batch.universe_sha256 != universe.sha256 or tuple(
        status.ticker for status in batch.statuses
    ) != tuple(etf.ticker for etf in universe.etfs):
        raise VintageError("Batch statuses must match the universe order.")
    validate_holdings_frame(batch.frame, {etf.ticker for etf in universe.etfs})
    final = write_vintage_dir(
        root / f"holdings-{batch.batch_id}",
        {f"raw/{ticker}.xlsx": raw for ticker, raw in batch.raw_payloads.items()},
        {
            "format_version": HOLDINGS_VINTAGE_FORMAT_VERSION,
            "batch_id": batch.batch_id,
            "started_at": batch.started_at.isoformat(),
            "universe_sha256": batch.universe_sha256,
            "row_count": len(batch.frame),
            "statuses": [_status_record(status) for status in batch.statuses],
        },
    )
    load_holdings_vintage(final, universe)
    return final


def load_holdings_vintage(path: Path, universe: Universe) -> HoldingsVintage:
    """Load a holdings vintage, verifying digests and re-parsing raw files.

    Raises:
        VintageError: If files are missing, extra, or altered, the universe
            differs, or re-parsed rows disagree with the manifest.
    """
    manifest, payloads = read_vintage_dir(path, HOLDINGS_VINTAGE_FORMAT_VERSION)
    if manifest.get("universe_sha256") != universe.sha256:
        raise VintageError("Holdings vintage was fetched for a different universe.")
    statuses = manifest.get("statuses")
    if not isinstance(statuses, list):
        raise VintageError("Manifest must list statuses.")
    frames: list[pd.DataFrame] = []
    for status in statuses:
        if status["status"] != "success":
            continue
        frame = holdings_rows_from_payload(
            payloads[f"raw/{status['ticker']}.xlsx"],
            status["ticker"],
            datetime.fromisoformat(status["retrieved_at"]),
        )
        if (
            len(frame) != status["row_count"]
            or frame["as_of"].iloc[0].date().isoformat() != status["as_of"]
        ):
            raise VintageError(f"{status['ticker']} differs from its manifest status.")
        frames.append(frame)
    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "position"], kind="mergesort")
        .reset_index(drop=True)
        if frames
        else empty_holdings_frame()
    )
    validate_holdings_frame(combined, {etf.ticker for etf in universe.etfs})
    if len(combined) != manifest.get("row_count"):
        raise VintageError("Holdings row count differs from the manifest.")
    return HoldingsVintage(path=path, manifest=manifest, frame=combined)
