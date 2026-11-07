"""Immutable on-disk NAV vintages: raw issuer files, rows, and a manifest.

A vintage directory holds the exact raw workbook bytes for every ticker
requested in one batch (``raw/<TICKER>.xlsx``), the validated canonical rows
(``nav.parquet``), and ``manifest.json`` with statuses and SHA-256 digests of
every file. Each vintage is written to a temporary sibling directory and
renamed into place, so a vintage is either complete or absent. Vintages are
never merged or overwritten; a later download is a new vintage, which keeps
issuer restatements visible instead of silently replacing history.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from etf_monitor.issuer_nav.acquire import (
    NavBatchResult,
    NavTickerStatus,
    nav_rows_from_payload,
)
from etf_monitor.issuer_nav.schema import (
    NAV_COLUMNS,
    empty_nav_frame,
    validate_nav_frame,
)
from etf_monitor.universe import Universe

NAV_VINTAGE_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
FRAME_NAME = "nav.parquet"
_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string()),
        pa.field("date", pa.timestamp("ns")),
        pa.field("nav", pa.float64()),
        pa.field("shares_outstanding", pa.int64()),
        pa.field("total_net_assets", pa.float64()),
        pa.field("retrieved_at", pa.timestamp("ns", tz="UTC")),
        pa.field("source", pa.string()),
    ]
)


class NavVintageError(ValueError):
    """Indicate that a NAV vintage cannot be written or fails verification."""


@dataclass(frozen=True, slots=True)
class NavVintage:
    """A verified NAV vintage loaded from disk."""

    path: Path
    manifest: dict[str, object]
    frame: pd.DataFrame


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _iso_date(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def _iso_instant(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _status_record(status: NavTickerStatus) -> dict[str, object]:
    return {
        "ticker": status.ticker,
        "status": status.status,
        "reason_code": status.reason_code,
        "detail": status.detail,
        "source": status.source,
        "retrieved_at": _iso_instant(status.retrieved_at),
        "raw_sha256": status.raw_sha256,
        "row_count": status.row_count,
        "first_date": _iso_date(status.first_date),
        "last_date": _iso_date(status.last_date),
        "rows_before_history_start": status.rows_before_history_start,
        "missing_shares_outstanding": status.missing_shares_outstanding,
        "missing_total_net_assets": status.missing_total_net_assets,
        "missing_sessions": [_iso_date(day) for day in status.missing_sessions],
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _frame_bytes(frame: pd.DataFrame, scratch: Path) -> bytes:
    table = pa.Table.from_pandas(
        frame, schema=_PARQUET_SCHEMA, preserve_index=False, safe=True
    )
    pq.write_table(table, scratch)
    payload = scratch.read_bytes()
    scratch.unlink()
    return payload


def _read_frame(path: Path) -> pd.DataFrame:
    table = pq.read_table(path)
    if not table.schema.equals(_PARQUET_SCHEMA, check_metadata=False):
        raise NavVintageError("nav.parquet does not match the approved Arrow schema.")
    frame = table.to_pandas(
        types_mapper=lambda arrow_type: (
            pd.Int64Dtype()
            if pa.types.is_int64(arrow_type)
            else pd.StringDtype()
            if pa.types.is_string(arrow_type)
            else None
        )
    )
    return cast(pd.DataFrame, frame.loc[:, list(NAV_COLUMNS)])


def write_nav_vintage(batch: NavBatchResult, root: Path, universe: Universe) -> Path:
    """Persist one NAV batch as a new immutable vintage directory.

    Args:
        batch: Result of :func:`etf_monitor.issuer_nav.acquire.fetch_nav_batch`.
        root: Directory that holds NAV vintages.
        universe: Universe the batch was fetched for.

    Returns:
        Path of the new vintage directory, verified by reloading it.

    Raises:
        NavVintageError: If the batch does not match the universe, the vintage
            already exists, or reload verification fails.
    """
    if batch.universe_sha256 != universe.sha256 or tuple(
        status.ticker for status in batch.statuses
    ) != tuple(etf.ticker for etf in universe.etfs):
        raise NavVintageError("Batch statuses must match the universe order.")
    validate_nav_frame(batch.frame, {etf.ticker for etf in universe.etfs})
    final = root / f"nav-{batch.batch_id}"
    if final.exists():
        raise NavVintageError(f"NAV vintage {final} already exists.")
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".staging-{batch.batch_id}-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        files: dict[str, str] = {}
        for ticker, payload in sorted(batch.raw_payloads.items()):
            relative = f"raw/{ticker}.xlsx"
            _write_bytes(staging / relative, payload)
            files[relative] = _sha256(payload)
        frame_payload = _frame_bytes(batch.frame, staging / ".frame.tmp")
        _write_bytes(staging / FRAME_NAME, frame_payload)
        files[FRAME_NAME] = _sha256(frame_payload)
        manifest: dict[str, object] = {
            "format_version": NAV_VINTAGE_FORMAT_VERSION,
            "batch_id": batch.batch_id,
            "started_at": batch.started_at.isoformat(),
            "history_start": batch.history_start.isoformat(),
            "universe_sha256": batch.universe_sha256,
            "row_count": len(batch.frame),
            "statuses": [_status_record(status) for status in batch.statuses],
            "files": files,
        }
        _write_bytes(
            staging / MANIFEST_NAME,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    load_nav_vintage(final, universe)
    return final


def load_nav_vintage(path: Path, universe: Universe) -> NavVintage:
    """Load a NAV vintage after verifying digests and re-deriving rows from raw.

    Args:
        path: Vintage directory.
        universe: Universe the vintage must have been fetched for.

    Returns:
        The manifest and the validated canonical frame.

    Raises:
        NavVintageError: If any file is missing, extra, altered, or the rows
            differ from those re-parsed from the stored raw workbooks.
    """
    try:
        manifest = json.loads((path / MANIFEST_NAME).read_text())
    except (OSError, ValueError) as error:
        raise NavVintageError(f"Cannot read NAV manifest in {path}: {error}") from error
    if manifest.get("format_version") != NAV_VINTAGE_FORMAT_VERSION:
        raise NavVintageError("Unsupported NAV vintage format version.")
    if manifest.get("universe_sha256") != universe.sha256:
        raise NavVintageError("NAV vintage was fetched for a different universe.")
    files = manifest["files"]
    on_disk = {
        file.relative_to(path).as_posix()
        for file in path.rglob("*")
        if file.is_file() and file.name != MANIFEST_NAME
    }
    if on_disk != set(files):
        raise NavVintageError("NAV vintage files differ from the manifest.")
    for relative, digest in files.items():
        if _sha256((path / relative).read_bytes()) != digest:
            raise NavVintageError(f"Digest mismatch for {relative}.")
    frame = _read_frame(path / FRAME_NAME)
    allowed = {etf.ticker for etf in universe.etfs}
    validate_nav_frame(frame, allowed)
    if len(frame) != manifest["row_count"]:
        raise NavVintageError("NAV row count differs from the manifest.")

    history_start = date.fromisoformat(manifest["history_start"])
    rederived: list[pd.DataFrame] = []
    for status in manifest["statuses"]:
        if status["status"] != "success":
            continue
        ticker = status["ticker"]
        raw = (path / f"raw/{ticker}.xlsx").read_bytes()
        rows, _ = nav_rows_from_payload(
            raw, ticker, history_start, datetime.fromisoformat(status["retrieved_at"])
        )
        rederived.append(rows)
    expected = (
        pd.concat(rederived, ignore_index=True)
        .sort_values(["ticker", "date"], kind="mergesort")
        .reset_index(drop=True)
        if rederived
        else empty_nav_frame()
    )
    try:
        pd.testing.assert_frame_equal(frame, expected, check_exact=True)
    except AssertionError as error:
        raise NavVintageError(
            "nav.parquet differs from rows re-derived from raw files."
        ) from error
    return NavVintage(path=path, manifest=manifest, frame=frame)
