"""Locked, atomic persistence for canonical price batches and source vintages."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from etf_monitor.prices.acquire import BatchResult, TickerStatus
from etf_monitor.prices.schema import (
    CANONICAL_COLUMNS,
    MARKET_COLUMNS,
    PRICE_COLUMNS,
    canonicalize_frame,
    empty_canonical_frame,
    validate_canonical,
)
from etf_monitor.universe import Universe, load_universe

_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_POLL_SECONDS = 0.05
_PARQUET_SCHEMA = pa.schema(
    [
        ("ticker", pa.string()),
        ("date", pa.timestamp("ns")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("adjusted_close", pa.float64()),
        ("volume", pa.int64()),
        ("retrieved_at", pa.timestamp("ns", tz="UTC")),
        ("source", pa.string()),
    ]
)


class StoreError(RuntimeError):
    """Base class for a rejected price persistence transaction."""


class VanishedObservationError(StoreError):
    """Indicate a successful request omitted a stored in-window observation."""


class FieldLossError(StoreError):
    """Indicate an incoming vintage lost a previously present market value."""


class SameVintageConflictError(StoreError):
    """Indicate different market values claim the same retrieval vintage."""


class StaleVintageError(StoreError):
    """Indicate an older retrieval attempted to replace newer market values."""


@dataclass(frozen=True, slots=True)
class StorePaths:
    """Locations for one canonical parquet, its snapshots, and batch records."""

    canonical: Path
    snapshots: Path
    acquisitions: Path

    @classmethod
    def under(cls, root: Path) -> StorePaths:
        """Build the standard store paths under an explicit data root."""
        return cls(
            canonical=root / "processed/etf_prices_daily.parquet",
            snapshots=root / "snapshots/prices",
            acquisitions=root / "processed/acquisitions",
        )


@dataclass(frozen=True, slots=True)
class PersistResult:
    """Describe the canonical write, revisions, and recorded provenance."""

    persisted: bool
    record_path: Path
    revision_counts: dict[str, int]
    snapshot_path: Path | None
    snapshot_sha256: str | None
    canonical_sha256: str | None


def _lock_path(canonical: Path) -> Path:
    resolved = canonical.resolve(strict=False)
    return resolved.with_name(f".{resolved.name}.lock")


def _try_lock(lock_file: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(lock_file: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _transaction_lock(canonical: Path) -> Iterator[None]:
    """Hold an adjacent cross-process lock for the complete store operation."""
    lock_path = _lock_path(canonical)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    with lock_path.open("a+b") as lock_file:
        if sys.platform == "win32":
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
        while True:
            try:
                _try_lock(lock_file)
                break
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out after {_LOCK_TIMEOUT_SECONDS:g} seconds waiting "
                        f"for {lock_path}."
                    ) from error
                time.sleep(min(_LOCK_POLL_SECONDS, remaining))
        try:
            yield
        finally:
            _unlock(lock_file)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing(value: object) -> bool:
    missing = pd.isna(cast(Any, value))
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _same_market_value(left: object, right: object) -> bool:
    left_missing = _is_missing(left)
    right_missing = _is_missing(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    return bool(left == right)


def _record_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return empty_canonical_frame()
    frame = pd.DataFrame(
        {
            "ticker": pd.Series(
                [record["ticker"] for record in records], dtype="string"
            ),
            "date": pd.to_datetime(cast(Any, [record["date"] for record in records])),
            **{
                column: pd.Series(
                    [record[column] for record in records], dtype="float64"
                )
                for column in PRICE_COLUMNS
            },
            "volume": pd.Series(
                pd.array(
                    cast(Any, [record["volume"] for record in records]),
                    dtype="Int64",
                )
            ),
            "retrieved_at": pd.Series(
                cast(Any, [record["retrieved_at"] for record in records]),
                dtype="datetime64[ns, UTC]",
            ),
            "source": pd.Series(
                [record["source"] for record in records], dtype="string"
            ),
        },
        columns=CANONICAL_COLUMNS,
    )
    return canonicalize_frame(frame)


def _validate_batch(batch: BatchResult, universe: Universe) -> tuple[TickerStatus, ...]:
    universe_tickers = tuple(etf.ticker for etf in universe.etfs)
    if tuple(status.ticker for status in batch.statuses) != universe_tickers:
        raise StoreError("Batch statuses must match the configured universe order.")
    if (
        len(batch.batch_id) != 22
        or batch.batch_id[8] != "T"
        or batch.batch_id[-1] != "Z"
        or not batch.batch_id[:8].isdigit()
        or not batch.batch_id[9:-1].isdigit()
    ):
        raise StoreError("batch_id must be a UTC microsecond identifier.")
    if batch.request_start >= batch.request_end_exclusive:
        raise StoreError("Batch request window must be non-empty.")
    for status in batch.statuses:
        if (
            status.request_start != batch.request_start
            or status.request_end_exclusive != batch.request_end_exclusive
        ):
            raise StoreError(f"Request bounds do not match for {status.ticker}.")
        if status.status not in {"success", "empty", "failed"}:
            raise StoreError(f"Unsupported status for {status.ticker}.")
        if status.returned_dates != tuple(sorted(set(status.returned_dates))):
            raise StoreError(
                f"Returned dates are not unique/sorted for {status.ticker}."
            )
        if any(
            value < batch.request_start or value >= batch.request_end_exclusive
            for value in status.returned_dates
        ):
            raise StoreError(
                f"Returned date is outside the request for {status.ticker}."
            )
        if status.status == "success":
            if status.reason is not None:
                raise StoreError(f"Successful status has a reason for {status.ticker}.")
            if status.row_count != len(status.returned_dates):
                raise StoreError(f"Row count disagrees for {status.ticker}.")
            if status.first_date != (
                status.returned_dates[0] if status.returned_dates else None
            ) or status.last_date != (
                status.returned_dates[-1] if status.returned_dates else None
            ):
                raise StoreError(f"Date bounds disagree for {status.ticker}.")
            if status.retrieved_at is None or status.retrieved_at.tz is None:
                raise StoreError(
                    f"Successful status lacks retrieval time for {status.ticker}."
                )
            if str(status.retrieved_at.tz) != "UTC":
                raise StoreError(f"Retrieval time must be UTC for {status.ticker}.")
        elif status.status == "empty":
            if (
                status.reason != "empty_response"
                or status.row_count
                or status.returned_dates
            ):
                raise StoreError(f"Empty status is inconsistent for {status.ticker}.")
        elif status.row_count or status.returned_dates or not status.reason:
            raise StoreError(f"Failed status is inconsistent for {status.ticker}.")

    canonical = canonicalize_frame(batch.rows)
    validate_canonical(canonical, universe)
    successful_tickers = {
        status.ticker for status in batch.statuses if status.status == "success"
    }
    row_tickers = set(canonical["ticker"].array)
    if not row_tickers.issubset(successful_tickers):
        raise StoreError("Batch rows include a non-successful ticker.")
    for status in batch.statuses:
        ticker_rows = canonical.loc[canonical["ticker"].eq(status.ticker)]
        actual_dates = tuple(timestamp.date() for timestamp in ticker_rows["date"])
        if status.status != "success" and not ticker_rows.empty:
            raise StoreError(
                f"Non-successful ticker has canonical rows: {status.ticker}."
            )
        if status.status == "success":
            if (
                actual_dates != status.returned_dates
                or len(ticker_rows) != status.row_count
            ):
                raise StoreError(f"Status rows disagree for {status.ticker}.")
            if status.retrieved_at is None:
                raise StoreError(f"Missing retrieval time for {status.ticker}.")
            if not ticker_rows["retrieved_at"].eq(status.retrieved_at).all():
                raise StoreError(f"Retrieval time disagrees for {status.ticker}.")
    return tuple(batch.statuses)


def _vanished_keys(
    existing: pd.DataFrame, statuses: tuple[TickerStatus, ...]
) -> list[tuple[str, str]]:
    vanished: list[tuple[str, str]] = []
    for status in statuses:
        if status.status != "success":
            continue
        in_window = (
            existing["ticker"].eq(status.ticker)
            & existing["date"].ge(pd.Timestamp(status.request_start))
            & existing["date"].lt(pd.Timestamp(status.request_end_exclusive))
        )
        for timestamp in existing.loc[in_window, "date"].sort_values():
            if timestamp.date() not in status.returned_dates:
                vanished.append((status.ticker, timestamp.date().isoformat()))
    return vanished


def _merge_vintages(
    existing: pd.DataFrame, incoming: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    existing_records = cast(list[dict[str, object]], existing.to_dict(orient="records"))
    current = {
        (str(record["ticker"]), cast(pd.Timestamp, record["date"])): record
        for record in existing_records
    }
    revisions: dict[str, int] = {}
    incoming_records = cast(list[dict[str, object]], incoming.to_dict(orient="records"))
    for new_record in incoming_records:
        key = (str(new_record["ticker"]), cast(pd.Timestamp, new_record["date"]))
        old_record = current.get(key)
        if old_record is None:
            current[key] = new_record
            continue

        loses_value = any(
            not _is_missing(old_record[column]) and _is_missing(new_record[column])
            for column in MARKET_COLUMNS
        )
        if loses_value:
            raise FieldLossError(f"Incoming row loses a market field for {key}.")
        identical = all(
            _same_market_value(old_record[column], new_record[column])
            for column in MARKET_COLUMNS
        )
        if identical:
            new_retrieved_at = cast(pd.Timestamp, new_record["retrieved_at"])
            old_retrieved_at = cast(pd.Timestamp, old_record["retrieved_at"])
            if new_retrieved_at > old_retrieved_at:
                old_record["retrieved_at"] = new_record["retrieved_at"]
            continue
        new_retrieved_at = cast(pd.Timestamp, new_record["retrieved_at"])
        old_retrieved_at = cast(pd.Timestamp, old_record["retrieved_at"])
        if new_retrieved_at < old_retrieved_at:
            raise StaleVintageError(f"Incoming vintage is stale for {key}.")
        if new_retrieved_at == old_retrieved_at:
            raise SameVintageConflictError(
                f"Different market values claim the same vintage for {key}."
            )
        current[key] = new_record
        ticker = str(new_record["ticker"])
        revisions[ticker] = revisions.get(ticker, 0) + 1

    ordered = [current[key] for key in sorted(current)]
    return _record_frame(ordered), revisions


def _fsync_directory(path: Path) -> None:
    if sys.platform == "win32":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pandas_type_mapper(arrow_type: pa.DataType) -> pd.Int64Dtype | None:
    if pa.types.is_int64(arrow_type):
        return pd.Int64Dtype()
    return None


def _read_canonical_parquet(path: Path) -> pd.DataFrame:
    table = pq.read_table(path)
    if not table.schema.equals(_PARQUET_SCHEMA, check_metadata=False):
        raise StoreError(
            "Canonical Parquet schema does not match the approved Arrow schema."
        )
    return canonicalize_frame(table.to_pandas(types_mapper=_pandas_type_mapper))


def _atomic_parquet_write(
    frame: pd.DataFrame, destination: Path, universe: Universe
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.stem}.",
            suffix=".tmp.parquet",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        table = pa.Table.from_pandas(
            frame,
            schema=_PARQUET_SCHEMA,
            preserve_index=False,
            safe=True,
        )
        pq.write_table(table, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        _assert_parquet_roundtrip(temporary, frame, universe)
        os.replace(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
        _assert_parquet_roundtrip(destination, frame, universe)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _assert_parquet_roundtrip(
    path: Path, expected: pd.DataFrame, universe: Universe
) -> None:
    loaded = _read_canonical_parquet(path)
    validate_canonical(loaded, universe)
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_exact=True,
        check_dtype=True,
    )


def _snapshot_existing(
    canonical: Path, snapshot_dir: Path, batch_id: str
) -> tuple[Path, str]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    destination = snapshot_dir / f"prices_before_{batch_id}.parquet"
    if destination.exists():
        raise StoreError(f"Snapshot already exists for batch {batch_id}.")
    original_sha256 = _sha256_file(canonical)
    try:
        with destination.open("xb") as output, canonical.open("rb") as source:
            shutil.copyfileobj(source, output)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as error:
        raise StoreError(f"Snapshot already exists for batch {batch_id}.") from error
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    snapshot_sha256 = _sha256_file(destination)
    if snapshot_sha256 != original_sha256:
        destination.unlink(missing_ok=True)
        raise OSError("Snapshot SHA-256 does not match the existing canonical file.")
    _fsync_directory(snapshot_dir)
    return destination, snapshot_sha256


def _status_record(status: TickerStatus) -> dict[str, object]:
    return {
        "ticker": status.ticker,
        "status": status.status,
        "reason": status.reason,
        "detail": status.detail,
        "request_start": status.request_start.isoformat(),
        "request_end_exclusive": status.request_end_exclusive.isoformat(),
        "retrieved_at": (
            status.retrieved_at.isoformat() if status.retrieved_at is not None else None
        ),
        "returned_dates": [value.isoformat() for value in status.returned_dates],
        "row_count": status.row_count,
        "first_date": status.first_date.isoformat() if status.first_date else None,
        "last_date": status.last_date.isoformat() if status.last_date else None,
        "null_row_dates": [value.isoformat() for value in status.null_row_dates],
        "adjclose_block_absent": status.adjclose_block_absent,
        "trailing_missing_sessions": [
            value.isoformat() for value in status.trailing_missing_sessions
        ],
        "raw_sha256": status.raw_sha256,
    }


def _atomic_json_write(content: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        content, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def persist_batch(
    batch: BatchResult, store_paths: StorePaths, universe: Universe | None = None
) -> PersistResult:
    """Persist one validated batch as a locked latest-vintage transaction.

    Successful rows are merged without mixing source vintages. Missing stored
    dates inside a successful request window, stale changes, same-vintage
    conflicts, or field loss abort before the canonical file is replaced.

    Args:
        batch: Structured result returned by ``fetch_price_batch``.
        store_paths: Canonical, snapshot, and acquisition-record destinations.
        universe: Universe the batch was fetched for; defaults to the
            packaged crowding universe.

    Returns:
        Persistence state and hashes for the batch record.

    Raises:
        StoreError: If the batch is inconsistent or a vintage transaction is
            rejected.
        TimeoutError: If the adjacent cross-process lock is unavailable for
            30 seconds.
    """
    if universe is None:
        universe = load_universe()
    statuses = _validate_batch(batch, universe)
    record_path = store_paths.acquisitions / f"{batch.batch_id}.json"
    if record_path.exists():
        raise StoreError(
            f"Acquisition record already exists for batch {batch.batch_id}."
        )
    with _transaction_lock(store_paths.canonical):
        if record_path.exists():
            raise StoreError(
                f"Acquisition record already exists for batch {batch.batch_id}."
            )
        snapshot_path: Path | None = None
        snapshot_sha256: str | None = None
        canonical_sha256: str | None = None
        revision_counts: dict[str, int] = {}
        persisted = False
        failure: str | None = None
        try:
            successful = [status for status in statuses if status.status == "success"]
            incoming = canonicalize_frame(batch.rows)
            if not successful:
                if store_paths.canonical.exists():
                    canonical_sha256 = _sha256_file(store_paths.canonical)
            else:
                if store_paths.canonical.exists():
                    existing = _read_canonical_parquet(store_paths.canonical)
                    validate_canonical(existing, universe)
                else:
                    existing = empty_canonical_frame()

                if not incoming.empty and not existing.empty:
                    existing_sources = set(existing["source"].array)
                    incoming_sources = set(incoming["source"].array)
                    if existing_sources != incoming_sources:
                        raise StoreError(
                            "Incoming source differs from the existing price store."
                        )
                vanished = _vanished_keys(existing, statuses)
                if vanished:
                    raise VanishedObservationError(
                        "Stored observations vanished inside successful request "
                        f"coverage: {vanished}."
                    )
                if incoming.empty:
                    if store_paths.canonical.exists():
                        canonical_sha256 = _sha256_file(store_paths.canonical)
                else:
                    combined, revision_counts = _merge_vintages(existing, incoming)
                    validate_canonical(combined, universe)
                    if any(revision_counts.values()) and store_paths.canonical.exists():
                        snapshot_path, snapshot_sha256 = _snapshot_existing(
                            store_paths.canonical, store_paths.snapshots, batch.batch_id
                        )
                    _atomic_parquet_write(combined, store_paths.canonical, universe)
                    canonical_sha256 = _sha256_file(store_paths.canonical)
                    persisted = True
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            recorded_revision_counts = {
                status.ticker: revision_counts.get(status.ticker, 0)
                for status in statuses
            }
            record = {
                "batch_id": batch.batch_id,
                "request_start": batch.request_start.isoformat(),
                "request_end_exclusive": batch.request_end_exclusive.isoformat(),
                "statuses": [_status_record(status) for status in statuses],
                "persisted": False,
                "revision_counts": recorded_revision_counts,
                "snapshot_path": str(snapshot_path) if snapshot_path else None,
                "snapshot_sha256": snapshot_sha256,
                "canonical_sha256": canonical_sha256,
                "failure": failure,
            }
            _atomic_json_write(record, record_path)
            raise

        recorded_revision_counts = {
            status.ticker: revision_counts.get(status.ticker, 0) for status in statuses
        }
        record = {
            "batch_id": batch.batch_id,
            "request_start": batch.request_start.isoformat(),
            "request_end_exclusive": batch.request_end_exclusive.isoformat(),
            "statuses": [_status_record(status) for status in statuses],
            "persisted": persisted,
            "revision_counts": recorded_revision_counts,
            "snapshot_path": str(snapshot_path) if snapshot_path else None,
            "snapshot_sha256": snapshot_sha256,
            "canonical_sha256": canonical_sha256,
            "failure": failure,
        }
        _atomic_json_write(record, record_path)
        return PersistResult(
            persisted=persisted,
            record_path=record_path,
            revision_counts=recorded_revision_counts,
            snapshot_path=snapshot_path,
            snapshot_sha256=snapshot_sha256,
            canonical_sha256=canonical_sha256,
        )
