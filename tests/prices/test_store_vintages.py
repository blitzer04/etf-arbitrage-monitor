"""Vintage ordering, snapshot, field-loss, and atomic failure tests."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from etf_monitor.prices.store import (
    FieldLossError,
    SameVintageConflictError,
    StaleVintageError,
    StoreError,
    StorePaths,
    persist_batch,
)
from tests.prices.helpers import (
    DEFAULT_RETRIEVED_AT,
    fictional_payload,
    make_batch,
)


def _paths(root: Path) -> StorePaths:
    return StorePaths(
        root / "prices.parquet", root / "snapshots", root / "acquisitions"
    )


def _initial(paths: StorePaths) -> bytes:
    persist_batch(
        make_batch(
            {"SPY": fictional_payload()},
            batch_instant=DEFAULT_RETRIEVED_AT - timedelta(microseconds=2),
        ),
        paths,
    )
    return paths.canonical.read_bytes()


def test_identical_later_vintage_advances_watermark_without_snapshot(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _initial(paths)
    later = datetime(2024, 7, 11, 16, 0, tzinfo=UTC)

    result = persist_batch(
        make_batch({"SPY": fictional_payload()}, retrieved_at=later), paths
    )
    loaded = pd.read_parquet(paths.canonical)

    assert result.persisted
    assert result.snapshot_path is None
    assert loaded["retrieved_at"].iloc[0] == pd.Timestamp(later)
    assert not paths.snapshots.exists()


def test_revision_snapshots_exact_old_bytes_before_replace(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    old_bytes = _initial(paths)
    later = datetime(2024, 7, 11, 16, 0, tzinfo=UTC)
    changed = make_batch(
        {"SPY": fictional_payload(adjusted_close=100.75)},
        retrieved_at=later,
    )

    result = persist_batch(changed, paths)

    assert result.snapshot_path is not None
    assert result.snapshot_path.name == f"prices_before_{changed.batch_id}.parquet"
    assert result.snapshot_path.read_bytes() == old_bytes
    assert result.snapshot_sha256 == hashlib.sha256(old_bytes).hexdigest()
    assert result.revision_counts["SPY"] == 1
    record = json.loads(result.record_path.read_text())
    assert record["revision_counts"]["SPY"] == 1
    assert record["snapshot_path"] == str(result.snapshot_path)
    assert record["snapshot_sha256"] == result.snapshot_sha256


@pytest.mark.parametrize(
    ("incoming_time", "error_type"),
    [
        (DEFAULT_RETRIEVED_AT, SameVintageConflictError),
        (datetime(2024, 7, 9, 16, 0, tzinfo=UTC), StaleVintageError),
    ],
)
def test_changed_same_or_older_vintage_is_rejected_without_canonical_change(
    tmp_path: Path,
    incoming_time: datetime,
    error_type: type[Exception],
) -> None:
    paths = _paths(tmp_path)
    _initial(paths)
    before = paths.canonical.read_bytes()
    changed = make_batch(
        {"SPY": fictional_payload(adjusted_close=100.75)},
        retrieved_at=incoming_time,
        batch_instant=incoming_time - timedelta(microseconds=1),
    )

    with pytest.raises(error_type):
        persist_batch(changed, paths)

    assert paths.canonical.read_bytes() == before
    assert not paths.snapshots.exists()


def test_field_disappearance_is_rejected_without_mixing_vintages(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _initial(paths)
    before = paths.canonical.read_bytes()
    changed = make_batch(
        {"SPY": fictional_payload(adjusted_close=None)},
        retrieved_at=datetime(2024, 7, 11, 16, tzinfo=UTC),
    )

    with pytest.raises(FieldLossError):
        persist_batch(changed, paths)

    assert paths.canonical.read_bytes() == before


def test_reversed_arrival_order_rejects_older_change_and_keeps_identical_values(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    old_time = DEFAULT_RETRIEVED_AT
    new_time = datetime(2024, 7, 11, 16, tzinfo=UTC)
    persist_batch(
        make_batch(
            {"SPY": fictional_payload(adjusted_close=100.75)},
            retrieved_at=new_time,
        ),
        paths,
    )
    latest_bytes = paths.canonical.read_bytes()

    with pytest.raises(StaleVintageError):
        persist_batch(
            make_batch(
                {"SPY": fictional_payload(adjusted_close=100.5)},
                retrieved_at=old_time,
                batch_instant=old_time - timedelta(microseconds=2),
            ),
            paths,
        )
    persist_batch(
        make_batch(
            {"SPY": fictional_payload(adjusted_close=100.75)},
            retrieved_at=old_time,
            batch_instant=old_time - timedelta(microseconds=1),
        ),
        paths,
    )

    loaded = pd.read_parquet(paths.canonical)
    assert loaded["adjusted_close"].iloc[0] == 100.75
    assert loaded["retrieved_at"].iloc[0] == pd.Timestamp(new_time)
    assert paths.canonical.read_bytes() != b""
    assert latest_bytes != b""


def test_failure_during_canonical_replace_preserves_bytes_after_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    old_bytes = _initial(paths)
    batch = make_batch(
        {"SPY": fictional_payload(adjusted_close=100.75)},
        retrieved_at=datetime(2024, 7, 11, 16, tzinfo=UTC),
    )
    original_replace = os.replace

    def fail_canonical_replace(source: object, destination: object) -> None:
        if Path(str(destination)) == paths.canonical:
            raise OSError("injected canonical replace failure")
        original_replace(Path(str(source)), Path(str(destination)))

    monkeypatch.setattr("etf_monitor.prices.store.os.replace", fail_canonical_replace)

    with pytest.raises(OSError, match="injected"):
        persist_batch(batch, paths)

    assert paths.canonical.read_bytes() == old_bytes
    snapshots = list(paths.snapshots.glob("*.parquet"))
    assert len(snapshots) == 1
    assert (
        hashlib.sha256(snapshots[0].read_bytes()).hexdigest()
        == hashlib.sha256(old_bytes).hexdigest()
    )


def test_snapshot_failure_aborts_before_canonical_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    old_bytes = _initial(paths)
    batch = make_batch(
        {"SPY": fictional_payload(adjusted_close=100.75)},
        retrieved_at=datetime(2024, 7, 11, 16, tzinfo=UTC),
    )

    def fail_snapshot(
        canonical: Path, snapshot_dir: Path, batch_id: str
    ) -> tuple[Path, str]:
        raise OSError("snapshot failure")

    monkeypatch.setattr("etf_monitor.prices.store._snapshot_existing", fail_snapshot)

    with pytest.raises(OSError, match="snapshot failure"):
        persist_batch(batch, paths)

    assert paths.canonical.read_bytes() == old_bytes


def test_existing_snapshot_for_batch_id_is_not_overwritten(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    old_bytes = _initial(paths)
    batch = make_batch(
        {"SPY": fictional_payload(adjusted_close=100.75)},
        retrieved_at=datetime(2024, 7, 11, 16, tzinfo=UTC),
    )
    paths.snapshots.mkdir()
    snapshot = paths.snapshots / f"prices_before_{batch.batch_id}.parquet"
    snapshot.write_bytes(b"existing snapshot")

    with pytest.raises(StoreError, match="Snapshot already exists"):
        persist_batch(batch, paths)

    assert paths.canonical.read_bytes() == old_bytes
    assert snapshot.read_bytes() == b"existing snapshot"


def test_store_rejects_mixing_source_vintages(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _initial(paths)
    incoming = make_batch(
        {"QQQ": fictional_payload(ticker="QQQ")},
        batch_instant=DEFAULT_RETRIEVED_AT - timedelta(microseconds=1),
    )
    incoming.rows["source"] = pd.Series(
        ["FICTIONAL_synthetic"] * len(incoming.rows),
        index=incoming.rows.index,
        dtype="string",
    )
    before = paths.canonical.read_bytes()

    with pytest.raises(StoreError, match="source differs"):
        persist_batch(incoming, paths)

    assert paths.canonical.read_bytes() == before
