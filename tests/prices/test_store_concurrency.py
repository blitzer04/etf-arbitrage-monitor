"""Cross-process serialization and lock-timeout tests."""

from __future__ import annotations

import errno
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from etf_monitor.prices.store import StorePaths, persist_batch
from tests.prices.helpers import (
    DEFAULT_RETRIEVED_AT,
    fictional_payload,
    make_batch,
)


def _persist_in_child(batch: object, paths: object) -> bool:
    return persist_batch(batch, paths).persisted  # type: ignore[arg-type]


def test_concurrent_writers_serialize_and_preserve_both_batches(
    tmp_path: Path,
) -> None:
    paths = StorePaths(
        tmp_path / "prices.parquet",
        tmp_path / "snapshots",
        tmp_path / "acquisitions",
    )
    spy_batch = make_batch(
        {"SPY": fictional_payload(ticker="SPY")},
        batch_instant=DEFAULT_RETRIEVED_AT - timedelta(microseconds=2),
    )
    qqq_batch = make_batch(
        {"QQQ": fictional_payload(ticker="QQQ", close=200.0)},
        batch_instant=DEFAULT_RETRIEVED_AT - timedelta(microseconds=1),
    )

    with ProcessPoolExecutor(max_workers=2) as workers:
        results = list(
            workers.map(
                _persist_in_child,
                (spy_batch, qqq_batch),
                (paths, paths),
            )
        )
    loaded = pd.read_parquet(paths.canonical)

    assert results == [True, True]
    assert set(loaded["ticker"]) == {"SPY", "QQQ"}
    assert len(loaded) == 2

    serial_paths = StorePaths(
        tmp_path / "serial.parquet",
        tmp_path / "serial-snapshots",
        tmp_path / "serial-acquisitions",
    )
    persist_batch(spy_batch, serial_paths)
    persist_batch(qqq_batch, serial_paths)
    serial = pd.read_parquet(serial_paths.canonical)
    pd.testing.assert_frame_equal(
        loaded.sort_values(["ticker", "date"]).reset_index(drop=True),
        serial.sort_values(["ticker", "date"]).reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
    )


def test_lock_timeout_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = StorePaths(
        tmp_path / "prices.parquet",
        tmp_path / "snapshots",
        tmp_path / "acquisitions",
    )
    batch = make_batch({"SPY": fictional_payload()})
    monkeypatch.setattr("etf_monitor.prices.store._LOCK_TIMEOUT_SECONDS", 0.01)

    def blocked(lock_file: object) -> None:
        raise OSError(errno.EAGAIN, "held by another process")

    monkeypatch.setattr("etf_monitor.prices.store._try_lock", blocked)
    with pytest.raises(TimeoutError, match="Timed out"):
        persist_batch(batch, paths)
