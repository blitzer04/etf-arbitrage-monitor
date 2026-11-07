"""Exact Parquet round-trip tests for canonical price persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from etf_monitor.prices.schema import canonicalize_frame
from etf_monitor.prices.store import StoreError, StorePaths, persist_batch
from tests.prices.helpers import canonical_frame, fictional_payload, make_batch


def test_parquet_round_trip_preserves_nulls_and_maximum_int64_volume(
    tmp_path: Path,
) -> None:
    batch = make_batch(
        {
            "SPY": fictional_payload(
                volume=2**63 - 1,
                adjusted_close=None,
            )
        }
    )
    paths = StorePaths(
        tmp_path / "prices.parquet",
        tmp_path / "snapshots",
        tmp_path / "acquisitions",
    )

    result = persist_batch(batch, paths)
    loaded = canonicalize_frame(pd.read_parquet(paths.canonical))
    record = json.loads(result.record_path.read_text())

    assert result.persisted
    assert loaded["volume"].iloc[0] == 2**63 - 1
    assert pd.isna(loaded["adjusted_close"].iloc[0])
    assert loaded["volume"].dtype == pd.Int64Dtype()
    assert loaded["open"].dtype == "float64"
    assert record["batch_id"] == batch.batch_id
    assert record["persisted"] is True
    assert record["canonical_sha256"] == result.canonical_sha256
    assert len(record["statuses"]) == 24
    assert len(record["revision_counts"]) == 24
    assert record["statuses"][0]["detail"] is None
    pd.testing.assert_frame_equal(
        canonicalize_frame(batch.rows),
        loaded,
        check_dtype=True,
        check_exact=True,
    )


def test_all_failed_batch_records_status_but_does_not_write_canonical(
    tmp_path: Path,
) -> None:
    paths = StorePaths(
        tmp_path / "prices.parquet",
        tmp_path / "snapshots",
        tmp_path / "acquisitions",
    )
    batch = make_batch({"SPY": OSError("fictional failure")})

    result = persist_batch(batch, paths)

    assert not result.persisted
    assert not paths.canonical.exists()
    assert result.record_path.exists()
    record = json.loads(result.record_path.read_text())
    assert record["persisted"] is False
    assert record["revision_counts"]["SPY"] == 0
    assert record["statuses"][0]["detail"] == "fictional failure"


def test_parquet_without_pandas_metadata_preserves_large_and_null_volumes(
    tmp_path: Path,
) -> None:
    paths = StorePaths(
        tmp_path / "prices.parquet",
        tmp_path / "snapshots",
        tmp_path / "acquisitions",
    )
    existing = pd.concat(
        [
            canonical_frame(volume=2**53 + 1),
            canonical_frame(ticker="QQQ", day="2024-07-03", volume=None),
        ],
        ignore_index=True,
    )
    table = pa.Table.from_pandas(existing, preserve_index=False)
    pq.write_table(table.replace_schema_metadata(None), paths.canonical)

    batch = make_batch({"KRE": fictional_payload(ticker="KRE")})
    persist_batch(batch, paths)
    loaded = canonicalize_frame(pq.read_table(paths.canonical).to_pandas())

    volumes = loaded.set_index("ticker")["volume"]
    assert volumes.loc["SPY"] == 2**53 + 1
    assert pd.isna(volumes.loc["QQQ"])
    assert loaded["volume"].dtype == pd.Int64Dtype()


@pytest.mark.parametrize("invalid_schema", ["float_volume", "extra_column"])
def test_existing_parquet_with_wrong_arrow_schema_is_rejected(
    tmp_path: Path, invalid_schema: str
) -> None:
    paths = StorePaths(
        tmp_path / "prices.parquet",
        tmp_path / "snapshots",
        tmp_path / "acquisitions",
    )
    frame = canonical_frame()
    if invalid_schema == "float_volume":
        frame["volume"] = pd.Series([1.0], dtype="float64")
    else:
        frame["unexpected"] = 1
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), paths.canonical)

    with pytest.raises(StoreError, match="schema"):
        persist_batch(make_batch({"KRE": fictional_payload(ticker="KRE")}), paths)


def test_store_paths_are_relative_only_to_explicit_data_root(tmp_path: Path) -> None:
    paths = StorePaths.under(tmp_path)

    assert paths.canonical == tmp_path / "processed/etf_prices_daily.parquet"
    assert paths.snapshots == tmp_path / "snapshots/prices"
    assert paths.acquisitions == tmp_path / "processed/acquisitions"


def test_existing_acquisition_record_is_not_overwritten(tmp_path: Path) -> None:
    paths = StorePaths(
        tmp_path / "prices.parquet",
        tmp_path / "snapshots",
        tmp_path / "acquisitions",
    )
    batch = make_batch({"SPY": fictional_payload()})
    paths.acquisitions.mkdir()
    record_path = paths.acquisitions / f"{batch.batch_id}.json"
    record_path.write_bytes(b"existing record")

    with pytest.raises(StoreError, match="record already exists"):
        persist_batch(batch, paths)

    assert record_path.read_bytes() == b"existing record"
    assert not paths.canonical.exists()
    assert not paths.snapshots.exists()
