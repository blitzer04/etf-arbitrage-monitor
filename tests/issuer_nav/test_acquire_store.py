"""Offline NAV batch acquisition and immutable vintage tests."""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from etf_monitor.issuer_nav.acquire import NavBatchResult, fetch_nav_batch
from etf_monitor.issuer_nav.sources import NavSource
from etf_monitor.issuer_nav.store import (
    NavVintageError,
    load_nav_vintage,
    write_nav_vintage,
)
from etf_monitor.providers.transport import ProviderTransportError
from tests.fixtures.ssga_nav import build_ssga_workbook
from tests.issuer_nav.helpers import RETRIEVED_AT, MappingNavTransport, small_universe

START = date(2024, 1, 1)
SOURCES = {
    "SPY": NavSource("SPY", "ssga_navhist_xlsx", None),
    "DIA": NavSource("DIA", "ssga_navhist_xlsx", None),
    "QQQ": NavSource("QQQ", None, "FICTIONAL reason: no public file"),
}


def _clock() -> datetime:
    return RETRIEVED_AT


def _batch(responses: dict[str, bytes | Exception]) -> NavBatchResult:
    return fetch_nav_batch(
        small_universe("SPY", "DIA", "QQQ"),
        START,
        MappingNavTransport(responses),
        _clock,
        SOURCES,
    )


def test_batch_records_statuses_and_isolates_failures() -> None:
    transport = MappingNavTransport(
        {
            "SPY": build_ssga_workbook(),
            "DIA": build_ssga_workbook("DIA", workbook_ticker="SPY"),
        }
    )
    batch = fetch_nav_batch(
        small_universe("SPY", "DIA", "QQQ"), START, transport, _clock, SOURCES
    )

    assert transport.calls == ["SPY", "DIA"]
    spy, dia, qqq = batch.statuses
    assert (spy.status, spy.row_count, spy.missing_sessions) == ("success", 2, ())
    assert (dia.status, dia.reason_code) == ("failed", "identity_mismatch")
    assert dia.raw_sha256 is not None
    assert (qqq.status, qqq.reason_code) == ("unavailable", "no_configured_source")
    assert set(batch.frame["ticker"]) == {"SPY"}
    assert set(batch.raw_payloads) == {"SPY", "DIA"}


def test_missing_sessions_are_reported_not_filled() -> None:
    raw = build_ssga_workbook(
        rows=(
            ("08-Jul-2024", 1.0, 10.0, 10.0),
            ("02-Jul-2024", 1.0, 10.0, 10.0),
        )
    )
    batch = _batch({"SPY": raw, "DIA": ProviderTransportError("FICTIONAL")})

    assert batch.statuses[0].missing_sessions == (date(2024, 7, 3), date(2024, 7, 5))
    assert len(batch.frame) == 2
    assert batch.statuses[1].reason_code == "transport"


def test_non_session_and_future_dates_fail_the_ticker() -> None:
    weekend = build_ssga_workbook(rows=(("06-Jul-2024", 1.0, 1.0, 1.0),))
    future = build_ssga_workbook("DIA", rows=(("10-Jul-2024", 1.0, 1.0, 1.0),))

    batch = _batch({"SPY": weekend, "DIA": future})

    assert [status.reason_code for status in batch.statuses[:2]] == [
        "nav_contract",
        "nav_contract",
    ]
    assert batch.frame.empty


def test_naive_clock_is_rejected() -> None:
    with pytest.raises(ValueError):
        fetch_nav_batch(
            small_universe("SPY"),
            START,
            MappingNavTransport({}),
            lambda: datetime(2024, 7, 9),
            SOURCES,
        )


def _written(tmp_path: Path, spy: bytes | None = None) -> Path:
    spy_payload = build_ssga_workbook() if spy is None else spy
    batch = _batch({"SPY": spy_payload, "DIA": build_ssga_workbook("DIA")})
    return write_nav_vintage(batch, tmp_path, small_universe("SPY", "DIA", "QQQ"))


def test_vintage_round_trip_keeps_raw_bytes_and_rows(tmp_path: Path) -> None:
    # openpyxl stamps the save time into the zip, so compare with the exact payload.
    spy = build_ssga_workbook()
    path = _written(tmp_path, spy)
    vintage = load_nav_vintage(path, small_universe("SPY", "DIA", "QQQ"))

    assert (path / "raw/SPY.xlsx").read_bytes() == spy
    assert len(vintage.frame) == 4
    assert vintage.frame["shares_outstanding"].dtype == pd.Int64Dtype()
    statuses = vintage.manifest["statuses"]
    assert isinstance(statuses, list)
    assert statuses[0]["retrieved_at"] == RETRIEVED_AT.isoformat()
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]


def test_existing_vintage_is_never_overwritten(tmp_path: Path) -> None:
    _written(tmp_path)
    with pytest.raises(NavVintageError):
        _written(tmp_path)


def test_wrong_universe_is_rejected(tmp_path: Path) -> None:
    path = _written(tmp_path)
    with pytest.raises(NavVintageError):
        load_nav_vintage(path, small_universe("SPY", "DIA"))


def _make_writable(path: Path) -> None:
    os.chmod(path, 0o644)


def test_altered_raw_file_is_detected(tmp_path: Path) -> None:
    path = _written(tmp_path)
    raw = path / "raw/SPY.xlsx"
    _make_writable(raw)
    raw.write_bytes(raw.read_bytes() + b"x")
    with pytest.raises(NavVintageError, match="Digest"):
        load_nav_vintage(path, small_universe("SPY", "DIA", "QQQ"))


def test_extra_file_is_detected(tmp_path: Path) -> None:
    path = _written(tmp_path)
    (path / "raw/XLK.xlsx").write_bytes(b"x")
    with pytest.raises(NavVintageError, match="differ"):
        load_nav_vintage(path, small_universe("SPY", "DIA", "QQQ"))


def test_rows_that_differ_from_raw_are_detected(tmp_path: Path) -> None:
    path = _written(tmp_path)
    raw = path / "raw/SPY.xlsx"
    replacement = build_ssga_workbook(
        rows=(
            ("03-Jul-2024", 101.5, 2001.0, 203000.0),
            ("02-Jul-2024", 100.0, 1900.0, 190000.0),
        )
    )
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    import hashlib

    manifest["files"]["raw/SPY.xlsx"] = hashlib.sha256(replacement).hexdigest()
    for file in (raw, manifest_path):
        _make_writable(file)
    raw.write_bytes(replacement)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(NavVintageError, match="re-derived"):
        load_nav_vintage(path, small_universe("SPY", "DIA", "QQQ"))


def test_retrieval_timestamp_is_utc(tmp_path: Path) -> None:
    vintage = load_nav_vintage(_written(tmp_path), small_universe("SPY", "DIA", "QQQ"))
    assert vintage.frame["retrieved_at"].iloc[0] == pd.Timestamp(
        datetime(2024, 7, 9, 1, 0, tzinfo=UTC)
    )
