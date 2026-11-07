"""Coverage, truncation, and partial-batch transaction tests."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from etf_monitor.prices.store import (
    StorePaths,
    VanishedObservationError,
    persist_batch,
)
from tests.fixtures.yahoo_chart import build_yahoo_chart_payload
from tests.prices.helpers import (
    DEFAULT_RETRIEVED_AT,
    fictional_payload,
    make_batch,
)


def _paths(root: Path) -> StorePaths:
    return StorePaths(
        root / "prices.parquet", root / "snapshots", root / "acquisitions"
    )


def test_successful_truncation_keeps_only_observed_rows(tmp_path: Path) -> None:
    batch = make_batch({"SPY": fictional_payload(dates=("2024-07-01", "2024-07-02"))})
    paths = _paths(tmp_path)

    persist_batch(batch, paths)
    loaded = pd.read_parquet(paths.canonical)

    assert len(loaded) == 2
    assert set(loaded["date"].dt.strftime("%Y-%m-%d")) == {
        "2024-07-01",
        "2024-07-02",
    }


def test_vanished_stored_observation_rejects_transaction_without_snapshot(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    persist_batch(
        make_batch(
            {"SPY": fictional_payload()},
            batch_instant=DEFAULT_RETRIEVED_AT - timedelta(microseconds=2),
        ),
        paths,
    )
    canonical_before = paths.canonical.read_bytes()
    later = make_batch(
        {"SPY": fictional_payload(dates=("2024-07-03",))},
        batch_instant=DEFAULT_RETRIEVED_AT - timedelta(microseconds=1),
    )

    with pytest.raises(VanishedObservationError, match="vanished"):
        persist_batch(later, paths)

    assert paths.canonical.read_bytes() == canonical_before
    assert not paths.snapshots.exists()


def test_all_null_success_cannot_hide_vanished_stored_observation(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    persist_batch(
        make_batch(
            {"SPY": fictional_payload()},
            batch_instant=DEFAULT_RETRIEVED_AT - timedelta(microseconds=2),
        ),
        paths,
    )
    canonical_before = paths.canonical.read_bytes()
    null_payload = json.dumps(
        build_yahoo_chart_payload(
            dates=("2024-07-02",),
            null_quote_positions={
                "open": (0,),
                "high": (0,),
                "low": (0,),
                "close": (0,),
                "volume": (0,),
            },
            null_adjusted_positions=(0,),
        )
    ).encode("utf-8")

    with pytest.raises(VanishedObservationError):
        persist_batch(
            make_batch(
                {"SPY": null_payload},
                batch_instant=DEFAULT_RETRIEVED_AT - timedelta(microseconds=1),
            ),
            paths,
        )

    assert paths.canonical.read_bytes() == canonical_before


def test_partial_batch_persists_success_and_preserves_failed_ticker_history(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    first = make_batch(
        {
            "SPY": fictional_payload(),
            "QQQ": fictional_payload(ticker="QQQ", close=200.0),
        }
    )
    persist_batch(first, paths)
    before = pd.read_parquet(paths.canonical).set_index("ticker")
    first_retrieved_at = first.statuses[0].retrieved_at
    assert first_retrieved_at is not None
    partial = make_batch(
        {
            "SPY": fictional_payload(close=102.0),
            "QQQ": OSError("fictional isolated failure"),
        },
        retrieved_at=first_retrieved_at.to_pydatetime().replace(day=11),
    )

    result = persist_batch(partial, paths)
    after = pd.read_parquet(paths.canonical).set_index("ticker")

    assert result.persisted
    assert after.loc["QQQ", "close"] == before.loc["QQQ", "close"]
    assert after.loc["QQQ", "retrieved_at"] == before.loc["QQQ", "retrieved_at"]
    assert after.loc["SPY", "close"] == 102.0


def test_failed_and_empty_tickers_do_not_trigger_vanished_check(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    persist_batch(make_batch({"QQQ": fictional_payload(ticker="QQQ")}), paths)
    partial = make_batch(
        {"QQQ": OSError("offline")},
        batch_instant=DEFAULT_RETRIEVED_AT + timedelta(microseconds=1),
    )
    canonical_before = paths.canonical.read_bytes()

    result = persist_batch(partial, paths)

    assert not result.persisted
    assert paths.canonical.exists()
    assert paths.canonical.read_bytes() == canonical_before
