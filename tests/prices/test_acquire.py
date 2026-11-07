"""Sequential acquisition, coverage, and transport-isolation tests."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from etf_monitor.prices.acquire import fetch_price_batch
from etf_monitor.prices.schema import PRICE_SOURCE
from etf_monitor.universe import load_universe
from tests.fixtures.yahoo_chart import build_yahoo_chart_payload
from tests.prices.helpers import (
    DEFAULT_END,
    DEFAULT_RETRIEVED_AT,
    DEFAULT_START,
    MappingTransport,
    fictional_payload,
    make_batch,
)


def test_requests_each_ticker_once_in_universe_order_with_ny_midnight_bounds() -> None:
    transport = MappingTransport()
    batch = fetch_price_batch(
        load_universe(),
        date(2024, 7, 1),
        date(2024, 7, 5),
        transport,
        lambda: DEFAULT_RETRIEVED_AT,
    )
    expected = [etf.ticker for etf in load_universe().etfs]

    assert [call[0] for call in transport.calls] == expected
    assert len(transport.calls) == len(expected)
    zone = ZoneInfo("America/New_York")
    expected_bounds = (
        int(datetime(2024, 7, 1, tzinfo=zone).timestamp()),
        int(datetime(2024, 7, 5, tzinfo=zone).timestamp()),
    )
    assert all(call[1:] == expected_bounds for call in transport.calls)
    assert len(batch.statuses) == len(expected)


def test_retrieval_clock_is_sampled_immediately_after_each_response() -> None:
    transport = MappingTransport()
    clock_call_counts: list[int] = []

    def clock() -> datetime:
        clock_call_counts.append(len(transport.calls))
        return DEFAULT_RETRIEVED_AT + timedelta(seconds=len(clock_call_counts))

    batch = fetch_price_batch(
        load_universe(),
        DEFAULT_START,
        DEFAULT_END,
        transport,
        clock,
    )

    assert clock_call_counts == list(range(25))
    assert batch.batch_id == "20240710T160001000000Z"
    assert batch.statuses[0].retrieved_at == pd.Timestamp(
        DEFAULT_RETRIEVED_AT + timedelta(seconds=2)
    )
    assert batch.statuses[-1].retrieved_at == pd.Timestamp(
        DEFAULT_RETRIEVED_AT + timedelta(seconds=25)
    )


def test_successful_truncated_response_keeps_rows_and_reports_trailing_sessions() -> (
    None
):
    batch = make_batch({"SPY": fictional_payload(dates=("2024-07-01", "2024-07-02"))})
    status = batch.statuses[0]

    assert status.status == "success"
    assert status.returned_dates == (date(2024, 7, 1), date(2024, 7, 2))
    assert status.last_date == date(2024, 7, 2)
    assert status.trailing_missing_sessions == (
        date(2024, 7, 3),
        date(2024, 7, 5),
        date(2024, 7, 8),
        date(2024, 7, 9),
    )
    assert len(batch.rows) == 2
    assert batch.rows["source"].eq(PRICE_SOURCE).all()


def test_success_null_row_and_absent_adjclose_are_recorded_without_fabrication() -> (
    None
):
    payload = fictional_payload(
        dates=("2024-07-02",),
        adjusted_close=None,
        include_adjusted_close=False,
    )
    batch = make_batch({"SPY": payload})
    status = batch.statuses[0]

    assert status.adjclose_block_absent
    assert batch.rows["close"].iloc[0] == 101.0
    assert pd.isna(batch.rows["adjusted_close"].iloc[0])


def test_ohlc_violation_fails_only_the_response_ticker() -> None:
    payload = json.dumps(
        build_yahoo_chart_payload(
            dates=("2024-07-02",),
            quote_values={"high": (98.0,)},
        )
    ).encode("utf-8")
    batch = make_batch(
        {
            "SPY": payload,
            "QQQ": fictional_payload(ticker="QQQ", close=200.0),
        }
    )

    assert batch.statuses[0].status == "failed"
    assert batch.statuses[0].reason == "invalid_observation"
    assert batch.statuses[1].status == "success"
    assert set(batch.rows["ticker"]) == {"QQQ"}


def test_volume_zero_and_null_remain_distinct() -> None:
    zero = make_batch({"SPY": fictional_payload(volume=0)})
    missing = make_batch({"SPY": fictional_payload(volume=None)})

    assert zero.rows["volume"].iloc[0] == 0
    assert pd.isna(missing.rows["volume"].iloc[0])


def test_all_null_observation_is_excluded_from_rows_and_returned_dates() -> None:
    payload = json.dumps(
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
    batch = make_batch({"SPY": payload})

    assert batch.statuses[0].status == "success"
    assert batch.statuses[0].null_row_dates == (date(2024, 7, 2),)
    assert batch.statuses[0].returned_dates == ()
    assert batch.rows.empty


def test_transport_failure_is_isolated_and_has_no_retrieval_time() -> None:
    batch = make_batch(
        {
            "SPY": OSError("fictional offline error"),
            "QQQ": fictional_payload(ticker="QQQ"),
        }
    )

    assert batch.statuses[0].status == "failed"
    assert batch.statuses[0].reason == "transport_error"
    assert batch.statuses[0].detail == "fictional offline error"
    assert batch.statuses[0].retrieved_at is None
    assert batch.statuses[1].status == "success"
    assert set(batch.rows["ticker"]) == {"QQQ"}


def test_parser_failure_preserves_reason_and_exception_detail() -> None:
    batch = make_batch({"SPY": b"not strict json"})
    status = batch.statuses[0]

    assert status.status == "failed"
    assert status.reason == "malformed_payload"
    assert status.detail is not None
    assert "Yahoo response is not strict JSON" in status.detail


def test_raw_response_bytes_are_saved_and_hashed_exactly(tmp_path: Path) -> None:
    payload = fictional_payload()
    batch = make_batch({"SPY": payload}, raw_dir=tmp_path)
    status = batch.statuses[0]

    assert status.raw_sha256 == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / batch.batch_id / "SPY.json").read_bytes() == payload


def test_retrieval_clock_must_be_aware_utc() -> None:
    class OneTicker:
        def fetch(self, symbol: str, period1: int, period2: int) -> bytes:
            return fictional_payload(ticker=symbol)

    def local_time() -> datetime:
        return datetime(2024, 7, 10, 12)

    try:
        fetch_price_batch(
            load_universe(),
            DEFAULT_START,
            DEFAULT_END,
            OneTicker(),
            local_time,
        )
    except ValueError as error:
        assert "timezone-aware UTC" in str(error)
    else:
        raise AssertionError("naive retrieval time was accepted")
