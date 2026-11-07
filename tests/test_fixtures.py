"""Tests for deterministic fictional provider and price fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from tests.fixtures.yahoo_chart import (
    build_synthetic_price_series,
    build_yahoo_chart_payload,
)


def test_chart_builder_emits_expected_synthetic_shape_and_nulls() -> None:
    payload = build_yahoo_chart_payload(
        null_quote_positions={"volume": (1,)},
        null_adjusted_positions=(0,),
    )
    chart = payload["chart"]
    assert chart["error"] is None
    result = chart["result"][0]
    assert result["meta"]["symbol"] == "SPY"
    assert result["meta"]["exchangeTimezoneName"] == "America/New_York"
    assert len(result["timestamp"]) == 2
    first_timestamp = result["timestamp"][0]
    assert datetime.fromtimestamp(first_timestamp, UTC).astimezone(
        ZoneInfo("America/New_York")
    ).time() == time(9, 30)
    indicators = result["indicators"]
    assert indicators["quote"][0]["volume"] == [1_000_000, None]
    assert indicators["adjclose"][0]["adjclose"] == [None, 101.5]
    assert "events" not in result


def test_chart_builder_matches_observed_payload_key_structure() -> None:
    payload = build_yahoo_chart_payload()
    assert set(payload) == {"chart"}
    chart = payload["chart"]
    assert set(chart) == {"result", "error"}
    result = chart["result"][0]
    assert set(result) == {"meta", "timestamp", "indicators"}
    meta = result["meta"]
    assert set(meta) == {
        "currency",
        "symbol",
        "exchangeName",
        "fullExchangeName",
        "instrumentType",
        "firstTradeDate",
        "regularMarketTime",
        "hasPrePostMarketData",
        "gmtoffset",
        "timezone",
        "exchangeTimezoneName",
        "regularMarketPrice",
        "regularMarketChangePercent",
        "fulldayPrice",
        "fulldayChange",
        "fulldayChangePercent",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "regularMarketDayHigh",
        "regularMarketDayLow",
        "regularMarketVolume",
        "longName",
        "shortName",
        "chartPreviousClose",
        "priceHint",
        "currentTradingPeriod",
        "dataGranularity",
        "range",
        "validRanges",
    }
    assert set(meta["currentTradingPeriod"]) == {"pre", "regular", "post"}
    assert all(
        set(period) == {"timezone", "start", "end", "gmtoffset"}
        for period in meta["currentTradingPeriod"].values()
    )
    indicators = result["indicators"]
    assert set(indicators) == {"quote", "adjclose"}
    assert set(indicators["quote"][0]) == {"open", "high", "low", "close", "volume"}
    assert set(indicators["adjclose"][0]) == {"adjclose"}


def test_chart_builder_supports_absent_adjusted_close_and_errors() -> None:
    missing_adjusted = build_yahoo_chart_payload(include_adjusted_close=False)
    assert "adjclose" not in missing_adjusted["chart"]["result"][0]["indicators"]

    failure = build_yahoo_chart_payload(provider_error={"code": "SyntheticError"})
    assert failure["chart"]["result"] is None
    assert failure["chart"]["error"] == {"code": "SyntheticError"}

    empty = build_yahoo_chart_payload(empty_result=True)
    assert empty["chart"]["result"] == []


def test_synthetic_price_series_is_repeatable_and_positive() -> None:
    prices = build_synthetic_price_series(30, seed=31)

    assert prices == build_synthetic_price_series(30, seed=31)
    assert all(price > 0 for price in prices)
