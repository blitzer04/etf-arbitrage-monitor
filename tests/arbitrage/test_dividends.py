"""Offline tests for Yahoo dividend events and ex-dividend basket values."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from etf_monitor.arbitrage.dividends import (
    DividendEvent,
    DividendPayloadError,
    as_traded_dividends,
    dividend_events_from_chart,
    ex_dividend_value,
)
from etf_monitor.arbitrage.splits import SplitEvent


def _chart(dividends: dict[str, object] | None = None, *, ticker: str = "AAA") -> bytes:
    result: dict[str, object] = {"meta": {"symbol": ticker}}
    if dividends is not None:
        result["events"] = {"dividends": dividends}
    return json.dumps({"chart": {"result": [result]}}).encode()


def test_dividend_events_are_sorted_and_identical_duplicate_dates_collapse() -> None:
    events = dividend_events_from_chart(
        _chart(
            {
                "first": {"date": 1794839400, "amount": 1.25},
                "same": {"date": 1794839400, "amount": 1.25},
                "second": {"date": 1773149400, "amount": 0.5},
            }
        ),
        "AAA",
    )

    assert events == (
        DividendEvent("AAA", date(2026, 3, 10), 0.5),
        DividendEvent("AAA", date(2026, 11, 16), 1.25),
    )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"chart":{"result":[{"meta":{"symbol":"AAA"}}]}}',
        _chart({}),
    ],
)
def test_no_dividend_events_return_empty_tuple(payload: bytes) -> None:
    assert dividend_events_from_chart(payload, "AAA") == ()


def test_integer_dividend_amount_is_normalized_to_float() -> None:
    events = dividend_events_from_chart(
        _chart({"event": {"date": 1794839400, "amount": 1}}), "AAA"
    )

    assert events == (DividendEvent("AAA", date(2026, 11, 16), 1.0),)


def test_truncated_chart_payload_is_rejected() -> None:
    with pytest.raises(DividendPayloadError, match="malformed"):
        dividend_events_from_chart(b'{"chart":{"result":[', "AAA")


@pytest.mark.parametrize("amount", [True, "1.25", float("nan"), float("inf"), 0, -0.5])
def test_invalid_dividend_amount_is_rejected(amount: object) -> None:
    with pytest.raises(DividendPayloadError):
        dividend_events_from_chart(
            _chart({"event": {"date": 1794839400, "amount": amount}}), "AAA"
        )


@pytest.mark.parametrize("event_date", [1794839400.0, "1794839400", True])
def test_non_integer_dividend_epoch_is_rejected(event_date: object) -> None:
    with pytest.raises(DividendPayloadError, match="integer date"):
        dividend_events_from_chart(
            _chart({"event": {"date": event_date, "amount": 1.0}}), "AAA"
        )


def test_out_of_range_dividend_epoch_is_rejected() -> None:
    with pytest.raises(DividendPayloadError, match="invalid epoch"):
        dividend_events_from_chart(
            _chart({"event": {"date": 2**63, "amount": 1.0}}), "AAA"
        )


def test_dividend_amount_rejects_lossy_large_integer_conversion() -> None:
    with pytest.raises(DividendPayloadError, match="represented exactly"):
        dividend_events_from_chart(
            _chart(
                {
                    "event": {
                        "date": 1794839400,
                        "amount": 2**53 + 1,
                    }
                }
            ),
            "AAA",
        )


def test_conflicting_duplicate_dividend_date_is_rejected() -> None:
    with pytest.raises(DividendPayloadError, match="conflicting dividends"):
        dividend_events_from_chart(
            _chart(
                {
                    "first": {"date": 1794839400, "amount": 1.0},
                    "second": {"date": 1794839400, "amount": 1.25},
                }
            ),
            "AAA",
        )


def test_dividend_payload_identity_must_match_requested_symbol() -> None:
    with pytest.raises(DividendPayloadError, match="not 'AAA'"):
        dividend_events_from_chart(_chart(ticker="BBB"), "AAA")


def test_as_traded_dividends_restores_pre_split_amount() -> None:
    dividends = (DividendEvent("AAA", date(2024, 1, 5), 0.5),)
    splits = (SplitEvent("AAA", date(2024, 1, 8), 2, 1),)

    assert as_traded_dividends(dividends, splits) == (
        DividendEvent("AAA", date(2024, 1, 5), 1.0),
    )


def test_as_traded_dividends_preserves_pre_split_capture_amount() -> None:
    dividends = (DividendEvent("AAA", date(2024, 1, 5), 1.0),)

    assert as_traded_dividends(dividends, ()) == dividends


def test_ex_dividend_value_uses_window_and_later_split_factor() -> None:
    holdings = pd.DataFrame(
        {
            "ticker": ["FUND_A", "FUND_A", "FUND_B", "FUND_C", "CASH_ONLY"],
            "security_ticker": ["AAA", None, "BBB", "CCC", None],
            "local_currency": ["USD", "USD", "USD", "USD", "USD"],
            "identifier": ["123", "999USDZ92", "456", "789", "999USDZ92"],
            "name": ["Company A", "Cash", "Company B", "Company C", "Cash"],
            "quantity": [10.0, 100.0, 4.0, 3.0, 250.0],
        }
    )
    dividends = (
        DividendEvent("AAA", date(2024, 1, 1), 9.0),
        DividendEvent("AAA", date(2024, 1, 8), 0.5),
        DividendEvent("AAA", date(2024, 1, 15), 1.5),
        DividendEvent("BBB", date(2024, 1, 15), 2.0),
    )
    splits = (SplitEvent("AAA", date(2024, 2, 1), 2, 1),)

    as_traded = as_traded_dividends(dividends, splits)
    result = ex_dividend_value(
        holdings,
        as_traded,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-15"),
    )

    assert result.to_dict() == pytest.approx(
        {"FUND_A": 40.0, "FUND_B": 8.0, "FUND_C": 0.0}
    )
    assert "CASH_ONLY" not in result.index
