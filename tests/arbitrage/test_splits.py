"""Split-event parsing and split-consistent premium/flow tests on fictional data."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from etf_monitor.arbitrage.flows import creation_redemption_flow_proxy
from etf_monitor.arbitrage.premium import premium_discount
from etf_monitor.arbitrage.splits import (
    SplitEvent,
    SplitPayloadError,
    split_events_from_chart,
)
from tests.issuer_nav.helpers import nav_frame
from tests.prices.helpers import canonical_frame


def _epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC).timestamp())


def _chart(symbol: str, splits: dict[str, dict[str, object]] | None) -> bytes:
    result: dict[str, object] = {"meta": {"symbol": symbol}}
    if splits is not None:
        result["events"] = {"splits": splits}
    return json.dumps({"chart": {"result": [result], "error": None}}).encode()


def _split(day: date, numerator: object, denominator: object) -> dict[str, object]:
    return {"date": _epoch(day), "numerator": numerator, "denominator": denominator}


def test_parses_forward_and_reverse_splits_sorted() -> None:
    raw = _chart(
        "SPY",
        {
            "b": _split(date(2024, 7, 3), 1.0, 3.0),
            "a": _split(date(2024, 7, 2), 2.0, 1.0),
        },
    )
    events = split_events_from_chart(raw, "SPY")
    assert events == (
        SplitEvent("SPY", date(2024, 7, 2), 2, 1),
        SplitEvent("SPY", date(2024, 7, 3), 1, 3),
    )
    assert events[1].ratio == pytest.approx(1 / 3)


def test_payload_without_events_has_no_splits() -> None:
    assert split_events_from_chart(_chart("SPY", None), "SPY") == ()


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_chart("QQQ", {}), "not 'SPY'"),
        (_chart("SPY", {"a": _split(date(2024, 7, 2), 1.5, 1.0)}), "numerator"),
        (_chart("SPY", {"a": _split(date(2024, 7, 2), 2.0, 0.0)}), "denominator"),
        (_chart("SPY", {"a": _split(date(2024, 7, 6), 2.0, 1.0)}), "non-session"),
        (
            _chart(
                "SPY",
                {
                    "a": _split(date(2024, 7, 2), 2.0, 1.0),
                    "b": _split(date(2024, 7, 2), 3.0, 1.0),
                },
            ),
            "two splits",
        ),
        (b"{not json", "malformed"),
    ],
)
def test_rejects_invalid_split_payloads(raw: bytes, message: str) -> None:
    with pytest.raises(SplitPayloadError, match=message):
        split_events_from_chart(raw, "SPY")


def test_premium_undoes_yahoo_split_adjustment_before_split_only() -> None:
    prices = pd.concat(
        [
            canonical_frame(
                day=day,
                open_price=close,
                high=close,
                low=close,
                close=close,
                adjusted_close=close,
                source="FICTIONAL_synthetic",
            )
            for day, close in (("2024-07-02", 50.0), ("2024-07-03", 51.0))
        ],
        ignore_index=True,
    )
    nav = nav_frame([("SPY", "2024-07-02", 100.0, 10), ("SPY", "2024-07-03", 51.0, 20)])
    split = (SplitEvent("SPY", date(2024, 7, 3), 2, 1),)

    result = premium_discount(prices, nav, split)

    assert result["split_factor"].tolist() == [2.0, 1.0]
    assert result["close_as_traded"].tolist() == [100.0, 51.0]
    assert result["premium_discount_bps"].tolist() == pytest.approx([0.0, 0.0])


def test_split_session_has_no_flow_proxy_but_later_sessions_do() -> None:
    nav = nav_frame(
        [
            ("SPY", "2024-07-01", 100.0, 10),
            ("SPY", "2024-07-02", 50.0, 20),
            ("SPY", "2024-07-03", 50.0, 23),
        ]
    )
    split = (SplitEvent("SPY", date(2024, 7, 2), 2, 1),)

    result = creation_redemption_flow_proxy(nav, split)

    assert result["status"].tolist() == ["first_observation", "split_session", "ok"]
    assert pd.isna(result.loc[1, "delta_shares"])
    assert pd.isna(result.loc[1, "flow_proxy_same_session_usd"])
    assert result.loc[2, "delta_shares"] == 3
    assert result.loc[2, "flow_proxy_same_session_usd"] == pytest.approx(150.0)
