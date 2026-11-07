"""Hand-calculated premium/discount and flow-proxy tests on fictional data."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from etf_monitor.arbitrage.flows import (
    creation_redemption_flow_proxy,
    premium_flow_lead_lag,
)
from etf_monitor.arbitrage.premium import PremiumInputError, premium_discount
from etf_monitor.prices.schema import PRICE_SOURCE
from tests.issuer_nav.helpers import nav_frame
from tests.prices.helpers import canonical_frame


def _prices(*rows: tuple[str, float, float | None]) -> pd.DataFrame:
    return pd.concat(
        [
            canonical_frame(
                day=day,
                open_price=close,
                high=close,
                low=close,
                close=close,
                adjusted_close=adjusted,
                source="FICTIONAL_synthetic",
            )
            for day, close, adjusted in rows
        ],
        ignore_index=True,
    )


def test_premium_uses_raw_close_in_basis_points() -> None:
    prices = _prices(("2024-07-02", 101.0, 50.0), ("2024-07-03", 99.0, 49.0))
    nav = nav_frame([("SPY", "2024-07-02", 100.0, 1), ("SPY", "2024-07-03", 100.0, 1)])

    result = premium_discount(prices, nav)

    assert result["premium_discount_bps"].tolist() == pytest.approx([100.0, -100.0])
    assert result["status"].tolist() == ["ok", "ok"]


def test_missing_inputs_give_null_premium_with_status() -> None:
    prices = _prices(("2024-07-02", 101.0, 100.0), ("2024-07-05", 101.0, 100.0))
    nav = nav_frame([("SPY", "2024-07-02", 100.0, 1), ("SPY", "2024-07-03", 100.0, 1)])

    result = premium_discount(prices, nav)

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-07-02",
        "2024-07-03",
        "2024-07-05",
    ]
    assert result["status"].tolist() == ["ok", "missing_close", "missing_nav"]
    assert result["premium_discount_bps"].isna().tolist() == [False, True, True]


def test_fictional_and_real_inputs_are_not_combined() -> None:
    real_prices = canonical_frame(source=PRICE_SOURCE)
    nav = nav_frame([("SPY", "2024-07-02", 100.0, 1)])
    with pytest.raises(PremiumInputError):
        premium_discount(real_prices, nav)


def test_flow_proxy_hand_calculation_both_alignments() -> None:
    nav = nav_frame(
        [
            ("SPY", "2024-07-01", 10.0, 1_000),
            ("SPY", "2024-07-02", 11.0, 1_100),
            ("SPY", "2024-07-03", 12.0, 1_050),
        ]
    )

    flows = creation_redemption_flow_proxy(nav)

    assert flows["status"].tolist() == ["first_observation", "ok", "ok"]
    assert flows["delta_shares"].tolist()[1:] == [100, -50]
    assert flows["flow_proxy_same_session_usd"].tolist()[1:] == [1_100.0, -600.0]
    assert flows["flow_proxy_prior_session_usd"].tolist()[1:] == [1_000.0, -550.0]
    assert pd.isna(flows["delta_shares"].iloc[0])


def test_session_gap_is_not_bridged_and_holiday_is_not_a_gap() -> None:
    nav = nav_frame(
        [
            ("SPY", "2024-07-03", 10.0, 1_000),
            ("SPY", "2024-07-05", 10.0, 1_010),  # 07-04 is a holiday
            ("SPY", "2024-07-09", 10.0, 1_100),  # 07-08 is missing
        ]
    )

    flows = creation_redemption_flow_proxy(nav)

    assert flows["status"].tolist() == [
        "first_observation",
        "ok",
        "missing_previous_session",
    ]
    assert flows["previous_session"].dt.strftime("%Y-%m-%d").tolist()[1:] == [
        "2024-07-03",
        "2024-07-08",
    ]
    assert math.isnan(flows["flow_proxy_same_session_usd"].iloc[2])


def test_missing_shares_give_null_delta() -> None:
    nav = nav_frame([("SPY", "2024-07-01", 10.0, None), ("SPY", "2024-07-02", 10.0, 5)])

    flows = creation_redemption_flow_proxy(nav)

    assert flows["status"].tolist() == [
        "first_observation",
        "missing_shares_outstanding",
    ]


def test_delta_is_exact_above_two_pow_53() -> None:
    nav = nav_frame(
        [("SPY", "2024-07-01", 1.0, 2**53 + 1), ("SPY", "2024-07-02", 1.0, 2**53 + 4)]
    )

    flows = creation_redemption_flow_proxy(nav)

    assert int(flows["delta_shares"].iloc[1]) == 3
    assert flows["delta_shares"].dtype == pd.Int64Dtype()


def test_lead_lag_detects_one_session_reporting_delay() -> None:
    days = ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05", "2024-07-08"]
    shares = [1_000, 1_000, 1_100, 1_100, 990]
    nav = nav_frame(
        [("SPY", day, 100.0, count) for day, count in zip(days, shares, strict=True)]
    )
    closes = [100.0, 101.0, 100.0, 99.0, 100.0]
    prices = _prices(
        *((day, close, close) for day, close in zip(days, closes, strict=True))
    )

    table = premium_flow_lead_lag(
        premium_discount(prices, nav), creation_redemption_flow_proxy(nav), (0, 1)
    )

    # Premiums (bps): 0, 100, 0, -100, 0. Relative share changes reported on
    # the next row: 0, +10%, 0, -10%. Lag 1 pairs them exactly; lag 0 does not.
    lag0 = table.loc[table["lag_sessions"] == 0].iloc[0]
    lag1 = table.loc[table["lag_sessions"] == 1].iloc[0]
    assert lag1["observations"] == 4
    assert lag1["correlation"] == pytest.approx(1.0)
    assert lag0["correlation"] == pytest.approx(0.0, abs=1e-12)
