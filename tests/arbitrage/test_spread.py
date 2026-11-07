from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_monitor.arbitrage.spread import abdi_ranaldo_spread


def _bars(closes: list[float], half_range: float = 0.0) -> pd.DataFrame:
    mid = np.array(closes)
    return pd.DataFrame(
        {
            "ticker": "SPY",
            "date": pd.bdate_range("2026-01-05", periods=len(closes)),
            "high": mid * (1 + half_range),
            "low": mid * (1 - half_range),
            "close": mid,
        }
    )


def test_bid_ask_bounce_recovers_the_spread() -> None:
    # Constant midpoint 100; the close alternates between bid and ask.
    spread = 0.002
    bars = _bars([100.0] * 40)
    bars["high"] = 100 * (1 + spread / 2)
    bars["low"] = 100 * (1 - spread / 2)
    bars["close"] = [100 * (1 + spread / 2 * (-1) ** i) for i in range(40)]
    out = abdi_ranaldo_spread(bars, window=20, min_periods=20)
    # Log-space estimator: within 0.1 bp of the 20 bp spread.
    assert out["spread_bps"].iloc[-1] == pytest.approx(20.0, abs=0.1)


def test_uses_only_information_up_to_the_session() -> None:
    bars = _bars([100.0 + i for i in range(30)], half_range=0.01)
    full = abdi_ranaldo_spread(bars, window=10, min_periods=5)
    head = abdi_ranaldo_spread(bars.iloc[:20], window=10, min_periods=5)
    pd.testing.assert_series_equal(
        full["spread_bps"].iloc[:20], head["spread_bps"], check_names=False
    )


def test_missing_bar_breaks_adjacent_products_without_filling() -> None:
    bars = _bars([100.0] * 10, half_range=0.01)
    bars.loc[4, "close"] = np.nan
    out = abdi_ranaldo_spread(bars, window=10, min_periods=1)
    # Products for (3,4) and (4,5) are missing: 9 possible, 7 usable.
    assert out["spread_products"].iloc[-1] == 7


def test_negative_mean_is_zero_and_short_history_is_null() -> None:
    bars = _bars([100.0, 101.0, 102.0, 103.0], half_range=0.0)
    out = abdi_ranaldo_spread(bars, window=3, min_periods=3)
    assert out["spread_bps"].iloc[:3].isna().all()
    assert out["spread_bps"].iloc[3] == 0.0


def test_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        abdi_ranaldo_spread(_bars([1.0]), window=2, min_periods=3)
