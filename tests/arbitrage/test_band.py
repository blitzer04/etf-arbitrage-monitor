from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_monitor.arbitrage.band import FundCosts, arbitrage_band

COSTS = {
    "JNK": FundCosts("JNK", 50_000, 500.0, 20.0, 5.0, stale_nav=False),
    "FEZ": FundCosts("FEZ", 50_000, None, 5.0, 2.0, stale_nav=True),
}


def _premium(rows: list[tuple[str, float | None, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "date": pd.Timestamp("2026-11-03"),
            "premium_discount_bps": [np.nan if r[1] is None else r[1] for r in rows],
            "nav": 100.0,
            "status": [r[2] for r in rows],
        }
    ).assign(date=lambda f: f["date"] + pd.to_timedelta(range(len(f)), "D"))


def _spread(premium: pd.DataFrame, bps: float | None = 10.0) -> pd.DataFrame:
    return premium.loc[:, ["ticker", "date"]].assign(
        spread_bps=np.nan if bps is None else bps
    )


def test_band_is_fee_plus_basket_plus_half_spread() -> None:
    p = _premium([("JNK", 40.0, "ok"), ("JNK", -40.0, "ok"), ("JNK", 10.0, "ok")])
    out = arbitrage_band(p, _spread(p), COSTS)
    # fee 500 / (50_000 * 100) = 1 bp; band = 1 + 20 + 5 = 26 bps.
    assert out["band_bps"].tolist() == pytest.approx([26.0] * 3)
    assert out["side"].tolist() == ["premium", "discount", "inside"]
    assert out["excess_bps"].tolist() == pytest.approx([14.0, 14.0, -16.0])


def test_missing_inputs_leave_band_null_with_status() -> None:
    p = _premium([("FEZ", 40.0, "ok"), ("JNK", None, "missing_nav")])
    out = arbitrage_band(p, _spread(p), COSTS)
    assert out["status"].tolist() == ["missing_creation_fee", "premium_missing_nav"]
    assert out["band_bps"].isna().all() and out["side"].isna().all()
    assert out["stale_nav"].tolist() == [True, False]


def test_unknown_ticker_is_rejected() -> None:
    p = _premium([("SPY", 1.0, "ok")])
    with pytest.raises(ValueError, match="SPY"):
        arbitrage_band(p, _spread(p), COSTS)


def test_spread_estimate_is_diagnostic_only() -> None:
    p = _premium([("JNK", 40.0, "ok")])
    wide = arbitrage_band(p, _spread(p, 500.0), COSTS)
    missing = arbitrage_band(p, _spread(p, None), COSTS)
    assert (
        wide["band_bps"].iloc[0] == missing["band_bps"].iloc[0] == pytest.approx(26.0)
    )
    assert wide["spread_estimate_bps"].iloc[0] == 500.0
    assert missing["status"].iloc[0] == "ok"


def test_persistent_needs_same_side_on_previous_row() -> None:
    rows: list[tuple[str, float | None, str]] = [
        ("JNK", 40.0, "ok"),
        ("JNK", 41.0, "ok"),
        ("JNK", -40.0, "ok"),
        ("JNK", 1.0, "ok"),
    ]
    p = _premium(rows).iloc[::-1]  # input order must not matter
    out = arbitrage_band(p, _spread(p), COSTS)
    assert out["side"].tolist() == ["premium", "premium", "discount", "inside"]
    assert out["persistent"].tolist() == [False, True, False, False]
