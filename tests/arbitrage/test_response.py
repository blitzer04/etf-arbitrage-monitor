from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_monitor.arbitrage.response import forward_share_response, summarize_response

DATES = pd.bdate_range("2026-10-05", periods=8)


def _flows(so: list[int], status: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": "JNK",
            "date": DATES[: len(so)],
            "shares_outstanding": pd.array(so, dtype="Int64"),
            "status": status or ["ok"] * len(so),
        }
    )


def _band(sides: list[str | None]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": "JNK",
            "date": DATES[: len(sides)],
            "side": pd.array(sides, dtype="string"),
            "excess_bps": 1.0,
            "stale_nav": False,
        }
    )


def test_window_starts_after_the_session() -> None:
    flows = _flows([100, 100, 110, 110, 120, 120])
    out = forward_share_response(_band(["premium", "inside", None]), flows, horizon=2)
    assert len(out) == 2  # rows without a side are dropped
    assert out["forward_rel_change"].tolist() == pytest.approx([0.10, 0.10])


def test_split_or_short_window_is_null() -> None:
    flows = _flows([100, 100, 300, 300], ["ok", "ok", "split_session", "ok"])
    out = forward_share_response(
        _band(["premium", "inside", "discount", "discount"]), flows, 1
    )
    assert np.isnan(out["forward_rel_change"].iloc[1])  # window holds the split
    assert out["forward_rel_change"].iloc[2] == 0.0
    assert np.isnan(out["forward_rel_change"].iloc[3])  # no session t+1


def test_summary_counts_creations_and_redemptions() -> None:
    response = pd.DataFrame(
        {
            "ticker": ["JNK", "JNK", "JNK", "FEZ"],
            "side": ["premium", "premium", "premium", "discount"],
            "forward_rel_change": [0.01, -0.02, np.nan, -0.01],
        }
    )
    out = summarize_response(response, {"JNK": "US bond", "FEZ": "Intl equity"})
    jnk = out.set_index("group").loc["US bond"]
    assert (jnk["events"], jnk["with_flow"]) == (3, 2)
    assert jnk["share_creation"] == 0.5
    with pytest.raises(ValueError):
        summarize_response(response, {"JNK": "US bond"})
