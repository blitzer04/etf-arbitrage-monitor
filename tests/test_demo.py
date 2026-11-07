from __future__ import annotations

from datetime import datetime

import pandas as pd

from etf_monitor.demo import build_demo_prices
from etf_monitor.universe import load_universe


def test_demo_prices_are_deterministic_and_explicitly_fictional() -> None:
    instant = datetime.fromisoformat("2018-01-02T22:00:00+00:00")

    first = build_demo_prices(instant)
    second = build_demo_prices(instant)

    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first["source"].unique().tolist() == ["FICTIONAL_synthetic"]
    assert set(first["ticker"].unique()) == {etf.ticker for etf in load_universe().etfs}
    assert len(first) == 24
