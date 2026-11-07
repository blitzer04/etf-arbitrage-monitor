"""Small canonical price-frame builders for signal regression tests."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from etf_monitor.prices.schema import empty_canonical_frame
from tests.prices.helpers import canonical_frame


def price_frame(
    ticker: str,
    sessions: pd.DatetimeIndex,
    adjusted_closes: Sequence[float | None],
    *,
    missing_rows: set[int] | None = None,
) -> pd.DataFrame:
    """Build fictional canonical prices with intentionally distinct closes."""
    if len(sessions) != len(adjusted_closes):
        raise ValueError("sessions and adjusted_closes must have equal lengths.")
    omitted = set() if missing_rows is None else missing_rows
    rows = [
        canonical_frame(
            ticker=ticker,
            day=session.strftime("%Y-%m-%d"),
            open_price=199.0 + position,
            high=202.0 + position,
            low=198.0 + position,
            close=200.0 + position,
            adjusted_close=adjusted_close,
            volume=1000 + position,
            source="FICTIONAL_synthetic",
        )
        for position, (session, adjusted_close) in enumerate(
            zip(sessions, adjusted_closes, strict=True)
        )
        if position not in omitted
    ]
    if not rows:
        return empty_canonical_frame()
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["ticker", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
