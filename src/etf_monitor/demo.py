"""Deterministic fictional price data for offline bundle demonstrations."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from etf_monitor.methodology import load_methodology
from etf_monitor.prices.schema import canonicalize_frame
from etf_monitor.universe import load_universe
from etf_monitor.xnys import resolve_target_session, xnys_sessions


def build_demo_prices(evaluation_instant: datetime) -> pd.DataFrame:
    """Build a full-history, deterministic FICTIONAL_synthetic price frame.

    The output is invented demonstration data and must never be described as
    provider-observed market history.

    Args:
        evaluation_instant: Aware instant used to select the final XNYS session
            and stamped as each row's retrieval time.

    Returns:
        Canonical fictional prices for every configured ETF and session through
        the resolved target.
    """
    if evaluation_instant.tzinfo is None or evaluation_instant.utcoffset() is None:
        raise ValueError("Demo evaluation instant must be timezone-aware.")
    instant_utc = evaluation_instant.astimezone(UTC)
    target = resolve_target_session(instant_utc)
    methodology = load_methodology()
    sessions = xnys_sessions(methodology.price_history_start, target.date())
    rows: list[dict[str, object]] = []
    for ticker_position, etf in enumerate(load_universe().etfs):
        price = 40.0 + 7.0 * ticker_position
        for position, session in enumerate(sessions):
            shock = ((position * 37 + ticker_position * 13) % 29 - 14) / 10_000.0
            price *= 1.0 + shock
            close = price
            rows.append(
                {
                    "ticker": etf.ticker,
                    "date": session,
                    "open": close * 0.999,
                    "high": close * 1.003,
                    "low": close * 0.997,
                    "close": close,
                    "adjusted_close": close,
                    "volume": 100_000 + ticker_position * 1_000 + position * 10,
                    "retrieved_at": instant_utc,
                    "source": "FICTIONAL_synthetic",
                }
            )
    frame = canonicalize_frame(pd.DataFrame(rows))
    return frame
