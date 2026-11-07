"""Align canonical adjusted prices to exact XNYS session positions."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from etf_monitor.methodology import load_methodology
from etf_monitor.prices.schema import validate_canonical
from etf_monitor.universe import load_universe
from etf_monitor.xnys import is_session, xnys_sessions


def _calendar_date(value: date | pd.Timestamp, name: str) -> date:
    if isinstance(value, pd.Timestamp):
        if value.tz is not None or value != value.normalize():
            raise ValueError(f"{name} must be a timezone-naive date label.")
        return value.date()
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a date or timezone-naive date label.")
    return value


def build_session_panel(
    prices: pd.DataFrame,
    ticker: str,
    target: date | pd.Timestamp,
    *,
    grid_start: date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Align one ETF's adjusted closes to a contiguous XNYS session grid.

    The default grid begins at the first XNYS session on or after the
    methodology ``price_history_start``. ``grid_start`` exists only to build
    small hand-checkable unit-test grids; the Gate 4 evaluation path must pass
    ``None``. Canonical rows outside the selected grid are rejected rather
    than silently filtered.

    Args:
        prices: Validated canonical daily prices. Rows for all tickers must lie
            within the selected grid.
        ticker: Configured ETF whose rows are aligned.
        target: Inclusive final XNYS session.
        grid_start: Optional test-only lower calendar date. The first session
            on or after it begins the grid.

    Returns:
        A DataFrame indexed by zero-based XNYS position with ``session``,
        ``row_present``, and float64 ``adjusted_close`` columns.

    Raises:
        ValueError: If the target is not an XNYS session or input dates fall
            outside the selected grid.
        TypeError: If a date argument has an unsupported type.
    """
    target_date = _calendar_date(target, "target")
    if grid_start is None:
        requested_start = load_methodology().price_history_start
    else:
        requested_start = _calendar_date(grid_start, "grid_start")
    if requested_start > target_date:
        raise ValueError("grid_start must be on or before target.")
    if not is_session(target_date):
        raise ValueError(f"target {target_date} is not an XNYS session.")
    if ticker not in {etf.ticker for etf in load_universe().etfs}:
        raise ValueError(f"ticker {ticker!r} is not in the configured universe.")

    validate_canonical(prices)
    sessions = xnys_sessions(requested_start, target_date)
    first_session = sessions[0]
    target_session = sessions[-1]
    outside_grid = prices["date"].lt(first_session) | prices["date"].gt(target_session)
    if outside_grid.any():
        examples = prices.loc[outside_grid, ["ticker", "date"]].head(5)
        raise ValueError(
            "Canonical price rows fall outside the session grid: "
            f"{examples.to_dict(orient='records')}."
        )

    ticker_prices = prices.loc[prices["ticker"].eq(ticker)].set_index("date")
    row_present = sessions.isin(ticker_prices.index)
    adjusted_close = (
        ticker_prices["adjusted_close"]
        .reindex(sessions)
        .to_numpy(dtype=np.float64, na_value=np.nan)
    )
    panel = pd.DataFrame(
        {
            "session": sessions,
            "row_present": np.asarray(row_present, dtype=np.bool_),
            "adjusted_close": adjusted_close,
        },
        index=pd.RangeIndex(len(sessions), name="xnys_position"),
    )
    panel["adjusted_close"] = panel["adjusted_close"].astype("float64")
    return panel
