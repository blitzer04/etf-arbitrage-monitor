"""Bid-ask spread estimate from daily high, low and close (Abdi-Ranaldo 2017).

Yahoo daily bars carry no quotes, so the quoted spread is estimated from
OHLC. With ``c`` the log close and ``eta`` the log high-low midpoint,

    s^2 = 4 * E[(c_t - eta_t) * (c_t - eta_{t+1})]

is estimated by the mean of the products over a trailing window, negative
means are set to zero (a known small-sample property of the estimator) and
``spread_bps = 10_000 * sqrt(s^2)`` is the full relative spread.

Point-in-time: the product pairing session t-1 with session t is first known
at session t's close, so the value on session t uses only products whose
second session is at or before t. A window needs ``min_periods`` products;
any missing bar breaks only the products that use it, nothing is filled.

This is an estimate of the effective spread, not a quoted spread; it is
noisy for funds whose daily range is dominated by volatility rather than
the bid-ask bounce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPREAD_COLUMNS = ("ticker", "date", "spread_products", "spread_bps")


def abdi_ranaldo_spread(
    prices: pd.DataFrame, window: int = 21, min_periods: int = 15
) -> pd.DataFrame:
    """Trailing Abdi-Ranaldo spread per ticker and session.

    Args:
        prices: Canonical price rows with ``ticker, date, high, low, close``.
            High, low and close must share one split adjustment (Yahoo's
            raw bars do); the estimator uses only within-day log ratios.
        window: Trailing number of consecutive-session products.
        min_periods: Minimum non-missing products for a value.

    Returns:
        ``ticker, date, spread_products, spread_bps`` with one row per input
        row. ``spread_bps`` is null when fewer than ``min_periods`` products
        are available.
    """
    if min_periods < 1 or window < min_periods:
        raise ValueError("Require 1 <= min_periods <= window.")
    frames: list[pd.DataFrame] = []
    ordered = prices.sort_values(["ticker", "date"], kind="stable")
    for ticker, rows in ordered.groupby("ticker", sort=True, observed=True):
        high = rows["high"].to_numpy(dtype=np.float64, na_value=np.nan)
        low = rows["low"].to_numpy(dtype=np.float64, na_value=np.nan)
        close = rows["close"].to_numpy(dtype=np.float64, na_value=np.nan)
        valid = (high > 0) & (low > 0) & (close > 0) & (high >= low)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_close = np.where(valid, np.log(close), np.nan)
            eta = np.where(valid, (np.log(high) + np.log(low)) / 2.0, np.nan)
        products = np.full(len(rows), np.nan)
        products[1:] = (log_close[:-1] - eta[:-1]) * (log_close[:-1] - eta[1:])
        series = pd.Series(products)
        mean = series.rolling(window, min_periods=min_periods).mean()
        counts = series.notna().astype(int).rolling(window, min_periods=1).sum()
        spread = 10_000.0 * np.sqrt(np.clip(4.0 * mean.to_numpy(), 0.0, None))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": rows["date"].to_numpy(),
                    "spread_products": counts.to_numpy(dtype=np.int64),
                    "spread_bps": spread,
                }
            )
        )
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in SPREAD_COLUMNS})
    return pd.concat(frames, ignore_index=True).loc[:, list(SPREAD_COLUMNS)]
