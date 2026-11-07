"""Daily ETF premium/discount of market close to issuer NAV.

``premium_discount_bps = 10_000 * (close_as_traded / nav - 1)`` for one
ticker and one XNYS session, in basis points. ``close`` is Yahoo's raw
(split-adjusted, not dividend-adjusted) regular-session close, not
``adjusted_close``: NAV is a per-share value on that session, and dividend
back-adjustment would scale historical closes away from the NAV they are
compared to. SSGA does not restate NAV for splits, so
``close_as_traded = close * split_factor`` undoes Yahoo's split adjustment
(see :mod:`etf_monitor.arbitrage.splits`). Positive values
are premiums (price above NAV), negative values discounts.

Timing: both inputs describe the same session's close. SSGA strikes NAV at
the 4:00 p.m. ET close and publishes it that evening, so a premium for
session t is knowable only after session t ends. For funds holding foreign
securities the NAV uses earlier foreign closes, so part of the measured
premium reflects stale component prices rather than mispricing.

Missing data: a session where either input is absent gets a null premium and
a status naming the missing input; nothing is filled from other sessions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from etf_monitor.arbitrage.splits import SplitEvent, split_factors
from etf_monitor.issuer_nav.schema import FICTIONAL_NAV_SOURCE

PREMIUM_COLUMNS = (
    "ticker",
    "date",
    "close",
    "split_factor",
    "close_as_traded",
    "nav",
    "premium_discount_bps",
    "status",
)
_FICTIONAL_PRICE_SOURCE = "FICTIONAL_synthetic"


class PremiumInputError(ValueError):
    """Indicate that price and NAV inputs cannot be combined."""


def _is_fictional(frame: pd.DataFrame) -> bool:
    return bool(
        frame["source"].isin([_FICTIONAL_PRICE_SOURCE, FICTIONAL_NAV_SOURCE]).any()
    )


def premium_discount(
    prices: pd.DataFrame,
    nav: pd.DataFrame,
    splits: Sequence[SplitEvent] = (),
) -> pd.DataFrame:
    """Compute daily premium/discount for every ticker present in ``nav``.

    Args:
        prices: Validated canonical Yahoo price frame.
        nav: Validated canonical issuer NAV frame.
        splits: Split events for the ``nav`` tickers from the Yahoo payloads
            that produced ``prices``.

    Returns:
        One row per ticker and session present in either input for tickers in
        ``nav``, sorted by ticker and date, with ``status`` of ``ok``,
        ``missing_close`` or ``missing_nav``.

    Raises:
        PremiumInputError: If fictional data would be mixed with real data.
    """
    if _is_fictional(prices) != _is_fictional(nav) and not (prices.empty or nav.empty):
        raise PremiumInputError("Fictional and real inputs must not be combined.")
    tickers = set(nav["ticker"].array)
    left = prices.loc[prices["ticker"].isin(tickers), ["ticker", "date", "close"]]
    right = nav.loc[:, ["ticker", "date", "nav"]]
    merged = left.merge(right, on=["ticker", "date"], how="outer", validate="1:1")
    merged = merged.sort_values(["ticker", "date"], kind="mergesort").reset_index(
        drop=True
    )
    merged["split_factor"] = split_factors(merged["ticker"], merged["date"], splits)
    merged["close_as_traded"] = merged["close"] * merged["split_factor"]
    close = merged["close_as_traded"].to_numpy(dtype=np.float64)
    nav_values = merged["nav"].to_numpy(dtype=np.float64)
    merged["premium_discount_bps"] = 10_000.0 * (close / nav_values - 1.0)
    status = np.where(
        np.isnan(nav_values),
        "missing_nav",
        np.where(np.isnan(close), "missing_close", "ok"),
    )
    merged["status"] = pd.array(status, dtype="string")
    merged["ticker"] = merged["ticker"].astype("string")
    return merged.loc[:, list(PREMIUM_COLUMNS)]
