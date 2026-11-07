"""Cost-based no-arbitrage band around NAV and dislocation flags.

An authorised participant (AP) profits from a premium by creating units
(buy the basket, deliver it, sell ETF shares) and from a discount by
redeeming. Ignoring capital and risk costs, the one-way cost in bps of NAV is

    band_bps = fee_bps + basket_cost_bps + half_spread_bps

* ``fee_bps = 10_000 * creation_fee_usd / (creation_unit_shares * nav)``:
  the issuer's fixed transaction fee per creation unit, from its SAI.
* ``basket_cost_bps``: an assumed one-way cost of trading the underlying
  basket, set per asset class in the cost config (documented assumption,
  not observed).
* ``half_spread_bps``: an assumed ETF half bid-ask spread per asset class
  (the cost of the ETF leg). Yahoo has no quotes, and the OHLC estimate in
  :mod:`etf_monitor.arbitrage.spread` is too noisy at daily horizons for
  these funds (zero in a large share of windows), so it is reported as
  ``spread_estimate_bps`` for reference only and does not enter the band.

A session is outside the band when ``|premium_discount_bps| > band_bps``;
``persistent`` marks a session whose previous NAV row for the same ticker
was outside the band on the same side, which screens out one-day
close/NAV measurement noise. For
funds whose NAV is struck from foreign closes (``stale_nav``), the premium
mixes mispricing with price moves after the foreign close, so those flags
are reported but kept separate. Any missing input leaves the band and the
flag null with a status; nothing is filled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BAND_COLUMNS = (
    "ticker",
    "date",
    "premium_discount_bps",
    "nav",
    "fee_bps",
    "basket_cost_bps",
    "half_spread_bps",
    "band_bps",
    "excess_bps",
    "side",
    "persistent",
    "spread_estimate_bps",
    "stale_nav",
    "status",
)


@dataclass(frozen=True)
class FundCosts:
    """Cost inputs for one fund; ``creation_fee_usd`` is None when unknown."""

    ticker: str
    creation_unit_shares: int
    creation_fee_usd: float | None
    basket_cost_bps: float
    half_spread_bps: float
    stale_nav: bool


def arbitrage_band(
    premium: pd.DataFrame,
    spread: pd.DataFrame,
    costs: dict[str, FundCosts],
) -> pd.DataFrame:
    """Band, excess over band and side for every premium row.

    Args:
        premium: Output of :func:`etf_monitor.arbitrage.premium.premium_discount`.
        spread: Output of :func:`etf_monitor.arbitrage.spread.abdi_ranaldo_spread`,
            carried through as a diagnostic only.
        costs: Cost inputs by ticker.

    Returns:
        One row per premium row. ``side`` is ``premium``/``discount`` outside
        the band and ``inside`` within it; ``excess_bps`` is
        ``|premium| - band`` (negative inside). Status is ``ok`` or names
        the first missing input.
    """
    joined = (
        premium.loc[:, ["ticker", "date", "premium_discount_bps", "nav", "status"]]
        .sort_values(["ticker", "date"], kind="stable")
        .reset_index(drop=True)
    ).merge(
        spread.loc[:, ["ticker", "date", "spread_bps"]],
        on=["ticker", "date"],
        how="left",
        validate="1:1",
    )
    tickers = joined["ticker"].astype(str)
    unknown = sorted(set(tickers) - set(costs))
    if unknown:
        raise ValueError(f"No cost config for {unknown}.")
    unit = tickers.map({t: c.creation_unit_shares for t, c in costs.items()})
    fee_usd = tickers.map(
        {
            t: np.nan if c.creation_fee_usd is None else c.creation_fee_usd
            for t, c in costs.items()
        }
    ).astype(np.float64)
    nav = joined["nav"].astype(np.float64)
    fee_bps = 10_000.0 * fee_usd / (unit.astype(np.float64) * nav)
    basket = tickers.map({t: c.basket_cost_bps for t, c in costs.items()})
    half_spread = tickers.map({t: c.half_spread_bps for t, c in costs.items()}).astype(
        np.float64
    )
    band = fee_bps + basket.astype(np.float64) + half_spread
    prem = joined["premium_discount_bps"].astype(np.float64)

    status = pd.Series("ok", index=joined.index, dtype="string")
    status[fee_usd.isna()] = "missing_creation_fee"
    premium_bad = (joined["status"] != "ok") | prem.isna()
    prefixed = joined["status"].astype(str).map(lambda value: f"premium_{value}")
    status[premium_bad] = prefixed[premium_bad]
    ok = status == "ok"

    excess = pd.Series(
        np.where(ok, prem.abs().to_numpy() - band.to_numpy(), np.nan),
        index=joined.index,
        dtype=np.float64,
    )
    side = pd.Series(pd.NA, index=joined.index, dtype="string")
    side[ok] = "inside"
    outside = ok.to_numpy() & (excess.to_numpy() > 0)
    sign = np.sign(prem.to_numpy())
    side[outside & (sign > 0)] = "premium"
    side[outside & (sign < 0)] = "discount"

    previous_side = side.groupby(joined["ticker"], sort=False).shift(1)
    persistent = pd.Series(pd.NA, index=joined.index, dtype="boolean")
    persistent[ok] = (
        side.isin(["premium", "discount"]) & (side == previous_side)
    ).fillna(False)[ok]

    out = pd.DataFrame(
        {
            "ticker": joined["ticker"],
            "date": joined["date"],
            "premium_discount_bps": prem,
            "nav": nav,
            "fee_bps": fee_bps,
            "basket_cost_bps": basket.astype(np.float64),
            "half_spread_bps": half_spread,
            "band_bps": band.where(ok),
            "excess_bps": excess,
            "side": side,
            "persistent": persistent,
            "spread_estimate_bps": joined["spread_bps"].astype(np.float64),
            "stale_nav": tickers.map({t: c.stale_nav for t, c in costs.items()}),
            "status": status,
        }
    )
    return out.loc[:, list(BAND_COLUMNS)]
