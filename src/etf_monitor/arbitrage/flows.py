"""Creation/redemption flow proxy from issuer shares-outstanding changes.

This is a **creation/redemption flow proxy**, not a reported fund flow. For
reporting date t and the immediately preceding XNYS session p(t):

* ``delta_shares = shares_outstanding[t] - shares_outstanding[p(t)]`` exact
  integer shares (nullable ``Int64``).
* ``flow_proxy_same_session_usd = delta_shares * nav[t]`` assumes the change
  reported on t was created/redeemed in session t.
* ``flow_proxy_prior_session_usd = delta_shares * nav[p(t)]`` assumes the
  issuer reports creations one session late (orders struck at p(t)'s NAV).

Both alignments are kept because the issuer does not document which applies;
:func:`premium_flow_lead_lag` measures them against the premium. Units are
U.S. dollars. The proxy ignores in-kind versus cash mechanics, fees, and
distributions, which change net assets without changing shares.

Missing data: the previous observation is always the previous XNYS session,
never the last available row. If that session has no row or either share
count is missing, ``delta_shares`` and both proxies are null with a status.
On a split session the share change is not a flow, so it is null with status
``split_session``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from etf_monitor.arbitrage.splits import SplitEvent
from etf_monitor.xnys import xnys_sessions

FLOW_COLUMNS = (
    "ticker",
    "date",
    "previous_session",
    "shares_outstanding",
    "previous_shares_outstanding",
    "delta_shares",
    "nav",
    "previous_nav",
    "flow_proxy_same_session_usd",
    "flow_proxy_prior_session_usd",
    "status",
)
LEAD_LAG_COLUMNS = ("ticker", "lag_sessions", "observations", "correlation")


def creation_redemption_flow_proxy(
    nav: pd.DataFrame, splits: Sequence[SplitEvent] = ()
) -> pd.DataFrame:
    """Compute daily share changes and both flow-proxy alignments.

    Args:
        nav: Validated canonical issuer NAV frame.
        splits: Split events; their sessions get no share change or proxy.

    Returns:
        One row per NAV row, sorted by ticker and date. ``status`` is ``ok``,
        ``first_observation``, ``missing_previous_session``,
        ``missing_shares_outstanding`` or ``split_session``.
    """
    frames: list[pd.DataFrame] = []
    for ticker, rows in nav.groupby("ticker", sort=True, observed=True):
        rows = rows.reset_index(drop=True)
        first = rows["date"].iloc[0].date()
        last = rows["date"].iloc[-1].date()
        sessions = xnys_sessions(first, last)
        previous_by_session = pd.Series(
            [pd.NaT, *sessions[:-1]], index=sessions, dtype="datetime64[ns]"
        )
        previous_session = previous_by_session.reindex(rows["date"]).to_numpy()
        lookup = rows.set_index("date")
        previous = lookup.reindex(pd.DatetimeIndex(previous_session))
        shares = rows["shares_outstanding"].reset_index(drop=True)
        previous_shares = previous["shares_outstanding"].reset_index(drop=True)
        delta = shares - previous_shares
        previous_nav = previous["nav"].to_numpy(dtype=np.float64)
        delta_float = delta.astype("Float64").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        is_first = pd.isna(previous_session)
        previous_missing = ~is_first & np.isnan(previous_nav)
        split_days = {event.date for event in splits if event.ticker == ticker}
        is_split = rows["date"].dt.date.isin(split_days).to_numpy()
        status = np.select(
            [is_first, is_split, previous_missing, delta.isna().to_numpy()],
            [
                "first_observation",
                "split_session",
                "missing_previous_session",
                "missing_shares_outstanding",
            ],
            default="ok",
        )
        delta = delta.mask(is_split)
        delta_float = np.where(is_split, np.nan, delta_float)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": pd.array([str(ticker)] * len(rows), dtype="string"),
                    "date": rows["date"],
                    "previous_session": pd.Series(
                        previous_session, dtype="datetime64[ns]"
                    ),
                    "shares_outstanding": shares,
                    "previous_shares_outstanding": previous_shares.astype("Int64"),
                    "delta_shares": delta.astype("Int64"),
                    "nav": rows["nav"],
                    "previous_nav": previous_nav,
                    "flow_proxy_same_session_usd": delta_float
                    * rows["nav"].to_numpy(dtype=np.float64),
                    "flow_proxy_prior_session_usd": delta_float * previous_nav,
                    "status": pd.array(status, dtype="string"),
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            {column: pd.Series(dtype="object") for column in FLOW_COLUMNS}
        )
    return pd.concat(frames, ignore_index=True).loc[:, list(FLOW_COLUMNS)]


def premium_flow_lead_lag(
    premium: pd.DataFrame,
    flows: pd.DataFrame,
    lags: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    """Correlate session-t premium with relative share change k rows later.

    The relative share change is ``delta_shares / previous_shares_outstanding``
    on ``ok`` flow rows. Lag k pairs the premium of session t with the share
    change reported on the k-th following NAV row of the same ticker, so lag
    1 tests the prior-session reporting alignment. Pairs with either value
    missing are dropped; this is a descriptive Pearson correlation, not a
    trading signal.

    Args:
        premium: Output of :func:`etf_monitor.arbitrage.premium.premium_discount`.
        flows: Output of :func:`creation_redemption_flow_proxy`.
        lags: Non-negative row offsets to evaluate.

    Returns:
        One row per ticker and lag with the pair count and correlation (null
        when fewer than three pairs or either side has zero variance).
    """
    records: list[dict[str, object]] = []
    joined = flows.merge(
        premium.loc[
            premium["status"] == "ok", ["ticker", "date", "premium_discount_bps"]
        ],
        on=["ticker", "date"],
        how="left",
        validate="1:1",
    )
    for ticker, rows in joined.groupby("ticker", sort=True, observed=True):
        relative = (
            rows["delta_shares"].astype("Float64")
            / rows["previous_shares_outstanding"].astype("Float64")
        ).where(rows["status"] == "ok")
        relative_values = relative.to_numpy(dtype=np.float64, na_value=np.nan)
        premium_values = rows["premium_discount_bps"].to_numpy(dtype=np.float64)
        for lag in lags:
            if lag < 0:
                raise ValueError("lags must be non-negative.")
            x = premium_values[: len(rows) - lag] if lag else premium_values
            y = relative_values[lag:]
            usable = ~(np.isnan(x) | np.isnan(y))
            count = int(usable.sum())
            correlation: float | None = None
            if count >= 3 and np.std(x[usable]) > 0 and np.std(y[usable]) > 0:
                correlation = float(np.corrcoef(x[usable], y[usable])[0, 1])
            records.append(
                {
                    "ticker": ticker,
                    "lag_sessions": lag,
                    "observations": count,
                    "correlation": correlation,
                }
            )
    return pd.DataFrame.from_records(records, columns=list(LEAD_LAG_COLUMNS))
