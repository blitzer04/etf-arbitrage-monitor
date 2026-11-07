"""Do creations/redemptions follow sessions outside the arbitrage band?

For each band row (session t) the relative share change over the next
``horizon`` XNYS sessions is

    forward_rel_change = SO[t + horizon] / SO[t] - 1

taken from the flow proxy's share counts. SSGA reports a creation one
session after its trade date (see the Gate 9 timing note), so the window
starts at t+1, the first session whose shares outstanding can reflect an AP
reacting to t's close. A premium is *consistent* when followed by net
creation (> 0) and a discount when followed by net redemption (< 0).
``inside`` rows give the baseline. Windows containing a split session or a
missing share count are null; this is descriptive, not a trading signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RESPONSE_COLUMNS = (
    "ticker",
    "date",
    "side",
    "excess_bps",
    "stale_nav",
    "forward_rel_change",
)
SUMMARY_COLUMNS = (
    "group",
    "side",
    "events",
    "with_flow",
    "share_creation",
    "share_redemption",
    "median_forward_rel_change",
)


def forward_share_response(
    band: pd.DataFrame, flows: pd.DataFrame, horizon: int = 5
) -> pd.DataFrame:
    """Forward relative share change after every band row with a side."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    frames: list[pd.DataFrame] = []
    rows = band.loc[band["side"].notna(), list(RESPONSE_COLUMNS[:5])]
    for ticker, fund in flows.sort_values(["ticker", "date"]).groupby(
        "ticker", sort=True, observed=True
    ):
        so = (
            fund["shares_outstanding"]
            .astype("Float64")
            .to_numpy(dtype=np.float64, na_value=np.nan)
        )
        broken = (fund["status"] == "split_session").to_numpy()
        n = len(fund)
        forward = np.full(n, np.nan)
        for i in range(n - horizon):
            if not broken[i + 1 : i + horizon + 1].any():
                forward[i] = so[i + horizon] / so[i] - 1.0
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": fund["date"].to_numpy(),
                    "forward_rel_change": forward,
                }
            )
        )
    lookup = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["ticker", "date", "forward_rel_change"])
    )
    out = rows.merge(lookup, on=["ticker", "date"], how="left", validate="1:1")
    return out.loc[:, list(RESPONSE_COLUMNS)]


def summarize_response(response: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """Creation/redemption frequencies by fund group and band side."""
    frame = response.assign(group=response["ticker"].astype(str).map(groups))
    if frame["group"].isna().any():
        raise ValueError("Every ticker needs a group.")
    records: list[dict[str, object]] = []
    for (group, side), rows in frame.groupby(["group", "side"], sort=True):
        change = rows["forward_rel_change"].dropna()
        records.append(
            {
                "group": group,
                "side": side,
                "events": len(rows),
                "with_flow": len(change),
                "share_creation": float((change > 0).mean()) if len(change) else None,
                "share_redemption": float((change < 0).mean()) if len(change) else None,
                "median_forward_rel_change": float(change.median())
                if len(change)
                else None,
            }
        )
    return pd.DataFrame.from_records(records, columns=list(SUMMARY_COLUMNS))
