"""Canonical issuer NAV / shares-outstanding columns, dtypes, and validation.

Units: ``nav`` is U.S. dollars per share as published by the issuer;
``shares_outstanding`` is an exact share count (nullable ``Int64``; missing
means the issuer published ``-`` or nothing); ``total_net_assets`` is U.S.
dollars. ``date`` is the timezone-naive XNYS session the issuer labels the
NAV with (the NAV strike after that session's close). ``retrieved_at`` is
the UTC instant the workbook bytes were received.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime

import numpy as np
import pandas as pd

from etf_monitor.providers.ssga_nav import SSGA_NAV_SOURCE, ParsedNavHistory
from etf_monitor.xnys import xnys_sessions

NAV_COLUMNS = (
    "ticker",
    "date",
    "nav",
    "shares_outstanding",
    "total_net_assets",
    "retrieved_at",
    "source",
)
FICTIONAL_NAV_SOURCE = "FICTIONAL_synthetic"
NAV_ALLOWED_SOURCES = (SSGA_NAV_SOURCE, FICTIONAL_NAV_SOURCE)
_REQUIRED_DTYPES: dict[str, object] = {
    "ticker": pd.StringDtype(),
    "date": np.dtype("datetime64[ns]"),
    "nav": np.dtype("float64"),
    "shares_outstanding": pd.Int64Dtype(),
    "total_net_assets": np.dtype("float64"),
    "retrieved_at": pd.DatetimeTZDtype(unit="ns", tz="UTC"),
    "source": pd.StringDtype(),
}


class NavValidationError(ValueError):
    """Indicate that a NAV frame violates the canonical issuer NAV contract."""


def empty_nav_frame() -> pd.DataFrame:
    """Return an empty DataFrame with exact canonical NAV dtypes."""
    return pd.DataFrame(
        {
            column: pd.Series(dtype=dtype)  # type: ignore[call-overload]
            for column, dtype in _REQUIRED_DTYPES.items()
        }
    )


def nav_frame_from_history(
    history: ParsedNavHistory, retrieved_at: datetime, source: str
) -> pd.DataFrame:
    """Convert parsed issuer rows into canonical representation without loss.

    Args:
        history: Validated parser output for one ticker.
        retrieved_at: Timezone-aware UTC instant the payload was received.
        source: Approved source identifier recorded on every row.

    Returns:
        Canonical NAV frame for one ticker sorted by date.

    Raises:
        NavValidationError: If ``retrieved_at`` is not UTC-aware.
    """
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise NavValidationError("retrieved_at must be timezone-aware UTC.")
    rows = history.observations
    count = len(rows)
    frame = pd.DataFrame(
        {
            "ticker": pd.array([history.ticker] * count, dtype="string"),
            "date": pd.to_datetime([row.date for row in rows]).astype("datetime64[ns]"),
            "nav": np.array([row.nav for row in rows], dtype=np.float64),
            "shares_outstanding": pd.array(
                [row.shares_outstanding for row in rows], dtype="Int64"
            ),
            "total_net_assets": np.array(
                [
                    np.nan if row.total_net_assets is None else row.total_net_assets
                    for row in rows
                ],
                dtype=np.float64,
            ),
            "retrieved_at": pd.Series(
                [pd.Timestamp(retrieved_at).tz_convert("UTC")] * count,
                dtype="datetime64[ns, UTC]",
            ),
            "source": pd.array([source] * count, dtype="string"),
        }
    )
    for position, row in enumerate(rows):
        shares = frame["shares_outstanding"].iloc[position]
        if (row.shares_outstanding is None) != pd.isna(shares) or (
            row.shares_outstanding is not None and int(shares) != row.shares_outstanding
        ):
            raise NavValidationError(
                f"Shares outstanding for {history.ticker} on {row.date} changed "
                "during conversion."
            )
    return frame


def validate_nav_frame(frame: pd.DataFrame, allowed_tickers: Collection[str]) -> None:
    """Validate NAV schema, identity, session dates, keys, ordering, and values.

    Args:
        frame: Candidate frame already in canonical representation.
        allowed_tickers: Tickers permitted in this frame.

    Raises:
        NavValidationError: If any contract rule fails.
    """
    if tuple(frame.columns) != NAV_COLUMNS:
        raise NavValidationError(f"Columns must be ordered exactly as {NAV_COLUMNS}.")
    for column, dtype in _REQUIRED_DTYPES.items():
        if frame[column].dtype != dtype:
            raise NavValidationError(
                f"Column '{column}' must use dtype {dtype}, received "
                f"{frame[column].dtype}."
            )
    if frame["ticker"].isna().any() or not set(frame["ticker"].array).issubset(
        set(allowed_tickers)
    ):
        raise NavValidationError("NAV frame contains a ticker outside the universe.")
    sources = set(frame["source"].dropna().array)
    if frame["source"].isna().any() or not sources.issubset(NAV_ALLOWED_SOURCES):
        raise NavValidationError(
            f"Column 'source' must use one of {NAV_ALLOWED_SOURCES}."
        )
    if len(sources) > 1:
        raise NavValidationError("NAV frames must contain a single source.")
    if frame["retrieved_at"].isna().any():
        raise NavValidationError("retrieved_at must never be null.")
    if (
        frame["date"].isna().any()
        or frame["date"].ne(frame["date"].dt.normalize()).any()
    ):
        raise NavValidationError("NAV dates must be non-null midnight labels.")
    if frame.empty:
        return
    sessions = xnys_sessions(frame["date"].min().date(), frame["date"].max().date())
    off_session = frame.loc[~frame["date"].isin(sessions), ["ticker", "date"]]
    if not off_session.empty:
        raise NavValidationError(
            f"NAV dates must be XNYS sessions; first offender {off_session.iloc[0].tolist()}."
        )
    if frame.duplicated(["ticker", "date"]).any():
        raise NavValidationError("Duplicate ticker/date keys in NAV frame.")
    ordered = frame.sort_values(["ticker", "date"], kind="mergesort")
    if not ordered.index.equals(frame.index) or not frame.index.equals(
        pd.RangeIndex(len(frame))
    ):
        raise NavValidationError("NAV rows must be sorted by ticker and date.")
    nav = frame["nav"].to_numpy()
    if not np.isfinite(nav).all() or (nav <= 0).any():
        raise NavValidationError("NAV must be finite and positive on every row.")
    shares = frame["shares_outstanding"].dropna()
    if (shares < 0).any():
        raise NavValidationError("Shares outstanding must be non-negative.")
    assets = frame["total_net_assets"].dropna().to_numpy()
    if not np.isfinite(assets).all() or (assets < 0).any():
        raise NavValidationError("Total net assets must be finite and non-negative.")
