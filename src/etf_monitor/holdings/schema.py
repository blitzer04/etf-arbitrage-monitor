"""Canonical holdings lines for one or more funds and their validation.

One row per published holdings line. ``ticker`` is the ETF, ``as_of`` the
issuer's ``As of`` date (timezone-naive XNYS session), ``position`` the
1-based line order in the workbook (lines are not deduplicated: SSGA lists
the same security ticker or identifier on several lines, e.g. two listings or
two currency balances). Units: ``weight_pct`` is percent of fund net assets
as published; ``quantity`` is shares (equity layout; cash lines carry the
currency amount); ``par_value`` and ``market_value`` are in the issuer's
reporting units (fixed-income layout). Missing (``-``) cells are null.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime

import numpy as np
import pandas as pd

from etf_monitor.providers.ssga_holdings import SSGA_HOLDINGS_SOURCE, ParsedHoldings
from etf_monitor.xnys import is_session

FICTIONAL_HOLDINGS_SOURCE = "FICTIONAL_synthetic"
HOLDINGS_ALLOWED_SOURCES = (SSGA_HOLDINGS_SOURCE, FICTIONAL_HOLDINGS_SOURCE)
_DTYPES: dict[str, object] = {
    "ticker": pd.StringDtype(),
    "as_of": np.dtype("datetime64[ns]"),
    "position": np.dtype("int64"),
    "layout": pd.StringDtype(),
    "name": pd.StringDtype(),
    "security_ticker": pd.StringDtype(),
    "identifier": pd.StringDtype(),
    "sedol": pd.StringDtype(),
    "isin": pd.StringDtype(),
    "weight_pct": np.dtype("float64"),
    "quantity": np.dtype("float64"),
    "par_value": np.dtype("float64"),
    "market_value": np.dtype("float64"),
    "coupon": np.dtype("float64"),
    "maturity": np.dtype("datetime64[ns]"),
    "local_currency": pd.StringDtype(),
    "retrieved_at": pd.DatetimeTZDtype(unit="ns", tz="UTC"),
    "source": pd.StringDtype(),
}
HOLDINGS_COLUMNS = tuple(_DTYPES)
_FLOAT_COLUMNS = ("weight_pct", "quantity", "par_value", "market_value", "coupon")


class HoldingsValidationError(ValueError):
    """Indicate that a holdings frame violates the canonical contract."""


def empty_holdings_frame() -> pd.DataFrame:
    """Return an empty DataFrame with exact canonical holdings dtypes."""
    return pd.DataFrame(
        {
            column: pd.Series(dtype=dtype)  # type: ignore[call-overload]
            for column, dtype in _DTYPES.items()
        }
    )


def holdings_frame(
    parsed: ParsedHoldings, retrieved_at: datetime, source: str
) -> pd.DataFrame:
    """Convert one parsed workbook into canonical rows without loss.

    Raises:
        HoldingsValidationError: If ``retrieved_at`` is not timezone-aware.
    """
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise HoldingsValidationError("retrieved_at must be timezone-aware UTC.")
    lines = parsed.holdings
    count = len(lines)

    def strings(values: list[str | None]) -> pd.arrays.StringArray:
        return pd.array(values, dtype="string")  # type: ignore[return-value]

    def floats(values: list[float | None]) -> np.ndarray:
        return np.array(
            [np.nan if value is None else value for value in values], dtype=np.float64
        )

    frame = pd.DataFrame(
        {
            "ticker": strings([parsed.ticker] * count),
            "as_of": pd.to_datetime([parsed.as_of] * count).astype("datetime64[ns]"),
            "position": np.arange(1, count + 1, dtype=np.int64),
            "layout": strings([parsed.layout] * count),
            "name": strings([line.name for line in lines]),
            "security_ticker": strings([line.security_ticker for line in lines]),
            "identifier": strings([line.identifier for line in lines]),
            "sedol": strings([line.sedol for line in lines]),
            "isin": strings([line.isin for line in lines]),
            "weight_pct": floats([line.weight_pct for line in lines]),
            "quantity": floats([line.quantity for line in lines]),
            "par_value": floats([line.par_value for line in lines]),
            "market_value": floats([line.market_value for line in lines]),
            "coupon": floats([line.coupon for line in lines]),
            "maturity": pd.to_datetime(
                [line.maturity for line in lines]  # type: ignore[arg-type]
            ).astype("datetime64[ns]"),
            "local_currency": strings([line.local_currency for line in lines]),
            "retrieved_at": pd.Series(
                [pd.Timestamp(retrieved_at).tz_convert("UTC")] * count,
                dtype="datetime64[ns, UTC]",
            ),
            "source": strings([source] * count),
        }
    )
    return frame


def validate_holdings_frame(
    frame: pd.DataFrame, allowed_tickers: Collection[str]
) -> None:
    """Validate schema, identity, keys, ordering, and finite values.

    Raises:
        HoldingsValidationError: If any contract rule fails.
    """
    if tuple(frame.columns) != HOLDINGS_COLUMNS:
        raise HoldingsValidationError(
            f"Columns must be ordered exactly as {HOLDINGS_COLUMNS}."
        )
    for column, dtype in _DTYPES.items():
        if frame[column].dtype != dtype:
            raise HoldingsValidationError(
                f"Column '{column}' must use dtype {dtype}, received "
                f"{frame[column].dtype}."
            )
    if frame["ticker"].isna().any() or not set(frame["ticker"].array).issubset(
        set(allowed_tickers)
    ):
        raise HoldingsValidationError("Holdings contain a ticker outside the universe.")
    if frame["source"].isna().any() or not set(frame["source"].array).issubset(
        HOLDINGS_ALLOWED_SOURCES
    ):
        raise HoldingsValidationError(
            f"Column 'source' must use one of {HOLDINGS_ALLOWED_SOURCES}."
        )
    for column in ("as_of", "layout", "name", "identifier", "retrieved_at"):
        if frame[column].isna().any():
            raise HoldingsValidationError(f"Column '{column}' must never be null.")
    if frame.empty:
        return
    if (
        frame.groupby("ticker")[["as_of", "layout", "retrieved_at"]]
        .nunique()
        .ne(1)
        .any(axis=None)
    ):
        raise HoldingsValidationError(
            "Each fund must have one as_of, layout, and retrieved_at."
        )
    if not all(is_session(day.date()) for day in frame["as_of"].unique()):
        raise HoldingsValidationError("as_of dates must be XNYS sessions.")
    if frame.duplicated(["ticker", "position"]).any():
        raise HoldingsValidationError("Duplicate ticker/position keys.")
    ordered = frame.sort_values(["ticker", "position"], kind="mergesort")
    if not ordered.index.equals(frame.index) or not frame.index.equals(
        pd.RangeIndex(len(frame))
    ):
        raise HoldingsValidationError("Rows must be sorted by ticker and position.")
    for column in _FLOAT_COLUMNS:
        values = frame[column].dropna().to_numpy()
        if not np.isfinite(values).all():
            raise HoldingsValidationError(f"Column '{column}' must be finite.")
