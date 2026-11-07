"""Canonical daily ETF price columns, dtypes, and validation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype

from etf_monitor.numeric import (
    NumericConversionError,
    price_series_float64,
    volume_series_int64,
)
from etf_monitor.universe import Universe, load_universe
from etf_monitor.xnys import xnys_sessions

CANONICAL_COLUMNS = (
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "retrieved_at",
    "source",
)
MARKET_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)
PRICE_COLUMNS = ("open", "high", "low", "close", "adjusted_close")
PRICE_SOURCE = "yahoo_chart_v8"
ALLOWED_SOURCES = ("yahoo_chart_v8", "FICTIONAL_synthetic")


class PriceValidationError(ValueError):
    """Indicate that a frame violates the canonical daily-price contract."""


def empty_canonical_frame() -> pd.DataFrame:
    """Return an empty DataFrame with exact canonical pandas dtypes."""
    return pd.DataFrame(
        {
            "ticker": pd.Series(dtype="string"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "adjusted_close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="Int64"),
            "retrieved_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "source": pd.Series(dtype="string"),
        },
        columns=CANONICAL_COLUMNS,
    )


def _string_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if not (
        isinstance(values.dtype, pd.StringDtype)
        or values.dtype == object
        or values.dtype == "str"
    ):
        raise PriceValidationError(f"Column '{column}' must contain strings.")
    if any(not isinstance(value, str) for value in values.array):
        raise PriceValidationError(f"Column '{column}' must contain only strings.")
    return values.astype("string")


def canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert supported source representations to the exact canonical schema.

    Numeric conversion is performed value-by-value so pandas inference cannot
    silently round large integers or change nulls.

    Args:
        frame: Candidate DataFrame containing exactly the canonical columns.

    Returns:
        A sorted copy with canonical column order and dtypes.

    Raises:
        PriceValidationError: If the frame's schema or any conversion is invalid.
    """
    duplicate_columns = frame.columns[frame.columns.duplicated()].unique().tolist()
    if duplicate_columns:
        raise PriceValidationError(f"Duplicate column labels: {duplicate_columns}.")
    missing = [column for column in CANONICAL_COLUMNS if column not in frame]
    extra = [column for column in frame if column not in CANONICAL_COLUMNS]
    if missing or extra:
        raise PriceValidationError(
            f"Canonical columns mismatch; missing={missing}, unexpected={extra}."
        )

    result = frame.loc[:, list(CANONICAL_COLUMNS)].copy()
    result["ticker"] = _string_column(result, "ticker")
    result["source"] = _string_column(result, "source")

    date_dtype = result["date"].dtype
    if isinstance(date_dtype, pd.DatetimeTZDtype) or not is_datetime64_dtype(
        date_dtype
    ):
        raise PriceValidationError("Column 'date' must be timezone-naive datetime64.")
    try:
        result["date"] = result["date"].astype("datetime64[ns]")
        retrieval_dtype = result["retrieved_at"].dtype
        if (
            not isinstance(retrieval_dtype, pd.DatetimeTZDtype)
            or str(retrieval_dtype.tz) != "UTC"
        ):
            raise PriceValidationError(
                "Column 'retrieved_at' must use timezone-aware UTC datetimes."
            )
        result["retrieved_at"] = result["retrieved_at"].astype("datetime64[ns, UTC]")
        for column in PRICE_COLUMNS:
            result[column] = price_series_float64(result[column])
        result["volume"] = volume_series_int64(result["volume"])
    except (NumericConversionError, OverflowError, TypeError, ValueError) as error:
        if isinstance(error, PriceValidationError):
            raise
        raise PriceValidationError(str(error)) from error

    result = result.sort_values(["ticker", "date"], kind="mergesort")
    return result.reset_index(drop=True)


def _key_examples(frame: pd.DataFrame, positions: list[int]) -> list[dict[str, str]]:
    return [
        {
            "ticker": str(frame.iloc[position]["ticker"]),
            "date": str(frame.iloc[position]["date"]),
        }
        for position in positions[:5]
    ]


def _is_missing(value: object) -> bool:
    missing = pd.isna(cast(Any, value))
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def validate_canonical(frame: pd.DataFrame, universe: Universe | None = None) -> None:
    """Validate canonical price values, identity, session dates, and ordering.

    Args:
        frame: Candidate frame already in canonical representation.
        universe: Universe whose tickers are allowed; defaults to the packaged
            crowding universe.

    Raises:
        PriceValidationError: If schema, identity, values, or key constraints fail.
    """
    if tuple(frame.columns) != CANONICAL_COLUMNS:
        raise PriceValidationError(
            f"Columns must be ordered exactly as {CANONICAL_COLUMNS}."
        )
    required_dtypes: dict[str, object] = {
        "ticker": pd.StringDtype(),
        "date": np.dtype("datetime64[ns]"),
        "open": np.dtype("float64"),
        "high": np.dtype("float64"),
        "low": np.dtype("float64"),
        "close": np.dtype("float64"),
        "adjusted_close": np.dtype("float64"),
        "volume": pd.Int64Dtype(),
        "retrieved_at": pd.DatetimeTZDtype(unit="ns", tz="UTC"),
        "source": pd.StringDtype(),
    }
    for column, dtype in required_dtypes.items():
        if frame[column].dtype != dtype:
            raise PriceValidationError(
                f"Column '{column}' must use dtype {dtype}, received "
                f"{frame[column].dtype}."
            )

    allowed_universe = load_universe() if universe is None else universe
    allowed_tickers = {etf.ticker for etf in allowed_universe.etfs}
    unknown_ticker_positions = [
        position
        for position, ticker in enumerate(frame["ticker"].array)
        if ticker not in allowed_tickers
    ]
    if unknown_ticker_positions:
        raise PriceValidationError(
            "Ticker is not in the configured universe at "
            f"{_key_examples(frame, unknown_ticker_positions)}."
        )
    sources = set(frame["source"].dropna().array)
    if frame["source"].isna().any() or not sources.issubset(ALLOWED_SOURCES):
        raise PriceValidationError(
            f"Column 'source' must use one of {ALLOWED_SOURCES!r}."
        )
    if len(sources) > 1:
        raise PriceValidationError("Canonical frames must contain a single source.")
    if (
        frame["date"].isna().any()
        or frame["date"].ne(frame["date"].dt.normalize()).any()
    ):
        raise PriceValidationError("Price dates must be non-null midnight labels.")
    if frame["retrieved_at"].isna().any():
        raise PriceValidationError("retrieved_at must never be null.")

    if frame.empty:
        invalid_sessions: list[int] = []
    else:
        sessions = xnys_sessions(frame["date"].min().date(), frame["date"].max().date())
        valid_session_mask = frame["date"].isin(sessions).to_numpy(dtype=np.bool_)
        invalid_sessions = np.flatnonzero(~valid_session_mask).tolist()
    if invalid_sessions:
        raise PriceValidationError(
            "Price dates must be XNYS sessions at "
            f"{_key_examples(frame, invalid_sessions)}."
        )

    duplicate_keys = frame.duplicated(["ticker", "date"], keep=False)
    if duplicate_keys.any():
        raise PriceValidationError(
            "Duplicate ticker/date keys at "
            f"{_key_examples(frame, np.flatnonzero(duplicate_keys.to_numpy()).tolist())}."
        )
    if not frame[["ticker", "date"]].equals(
        frame.sort_values(["ticker", "date"], kind="mergesort")[
            ["ticker", "date"]
        ].reset_index(drop=True)
    ):
        raise PriceValidationError("Canonical rows must be sorted by ticker and date.")

    invalid_rows: list[int] = []
    invalid_ohlc: list[int] = []
    invalid_volume: list[int] = []
    for position, row in enumerate(
        frame.loc[:, list(MARKET_COLUMNS)].itertuples(index=False, name=None)
    ):
        values = dict(zip(MARKET_COLUMNS, row, strict=True))
        if all(_is_missing(value) for value in values.values()):
            invalid_rows.append(position)
            continue
        for column in PRICE_COLUMNS:
            value = values[column]
            if not _is_missing(value) and (
                not np.isfinite(float(value)) or float(value) <= 0
            ):
                invalid_rows.append(position)
                break
        volume = values["volume"]
        if not _is_missing(volume) and int(volume) < 0:
            invalid_volume.append(position)

        present = {
            key: float(value)
            for key, value in values.items()
            if key in {"open", "high", "low", "close"} and not _is_missing(value)
        }
        low = present.get("low")
        high = present.get("high")
        if low is not None and high is not None and low > high:
            invalid_ohlc.append(position)
        if low is not None and "open" in present and not low <= present["open"]:
            invalid_ohlc.append(position)
        if high is not None and "open" in present and not present["open"] <= high:
            invalid_ohlc.append(position)
        if low is not None and "close" in present and not low <= present["close"]:
            invalid_ohlc.append(position)
        if high is not None and "close" in present and not present["close"] <= high:
            invalid_ohlc.append(position)

    if invalid_rows:
        raise PriceValidationError(
            "Rows must contain a market value and prices must be positive/finite; "
            f"invalid keys: {_key_examples(frame, sorted(set(invalid_rows)))}."
        )
    if invalid_ohlc:
        raise PriceValidationError(
            "OHLC relationship violation at "
            f"{_key_examples(frame, sorted(set(invalid_ohlc)))}."
        )
    if invalid_volume:
        raise PriceValidationError(
            "Volume must be non-negative at "
            f"{_key_examples(frame, sorted(set(invalid_volume)))}."
        )
