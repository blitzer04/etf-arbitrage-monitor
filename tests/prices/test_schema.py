"""Canonical schema and value-integrity tests."""

from __future__ import annotations

import pandas as pd
import pytest

from etf_monitor.prices.schema import (
    ALLOWED_SOURCES,
    PriceValidationError,
    canonicalize_frame,
    validate_canonical,
)
from tests.prices.helpers import canonical_frame


def test_canonical_frame_uses_exact_dtypes_and_preserves_zero_and_null() -> None:
    frame = canonical_frame(volume=0, adjusted_close=None)
    canonical = canonicalize_frame(frame)

    validate_canonical(canonical)
    assert canonical["volume"].iloc[0] == 0
    assert pd.isna(canonical["adjusted_close"].iloc[0])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"open_price": 0.0}, "positive"),
        ({"open_price": -1.0}, "positive"),
        ({"volume": -1}, "non-negative"),
        ({"low": 103.0}, "OHLC"),
        ({"close": 103.0}, "OHLC"),
        ({"open_price": 103.0}, "OHLC"),
    ],
)
def test_invalid_market_observations_are_rejected(
    changes: dict[str, object], message: str
) -> None:
    frame = canonical_frame(**changes)  # type: ignore[arg-type]

    with pytest.raises(PriceValidationError, match=message):
        validate_canonical(frame)


def test_all_null_market_row_is_rejected() -> None:
    frame = canonical_frame(
        open_price=float("nan"),
        high=float("nan"),
        low=float("nan"),
        close=float("nan"),
        adjusted_close=None,
        volume=None,
    )
    with pytest.raises(PriceValidationError, match="market value"):
        validate_canonical(frame)


def test_unknown_ticker_and_non_session_dates_are_rejected() -> None:
    unknown = canonicalize_frame(canonical_frame(ticker="FAKE"))
    with pytest.raises(PriceValidationError, match="configured universe"):
        validate_canonical(unknown)

    holiday = canonicalize_frame(canonical_frame(day="2024-07-04"))
    with pytest.raises(PriceValidationError, match="XNYS sessions"):
        validate_canonical(holiday)


def test_duplicate_canonical_keys_are_rejected() -> None:
    frame = pd.concat([canonical_frame(), canonical_frame()], ignore_index=True)
    with pytest.raises(PriceValidationError, match="Duplicate"):
        validate_canonical(frame)


def test_positive_low_volume_is_accepted_unchanged() -> None:
    frame = canonicalize_frame(canonical_frame(volume=1_200))

    validate_canonical(frame)
    assert frame["volume"].iloc[0] == 1_200


def test_sources_must_be_allowed_and_single_per_frame() -> None:
    fictional = canonicalize_frame(canonical_frame(source="FICTIONAL_synthetic"))
    validate_canonical(fictional)
    assert tuple(ALLOWED_SOURCES) == ("yahoo_chart_v8", "FICTIONAL_synthetic")

    mixed = pd.concat(
        [
            canonical_frame(day="2024-07-02"),
            canonical_frame(day="2024-07-03", source="FICTIONAL_synthetic"),
        ],
        ignore_index=True,
    )
    with pytest.raises(PriceValidationError, match="single source"):
        validate_canonical(canonicalize_frame(mixed))

    unknown = canonicalize_frame(canonical_frame(source="unknown"))
    with pytest.raises(PriceValidationError, match="one of"):
        validate_canonical(unknown)


def test_canonicalization_sorts_reversed_input_by_ticker_and_date() -> None:
    later = canonical_frame(day="2024-07-03")
    earlier = canonical_frame(day="2024-07-02")
    frame = pd.concat([later, earlier], ignore_index=True)

    canonical = canonicalize_frame(frame)

    assert canonical["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-07-02",
        "2024-07-03",
    ]
