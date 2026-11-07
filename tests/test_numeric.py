"""Lossless numeric dtype acceptance and rejection matrix."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from etf_monitor.numeric import NumericConversionError, price_scalar, volume_scalar
from etf_monitor.prices.schema import PriceValidationError, canonicalize_frame
from tests.prices.helpers import canonical_frame


@pytest.mark.parametrize(
    ("column", "values", "dtype", "expected_dtype", "expected_value"),
    [
        ("open", [1.5], np.dtype("float32"), np.dtype("float64"), 1.5),
        ("open", [1.5], np.dtype("float64"), np.dtype("float64"), 1.5),
        ("open", [np.nan], np.dtype("float64"), np.dtype("float64"), np.nan),
        ("open", [2**53], np.dtype("int64"), np.dtype("float64"), float(2**53)),
        ("open", [2**53], np.dtype("uint64"), np.dtype("float64"), float(2**53)),
        ("open", [pd.NA], "Float64", np.dtype("float64"), np.nan),
        ("open", [2**53], "Int64", np.dtype("float64"), float(2**53)),
        ("open", [2**53], "UInt64", np.dtype("float64"), float(2**53)),
        (
            "open",
            [1.5],
            pd.ArrowDtype(pa.float64()),
            np.dtype("float64"),
            1.5,
        ),
        (
            "open",
            [pd.NA],
            pd.ArrowDtype(pa.float32()),
            np.dtype("float64"),
            np.nan,
        ),
        (
            "open",
            [1.5],
            pd.ArrowDtype(pa.float32()),
            np.dtype("float64"),
            1.5,
        ),
        (
            "open",
            [2**53],
            pd.ArrowDtype(pa.int64()),
            np.dtype("float64"),
            float(2**53),
        ),
        (
            "open",
            [pd.NA],
            pd.ArrowDtype(pa.int64()),
            np.dtype("float64"),
            np.nan,
        ),
        (
            "open",
            [2**53],
            pd.ArrowDtype(pa.uint64()),
            np.dtype("float64"),
            float(2**53),
        ),
        (
            "open",
            [pd.NA],
            pd.ArrowDtype(pa.uint64()),
            np.dtype("float64"),
            np.nan,
        ),
        ("volume", [0], np.dtype("int64"), pd.Int64Dtype(), 0),
        ("volume", [2**53], np.dtype("uint64"), pd.Int64Dtype(), 2**53),
        ("volume", [1.0], np.dtype("float64"), pd.Int64Dtype(), 1),
        ("volume", [pd.NA], "Int64", pd.Int64Dtype(), pd.NA),
        ("volume", [pd.NA], "UInt64", pd.Int64Dtype(), pd.NA),
        (
            "volume",
            [2**53 + 1],
            pd.ArrowDtype(pa.uint64()),
            pd.Int64Dtype(),
            2**53 + 1,
        ),
        (
            "volume",
            [pd.NA],
            pd.ArrowDtype(pa.int64()),
            pd.Int64Dtype(),
            pd.NA,
        ),
        (
            "volume",
            [pd.NA],
            pd.ArrowDtype(pa.uint64()),
            pd.Int64Dtype(),
            pd.NA,
        ),
        (
            "volume",
            [1.0],
            pd.ArrowDtype(pa.float64()),
            pd.Int64Dtype(),
            1,
        ),
        (
            "volume",
            [1.0],
            pd.ArrowDtype(pa.float32()),
            pd.Int64Dtype(),
            1,
        ),
    ],
)
def test_supported_numpy_pandas_and_arrow_dtypes_convert_exactly(
    column: str,
    values: list[object],
    dtype: Any,
    expected_dtype: Any,
    expected_value: object,
) -> None:
    frame = canonical_frame()
    frame[column] = pd.Series(values, dtype=dtype)

    result = canonicalize_frame(frame)

    assert result[column].dtype == expected_dtype
    actual = result[column].iloc[0]
    if pd.isna(cast(Any, expected_value)):
        assert pd.isna(cast(Any, actual))
    else:
        assert actual == expected_value


@pytest.mark.parametrize(
    "dtype",
    [
        object,
        "string",
        "bool",
        "category",
        "Sparse[float64]",
        "complex128",
        pd.ArrowDtype(pa.decimal128(10, 2)),
    ],
)
def test_unsupported_numeric_dtypes_are_rejected(dtype: Any) -> None:
    frame = canonical_frame()
    frame["open"] = pd.Series([100], dtype=dtype)

    with pytest.raises(PriceValidationError):
        canonicalize_frame(frame)


def test_large_integer_price_that_cannot_round_trip_is_rejected() -> None:
    frame = canonical_frame()
    frame["open"] = pd.Series([2**53 + 1], dtype="Int64")

    with pytest.raises(PriceValidationError, match="exactly representable"):
        canonicalize_frame(frame)


def test_unsigned_volume_above_int64_is_rejected() -> None:
    frame = canonical_frame()
    frame["volume"] = pd.Series([2**63 + 1], dtype="UInt64")

    with pytest.raises(PriceValidationError, match="Int64 range"):
        canonicalize_frame(frame)


def test_non_integral_float_volume_is_rejected() -> None:
    frame = canonical_frame()
    frame["volume"] = pd.Series([1.5], dtype="float64")

    with pytest.raises(PriceValidationError, match="must be an integer"):
        canonicalize_frame(frame)


def test_float_volume_boundary_and_large_integer_are_distinguished() -> None:
    assert volume_scalar(2.0**53) == 2**53
    assert volume_scalar(2**53 + 1) == 2**53 + 1
    with pytest.raises(NumericConversionError) as error:
        volume_scalar(2.0**53 + 2)
    assert error.value.reason_code == "ambiguous_float_volume"


def test_extended_float_precision_is_not_silently_discarded() -> None:
    extended_price = np.longdouble("1.0000000000000000001")
    extended_volume = np.longdouble("9007199254740992.5")
    if extended_price == np.longdouble(float(extended_price)):
        pytest.skip("long double has no precision beyond float64 on this platform")

    with pytest.raises(NumericConversionError, match="exactly representable"):
        price_scalar(extended_price)
    with pytest.raises(NumericConversionError) as error:
        volume_scalar(extended_volume)
    assert error.value.reason_code == "ambiguous_float_volume"


def test_mixed_nullable_and_arrow_columns_canonicalize_together() -> None:
    frame = canonical_frame(adjusted_close=None)
    frame["open"] = pd.Series([pd.NA], dtype=pd.ArrowDtype(pa.float64()))
    frame["volume"] = pd.Series([2**53 + 1], dtype="UInt64")

    result = canonicalize_frame(frame)

    assert pd.isna(result["open"].iloc[0])
    assert result["volume"].iloc[0] == 2**53 + 1
    assert result["open"].dtype == np.dtype("float64")
    assert result["volume"].dtype == pd.Int64Dtype()
