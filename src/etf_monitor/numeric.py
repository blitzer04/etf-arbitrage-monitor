"""Lossless numeric conversion helpers for provider and canonical values."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
)


class NumericConversionError(ValueError):
    """Indicate that a provider numeric value cannot be represented losslessly."""

    def __init__(self, message: str, reason_code: str = "non_numeric_value") -> None:
        super().__init__(message)
        self.reason_code = reason_code


def is_supported_numeric_dtype(dtype: ExtensionDtype | np.dtype[Any]) -> bool:
    """Return whether a dense integer or floating dtype is supported."""
    return not isinstance(dtype, pd.SparseDtype) and (
        is_integer_dtype(dtype) or is_float_dtype(dtype)
    )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    missing = pd.isna(cast(Any, value))
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def price_scalar(value: object) -> float | None:
    """Convert one provider price to float64 without changing its value."""
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise NumericConversionError("boolean is not a price value")
    if isinstance(value, (int, np.integer)):
        integer = int(value)
        try:
            converted = float(integer)
        except OverflowError as error:
            raise NumericConversionError(
                "integer price is outside float64 range"
            ) from error
        if not math.isfinite(converted) or int(converted) != integer:
            raise NumericConversionError("integer price is not exactly representable")
        return converted
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        if not math.isfinite(converted):
            raise NumericConversionError("price must be finite")
        if converted != value:
            raise NumericConversionError(
                "floating-point price is not exactly representable as float64"
            )
        return converted
    raise NumericConversionError(f"unsupported price value type {type(value).__name__}")


def volume_scalar(value: object) -> int | None:
    """Convert one provider volume to nullable signed Int64 without loss."""
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise NumericConversionError("boolean is not a volume value")
    if isinstance(value, (int, np.integer)):
        converted = int(value)
    elif isinstance(value, (float, np.floating)):
        if not bool(np.isfinite(value)):
            raise NumericConversionError("volume must be finite")
        if abs(cast(Any, value)) > 2**53:
            raise NumericConversionError(
                "floating-point volume is ambiguous above 2**53",
                "ambiguous_float_volume",
            )
        float_value = float(value)
        if not float_value.is_integer():
            raise NumericConversionError(
                "volume must be an integer", "non_integral_volume"
            )
        converted = int(value)
        if value != converted:
            raise NumericConversionError(
                "volume must be an exactly representable integer",
                "non_integral_volume",
            )
    else:
        raise NumericConversionError(
            f"unsupported volume value type {type(value).__name__}"
        )

    if converted < 0 or converted > 2**63 - 1:
        raise NumericConversionError(
            "volume is outside the non-negative Int64 range", "volume_out_of_range"
        )
    return converted


def price_series_float64(values: pd.Series) -> pd.Series:
    """Convert a supported numeric Series to float64 with exact values/nulls."""
    if not is_supported_numeric_dtype(values.dtype):
        raise NumericConversionError(
            f"unsupported price dtype {values.dtype}; expected dense real numeric"
        )
    converted = [price_scalar(value) for value in values.array]
    result = pd.Series(
        np.asarray(
            [np.nan if value is None else value for value in converted],
            dtype=np.float64,
        ),
        index=values.index,
        name=values.name,
        dtype=np.float64,
    )
    for original, normalized in zip(values.array, result.array, strict=True):
        if _is_missing(original):
            if not math.isnan(float(normalized)):
                raise NumericConversionError("price null state changed during casting")
        elif price_scalar(original) != float(normalized):
            raise NumericConversionError("price changed during float64 casting")
    return result


def volume_series_int64(values: pd.Series) -> pd.Series:
    """Convert a supported numeric Series to nullable Int64 exactly."""
    if not is_supported_numeric_dtype(values.dtype):
        raise NumericConversionError(
            f"unsupported volume dtype {values.dtype}; expected dense real numeric"
        )
    converted = [volume_scalar(value) for value in values.array]
    return pd.Series(
        pd.array(converted, dtype="Int64"),
        index=values.index,
        name=values.name,
    )
