"""Canonical content hashing independent of Parquet byte serialization."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from etf_monitor.bundles.schemas import ArtifactSchema


def _is_null(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    missing = pd.isna(cast(Any, value))
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _encode(value: object, dtype: str) -> str:
    if dtype == "float64":
        if value is None or value is pd.NA or value is pd.NaT:
            return r"\N"
        number = float(cast(Any, value))
        if math.isnan(number):
            return "nan"
        if not math.isfinite(number):
            raise ValueError("Content hashes reject infinite floats.")
        return number.hex()
    if _is_null(value):
        return r"\N"
    if dtype in {"Int64", "int64"}:
        return str(int(cast(Any, value)))
    if dtype == "bool":
        return "1" if bool(value) else "0"
    if dtype == "date":
        if isinstance(value, pd.Timestamp):
            return value.date().isoformat()
        if isinstance(value, datetime):
            return value.date().isoformat()
        if not isinstance(value, date):
            raise TypeError(f"Expected a date value, received {type(value).__name__}.")
        return value.isoformat()
    if dtype.startswith("timestamp"):
        timestamp = pd.Timestamp(cast(Any, value))
        if dtype == "timestamp[ns, UTC]":
            if timestamp.tz is None:
                raise ValueError("UTC timestamp fields must be timezone-aware.")
            return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")
        if timestamp.tz is not None:
            raise ValueError("Naive timestamp fields must not have a timezone.")
        return timestamp.isoformat()
    if dtype == "string":
        return json.dumps(str(value), ensure_ascii=False, separators=(",", ":"))
    if dtype == "list<date>":
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise TypeError("list<date> fields must contain date sequences.")
        dates = []
        for item in value:
            if isinstance(item, pd.Timestamp):
                dates.append(item.date().isoformat())
            elif isinstance(item, date):
                dates.append(item.isoformat())
            else:
                raise TypeError("list<date> fields may contain only dates.")
        return json.dumps(dates, separators=(",", ":"))
    raise ValueError(f"Unsupported content-hash dtype {dtype!r}.")


def _validate_value(value: object, dtype: str) -> None:
    if dtype not in {
        "float64",
        "Int64",
        "int64",
        "bool",
        "string",
        "date",
        "timestamp[ns]",
        "timestamp[ns, UTC]",
        "list<date>",
    }:
        raise ValueError(f"Unsupported content-hash dtype {dtype!r}.")
    if _is_null(value):
        if dtype == "float64" and isinstance(value, (float, np.floating)):
            if math.isnan(float(value)):
                return
        return
    if dtype == "float64":
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError("float64 content fields require numeric scalar values.")
        if not math.isfinite(float(value)):
            raise ValueError("Content hashes reject infinite floats.")
    elif dtype in {"Int64", "int64"}:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("Integer content fields require integer scalar values.")
        if not -(2**63) <= int(value) <= 2**63 - 1:
            raise ValueError("Integer content values must fit signed int64.")
    elif dtype == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("Boolean content fields require boolean scalar values.")
    elif dtype == "string":
        if not isinstance(value, str):
            raise TypeError("String content fields require string scalar values.")
    elif dtype == "date":
        if not isinstance(value, (date, datetime, pd.Timestamp)):
            raise TypeError("Date content fields require date scalar values.")
        if isinstance(value, datetime) and value.tzinfo is not None:
            raise ValueError("Date content fields must be timezone-naive.")
    elif dtype.startswith("timestamp"):
        if not isinstance(value, (datetime, pd.Timestamp)):
            raise TypeError("Timestamp content fields require datetime values.")
        timestamp = pd.Timestamp(cast(Any, value))
        if dtype == "timestamp[ns, UTC]" and timestamp.tz is None:
            raise ValueError("UTC timestamp fields must be timezone-aware.")
        if dtype == "timestamp[ns]" and timestamp.tz is not None:
            raise ValueError("Naive timestamp fields must not have a timezone.")
    elif dtype == "list<date>":
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise TypeError("list<date> fields must contain date sequences.")
        if any(
            not isinstance(item, (date, pd.Timestamp))
            or (isinstance(item, datetime) and not isinstance(item, pd.Timestamp))
            for item in value
        ):
            raise TypeError("list<date> fields may contain only dates.")
        if any(
            isinstance(item, pd.Timestamp) and item.tz is not None for item in value
        ):
            raise ValueError("list<date> values must be timezone-naive.")
    else:
        raise ValueError(f"Unsupported content-hash dtype {dtype!r}.")


def content_sha256(df: pd.DataFrame, schema: ArtifactSchema) -> str:
    """Hash typed rows in key order using canonical scalar encodings."""
    expected_columns = [name for name, _ in schema.fields]
    if list(df.columns) != expected_columns:
        raise ValueError(f"{schema.name} columns do not match its content schema.")
    if not schema.keys or any(key not in df for key in schema.keys):
        raise ValueError(f"{schema.name} is missing a declared content key.")
    if len(set(expected_columns)) != len(expected_columns):
        raise ValueError(f"{schema.name} content fields must be unique.")
    for key in schema.keys:
        for value in df[key].array:
            if _is_null(value) or isinstance(value, (list, tuple, dict, np.ndarray)):
                raise ValueError(f"{schema.name} key {key!r} has an invalid value.")
    for name, dtype in schema.fields:
        for value in df[name].array:
            _validate_value(value, dtype)
    if df.duplicated(list(schema.keys)).any():
        raise ValueError(f"{schema.name} contains duplicate content keys.")
    ordered = df.sort_values(list(schema.keys), kind="mergesort").reset_index(drop=True)
    lines = ["\t".join(f"{name}:{dtype}" for name, dtype in schema.fields)]
    for row in ordered.itertuples(index=False, name=None):
        lines.append(
            "\t".join(
                _encode(value, dtype)
                for value, (_, dtype) in zip(row, schema.fields, strict=True)
            )
        )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def file_sha256(payload: bytes) -> str:
    """Hash the exact bytes of one serialized artifact."""
    return hashlib.sha256(payload).hexdigest()
