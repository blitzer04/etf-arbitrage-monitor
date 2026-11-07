from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from etf_monitor.bundles.content_hash import content_sha256
from etf_monitor.bundles.schemas import ArtifactSchema


def test_content_hash_is_order_independent_and_detects_one_ulp_changes() -> None:
    schema = ArtifactSchema(
        "sample",
        (("ticker", "string"), ("value", "float64")),
        ("ticker",),
    )
    frame = pd.DataFrame(
        {
            "ticker": pd.Series(["QQQ", "SPY"], dtype="string"),
            "value": np.array([2.0, 1.0], dtype=np.float64),
        }
    )
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    changed = frame.copy()
    changed.loc[1, "value"] = np.nextafter(1.0, 2.0)

    assert content_sha256(frame, schema) == content_sha256(shuffled, schema)
    assert content_sha256(frame, schema) != content_sha256(changed, schema)


def test_content_hash_distinguishes_float_nan_and_int64_null() -> None:
    schema = ArtifactSchema(
        "sample",
        (("key", "string"), ("value", "float64"), ("volume", "Int64")),
        ("key",),
    )
    nan_frame = pd.DataFrame(
        {
            "key": pd.Series(["row"], dtype="string"),
            "value": pd.Series([float("nan")], dtype="object"),
            "volume": pd.Series(pd.array([pd.NA], dtype="Int64")),
        }
    )
    null_float_frame = pd.DataFrame(
        {
            "key": pd.Series(["row"], dtype="string"),
            "value": pd.Series([None], dtype="object"),
            "volume": pd.Series(pd.array([pd.NA], dtype="Int64")),
        }
    )
    integer_value_frame = pd.DataFrame(
        {
            "key": pd.Series(["row"], dtype="string"),
            "value": pd.Series([float("nan")], dtype="object"),
            "volume": pd.Series(pd.array([0], dtype="Int64")),
        }
    )

    assert content_sha256(nan_frame, schema) != content_sha256(null_float_frame, schema)
    assert content_sha256(nan_frame, schema) != content_sha256(
        integer_value_frame, schema
    )


@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_content_hash_rejects_infinite_floats(value: float) -> None:
    schema = ArtifactSchema("sample", (("value", "float64"),), ("value",))
    frame = pd.DataFrame({"value": np.array([value], dtype=np.float64)})

    with pytest.raises(ValueError, match="infinite"):
        content_sha256(frame, schema)
