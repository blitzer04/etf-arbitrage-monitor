from __future__ import annotations

from pathlib import Path

from etf_monitor.bundles.bundle import verify_bundle
from etf_monitor.bundles.schemas import ARTIFACT_SCHEMAS
from tests.app.helpers import write_fictional_bundle

_PANDAS_DTYPES = {
    "string": "string",
    "date": "datetime64[ns]",
    "float64": "Float64",
    "int64": "int64",
    "Int64": "Int64",
    "bool": "bool",
    "timestamp[ns]": "datetime64[ns]",
    "timestamp[ns, UTC]": "datetime64[ns, UTC]",
    "list<date>": "object",
}


def test_frame_returns_contract_dtypes_and_independent_mutable_copies(
    tmp_path: Path,
) -> None:
    bundle_path = write_fictional_bundle(tmp_path / "bundles", "2018-01-02")
    bundle = verify_bundle(bundle_path)

    for name, schema in ARTIFACT_SCHEMAS.items():
        frame = bundle.frame(name)
        assert [str(frame[field].dtype) for field, _ in schema.fields] == [
            _PANDAS_DTYPES[dtype] for _, dtype in schema.fields
        ]

    original_tickers = bundle.artifacts["coverage"].column("ticker").to_pylist()
    original_counts = (
        bundle.artifacts["coverage"].column("present_row_count").to_pylist()
    )
    first_frame = bundle.frame("coverage")
    first_frame.loc[0, "ticker"] = "MUTATED"
    first_frame.loc[0, "present_row_count"] = -1

    second_frame = bundle.frame("coverage")
    assert second_frame.loc[0, "ticker"] == original_tickers[0]
    assert second_frame.loc[0, "present_row_count"] == original_counts[0]
    assert bundle.artifacts["coverage"].column("ticker").to_pylist() == original_tickers
    assert (
        bundle.artifacts["coverage"].column("present_row_count").to_pylist()
        == original_counts
    )


def test_frame_rejects_unknown_artifact_name(tmp_path: Path) -> None:
    bundle_path = write_fictional_bundle(tmp_path / "bundles", "2018-01-02")
    bundle = verify_bundle(bundle_path)

    try:
        bundle.frame("unknown")
    except KeyError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("Unknown artifact names must raise KeyError.")
