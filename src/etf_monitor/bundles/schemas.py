"""Stable artifact schemas shared by hashing, writing, and verification."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from etf_monitor.prices.schema import CANONICAL_COLUMNS


@dataclass(frozen=True, slots=True)
class ArtifactSchema:
    """Declare ordered fields, logical dtypes, and content-hash sort keys."""

    name: str
    fields: tuple[tuple[str, str], ...]
    keys: tuple[str, ...]

    def manifest_schema(self) -> list[list[str]]:
        return [[name, dtype] for name, dtype in self.fields]

    def arrow_schema(self) -> pa.Schema:
        fields = [
            pa.field(name, _arrow_type(dtype), nullable=True)
            for name, dtype in self.fields
        ]
        return pa.schema(fields)


def _arrow_type(dtype: str) -> pa.DataType:
    types = {
        "string": pa.string(),
        "date": pa.date32(),
        "float64": pa.float64(),
        "int64": pa.int64(),
        "Int64": pa.int64(),
        "bool": pa.bool_(),
        "timestamp[ns]": pa.timestamp("ns"),
        "timestamp[ns, UTC]": pa.timestamp("ns", tz="UTC"),
        "list<date>": pa.list_(pa.date32()),
    }
    try:
        return types[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported artifact dtype {dtype!r}.") from error


_INPUT_FIELDS = (
    ("ticker", "string"),
    ("date", "date"),
    ("open", "float64"),
    ("high", "float64"),
    ("low", "float64"),
    ("close", "float64"),
    ("adjusted_close", "float64"),
    ("volume", "Int64"),
    ("retrieved_at", "timestamp[ns, UTC]"),
    ("source", "string"),
)
_COVERAGE_FIELDS = (
    ("ticker", "string"),
    ("expected_session_count", "int64"),
    ("present_row_count", "int64"),
    ("missing_row_count", "int64"),
    ("missing_row_dates", "list<date>"),
    ("missing_adjusted_close_count", "int64"),
    ("missing_adjusted_close_dates", "list<date>"),
    ("rows_after_target", "int64"),
    ("first_row_date", "timestamp[ns]"),
    ("last_row_date", "timestamp[ns]"),
    ("last_eligible_adjusted_close_date", "timestamp[ns]"),
    ("price_staleness_sessions", "Int64"),
    ("target_row_present", "bool"),
    ("target_adjusted_close_present", "bool"),
    ("retrieved_at_min", "timestamp[ns, UTC]"),
    ("retrieved_at_max", "timestamp[ns, UTC]"),
    ("momentum_target_raw", "float64"),
    ("momentum_target_display", "float64"),
    ("momentum_target_percentile", "float64"),
    ("momentum_target_reference_count", "int64"),
    ("momentum_target_status", "string"),
    ("momentum_target_normalization_status", "string"),
    ("momentum_last_raw_eligible_date", "timestamp[ns]"),
    ("momentum_last_normalized_date", "timestamp[ns]"),
    ("momentum_raw_staleness_sessions", "Int64"),
    ("momentum_normalized_staleness_sessions", "Int64"),
    ("volatility_target_raw", "float64"),
    ("volatility_target_display", "float64"),
    ("volatility_target_percentile", "float64"),
    ("volatility_target_reference_count", "int64"),
    ("volatility_target_status", "string"),
    ("volatility_target_normalization_status", "string"),
    ("volatility_last_raw_eligible_date", "timestamp[ns]"),
    ("volatility_last_normalized_date", "timestamp[ns]"),
    ("volatility_raw_staleness_sessions", "Int64"),
    ("volatility_normalized_staleness_sessions", "Int64"),
)
_MOMENTUM_FIELDS = (
    ("ticker", "string"),
    ("signal_date", "timestamp[ns]"),
    ("first_use_session", "timestamp[ns]"),
    ("start_session", "timestamp[ns]"),
    ("start_adjusted_close", "float64"),
    ("end_adjusted_close", "float64"),
    ("raw_log_return", "float64"),
    ("simple_return_pct", "float64"),
    ("interior_missing_row_count", "int64"),
    ("interior_missing_adjclose_count", "int64"),
    ("status", "string"),
    ("percentile", "float64"),
    ("reference_count", "int64"),
    ("reference_first_session", "timestamp[ns]"),
    ("reference_last_session", "timestamp[ns]"),
    ("normalization_status", "string"),
)
_VOLATILITY_FIELDS = (
    ("ticker", "string"),
    ("signal_date", "timestamp[ns]"),
    ("first_use_session", "timestamp[ns]"),
    ("window_start_session", "timestamp[ns]"),
    ("window_end_session", "timestamp[ns]"),
    ("raw_annualized_vol", "float64"),
    ("display_vol_pct", "float64"),
    ("window_missing_row_count", "int64"),
    ("window_missing_adjclose_count", "int64"),
    ("status", "string"),
    ("percentile", "float64"),
    ("reference_count", "int64"),
    ("reference_first_session", "timestamp[ns]"),
    ("reference_last_session", "timestamp[ns]"),
    ("normalization_status", "string"),
)

ARTIFACT_SCHEMAS = {
    "input_prices": ArtifactSchema("input_prices", _INPUT_FIELDS, ("ticker", "date")),
    "coverage": ArtifactSchema("coverage", _COVERAGE_FIELDS, ("ticker",)),
    "momentum": ArtifactSchema("momentum", _MOMENTUM_FIELDS, ("ticker", "signal_date")),
    "volatility": ArtifactSchema(
        "volatility", _VOLATILITY_FIELDS, ("ticker", "signal_date")
    ),
    "dependence_cross_section": ArtifactSchema(
        "dependence_cross_section",
        (
            ("signal_date", "timestamp[ns]"),
            ("n_pairs", "int64"),
            ("pearson", "float64"),
            ("spearman", "float64"),
            ("status", "string"),
        ),
        ("signal_date",),
    ),
    "dependence_per_etf": ArtifactSchema(
        "dependence_per_etf",
        (
            ("ticker", "string"),
            ("n_pairs", "int64"),
            ("pearson", "float64"),
            ("spearman", "float64"),
            ("status", "string"),
        ),
        ("ticker",),
    ),
}

ARTIFACT_FILENAMES = {name: f"{name}.parquet" for name in ARTIFACT_SCHEMAS}

if tuple(name for name, _ in _INPUT_FIELDS) != CANONICAL_COLUMNS:
    raise RuntimeError("Bundle input-price schema drifted from Gate 2 canonical order.")
