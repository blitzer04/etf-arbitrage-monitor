from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest

from etf_monitor.bundles import (
    BundleIncompatibleError,
    compare_bundles,
    verify_bundle,
    write_bundle,
)
from etf_monitor.evaluation import evaluate
from tests.prices.helpers import canonical_frame

INSTANT = datetime.fromisoformat("2018-01-04T22:00:00+00:00")


def _revision_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    original_rows = [
        canonical_frame(
            ticker="SPY",
            day=day,
            adjusted_close=adjusted_close,
            volume=volume,
            source="FICTIONAL_synthetic",
        )
        for day, adjusted_close, volume in (
            ("2018-01-02", 100.0, 1_000),
            ("2018-01-03", 101.0, 1_001),
            ("2018-01-04", 102.0, 1_002),
        )
    ]
    revised_rows = [
        canonical_frame(
            ticker="SPY",
            day=day,
            adjusted_close=adjusted_close + 1.0,
            volume=volume + (1 if day == "2018-01-03" else 0),
            source="FICTIONAL_synthetic",
        )
        for day, adjusted_close, volume in (
            ("2018-01-02", 100.0, 1_000),
            ("2018-01-03", 101.0, 1_001),
            ("2018-01-04", 102.0, 1_002),
        )
    ]
    revised_rows.append(
        canonical_frame(
            ticker="SPY",
            day="2018-01-05",
            adjusted_close=104.0,
            volume=1_003,
            source="FICTIONAL_synthetic",
        )
    )

    def combine(rows: list[pd.DataFrame]) -> pd.DataFrame:
        return (
            pd.concat(rows, ignore_index=True)
            .sort_values(["ticker", "date"], kind="mergesort")
            .reset_index(drop=True)
        )

    return combine(original_rows), combine(revised_rows)


def test_revision_comparison_reports_exact_row_field_and_signal_counts(
    tmp_path: Path,
) -> None:
    first_prices, second_prices = _revision_prices()
    first_bundle = write_bundle(evaluate(first_prices, INSTANT), tmp_path / "a")
    second_bundle = write_bundle(evaluate(second_prices, INSTANT), tmp_path / "b")

    comparison = compare_bundles(
        verify_bundle(first_bundle),
        verify_bundle(second_bundle),
    )

    assert comparison.input_row_counts.iloc[0].to_dict() == {
        "rows_only_in_a": 0,
        "rows_only_in_b": 1,
        "overlapping_rows": 3,
    }
    fields = comparison.input_field_changes.set_index("field")["changed_count"]
    assert fields["adjusted_close"] == 3
    assert fields["volume"] == 1
    assert int(fields.drop(["adjusted_close", "volume"]).sum()) == 0
    ticker = comparison.input_changes_by_ticker.set_index("ticker").loc["SPY"]
    assert ticker["rows_only_in_b"] == 1
    assert ticker["overlapping_rows"] == 3
    assert ticker["changed_field_count"] == 4
    spy_signals = comparison.signal_changes_by_ticker.loc[
        comparison.signal_changes_by_ticker["ticker"].eq("SPY")
    ]
    assert (spy_signals["raw_changed_count"] == 0).all()
    assert (spy_signals["percentile_changed_count"] == 0).all()


def test_identical_bundle_comparison_is_zero_and_methodology_mismatch_refused(
    tmp_path: Path,
) -> None:
    prices, _ = _revision_prices()
    verified = verify_bundle(
        write_bundle(evaluate(prices, INSTANT), tmp_path / "bundle")
    )

    comparison = compare_bundles(verified, verified)
    assert comparison.input_row_counts.iloc[0].to_dict() == {
        "rows_only_in_a": 0,
        "rows_only_in_b": 0,
        "overlapping_rows": 3,
    }
    assert comparison.input_field_changes["changed_count"].sum() == 0
    assert comparison.signal_changes_by_ticker["raw_changed_count"].sum() == 0
    assert comparison.signal_changes_by_ticker["percentile_changed_count"].sum() == 0
    assert (
        not comparison.target_session_changes[
            [
                "raw_changed",
                "display_changed",
                "percentile_changed",
                "status_changed",
                "normalization_status_changed",
                "reference_count_changed",
            ]
        ]
        .to_numpy()
        .any()
    )

    changed_manifest = dict(verified.manifest)
    changed_manifest["methodology"] = MappingProxyType(
        {"version": "1.0.0", "sha256": "different"}
    )
    incompatible = replace(
        verified,
        manifest=MappingProxyType(changed_manifest),
    )
    with pytest.raises(BundleIncompatibleError, match="methodology"):
        compare_bundles(verified, incompatible)
