"""Descriptive, non-causal comparisons of verified evaluation vintages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from etf_monitor.bundles.bundle import (
    BundleIncompatibleError,
    VerifiedBundle,
    _mapping,
    _table_to_frame,
)
from etf_monitor.bundles.schemas import ARTIFACT_SCHEMAS


@dataclass(frozen=True, slots=True)
class VintageComparison:
    """Hold descriptive input, signal, and target-session change counts."""

    input_row_counts: pd.DataFrame
    input_field_changes: pd.DataFrame
    input_changes_by_ticker: pd.DataFrame
    signal_changes_by_ticker: pd.DataFrame
    target_session_changes: pd.DataFrame


def _equal(first: object, second: object) -> bool:
    if pd.isna(cast(Any, first)) and pd.isna(cast(Any, second)):
        return True
    return bool(first == second)


def _frame(bundle: VerifiedBundle, name: str) -> pd.DataFrame:
    return _table_to_frame(bundle.artifacts[name], ARTIFACT_SCHEMAS[name])


def _signal_changes(
    first: pd.DataFrame,
    second: pd.DataFrame,
    signal: str,
    target_a: pd.Timestamp,
    target_b: pd.Timestamp,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    value_columns = {
        "momentum": ("raw_log_return", "simple_return_pct", "percentile", "status"),
        "volatility": (
            "raw_annualized_vol",
            "display_vol_pct",
            "percentile",
            "status",
        ),
    }[signal]
    key = ["ticker", "signal_date"]
    first_rows = first.set_index(key)
    second_rows = second.set_index(key)
    all_keys = sorted(set(first_rows.index) | set(second_rows.index))
    changes: dict[str, dict[str, int]] = {}
    target_changes: list[dict[str, object]] = []
    target_first = first.loc[first["signal_date"].eq(target_a)].set_index("ticker")
    target_second = second.loc[second["signal_date"].eq(target_b)].set_index("ticker")
    raw_column, display_column = value_columns[:2]
    for ticker in sorted(set(target_first.index) | set(target_second.index)):
        if ticker not in target_first.index or ticker not in target_second.index:
            target_changes.append(
                {
                    "ticker": ticker,
                    "signal": signal,
                    "signal_date_a": target_a,
                    "signal_date_b": target_b,
                    "raw_changed": True,
                    "display_changed": True,
                    "percentile_changed": True,
                    "status_changed": True,
                    "normalization_status_changed": True,
                    "reference_count_changed": True,
                }
            )
            continue
        left_target = target_first.loc[ticker]
        right_target = target_second.loc[ticker]
        target_changes.append(
            {
                "ticker": ticker,
                "signal": signal,
                "signal_date_a": target_a,
                "signal_date_b": target_b,
                "raw_changed": not _equal(
                    left_target[raw_column], right_target[raw_column]
                ),
                "display_changed": not _equal(
                    left_target[display_column], right_target[display_column]
                ),
                "percentile_changed": not _equal(
                    left_target["percentile"], right_target["percentile"]
                ),
                "status_changed": not _equal(
                    left_target["status"], right_target["status"]
                ),
                "normalization_status_changed": not _equal(
                    left_target["normalization_status"],
                    right_target["normalization_status"],
                ),
                "reference_count_changed": not _equal(
                    left_target["reference_count"], right_target["reference_count"]
                ),
            }
        )
    for ticker, signal_date in all_keys:
        left_exists = (ticker, signal_date) in first_rows.index
        right_exists = (ticker, signal_date) in second_rows.index
        if not left_exists or not right_exists:
            row = first_rows.loc[(ticker, signal_date)] if left_exists else None
            other = second_rows.loc[(ticker, signal_date)] if right_exists else None
            left = {} if row is None else row.to_dict()
            right = {} if other is None else other.to_dict()
            changed = {column: True for column in value_columns}
        else:
            left = first_rows.loc[(ticker, signal_date)].to_dict()
            right = second_rows.loc[(ticker, signal_date)].to_dict()
            changed = {
                column: not _equal(left[column], right[column])
                for column in value_columns
            }
        ticker_counts = changes.setdefault(
            str(ticker), {"raw_changed_count": 0, "percentile_changed_count": 0}
        )
        raw_column = value_columns[0]
        ticker_counts["raw_changed_count"] += int(changed[raw_column])
        ticker_counts["percentile_changed_count"] += int(changed["percentile"])
    rows = [
        {"ticker": ticker, "signal": signal, **counts}
        for ticker, counts in changes.items()
    ]
    return rows, target_changes


def compare_bundles(a: VerifiedBundle, b: VerifiedBundle) -> VintageComparison:
    """Compare two verified bundles without inferring the cause of changes.

    Args:
        a: First verified bundle.
        b: Second verified bundle.

    Returns:
        Input row/field counts, per-ticker changes, signal changes, and
        target-session changes. All outputs are descriptive only.

    Raises:
        TypeError: If either argument is not a ``VerifiedBundle``.
        BundleIncompatibleError: If universe or methodology metadata differ.
    """
    if not isinstance(a, VerifiedBundle) or not isinstance(b, VerifiedBundle):
        raise TypeError("compare_bundles requires two verified bundles.")
    manifest_a = cast(dict[str, object], a.manifest)
    manifest_b = cast(dict[str, object], b.manifest)
    for key in ("universe", "methodology"):
        if _mapping(manifest_a[key], key) != _mapping(manifest_b[key], key):
            raise BundleIncompatibleError(f"Bundles use different {key} metadata.")

    input_a = _frame(a, "input_prices").set_index(["ticker", "date"])
    input_b = _frame(b, "input_prices").set_index(["ticker", "date"])
    keys_a = set(input_a.index)
    keys_b = set(input_b.index)
    overlap = sorted(keys_a & keys_b)
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    fields = [name for name in input_a.columns if name not in {"ticker", "date"}]
    field_counts = {field: 0 for field in fields}
    ticker_counts: dict[str, dict[str, int]] = {}
    for ticker in sorted({str(key[0]) for key in keys_a | keys_b}):
        ticker_counts[ticker] = {
            "rows_only_in_a": sum(key[0] == ticker for key in only_a),
            "rows_only_in_b": sum(key[0] == ticker for key in only_b),
            "overlapping_rows": sum(key[0] == ticker for key in overlap),
            "changed_field_count": 0,
        }
    for key in overlap:
        ticker = str(key[0])
        for field in fields:
            if not _equal(input_a.loc[key, field], input_b.loc[key, field]):
                field_counts[field] += 1
                ticker_counts[ticker]["changed_field_count"] += 1
    field_frame = pd.DataFrame(
        [
            {"field": name, "changed_count": count}
            for name, count in field_counts.items()
        ]
    )
    ticker_frame = pd.DataFrame(
        [{"ticker": ticker, **counts} for ticker, counts in ticker_counts.items()]
    )
    row_frame = pd.DataFrame(
        [
            {
                "rows_only_in_a": len(only_a),
                "rows_only_in_b": len(only_b),
                "overlapping_rows": len(overlap),
            }
        ]
    )

    signal_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    target_a = pd.Timestamp(str(manifest_a["target_session"]))
    target_b = pd.Timestamp(str(manifest_b["target_session"]))
    for signal in ("momentum", "volatility"):
        changes, targets = _signal_changes(
            _frame(a, signal),
            _frame(b, signal),
            signal,
            target_a,
            target_b,
        )
        signal_rows.extend(changes)
        target_rows.extend(targets)
    return VintageComparison(
        input_row_counts=row_frame,
        input_field_changes=field_frame,
        input_changes_by_ticker=ticker_frame,
        signal_changes_by_ticker=pd.DataFrame(signal_rows),
        target_session_changes=pd.DataFrame(target_rows),
    )
