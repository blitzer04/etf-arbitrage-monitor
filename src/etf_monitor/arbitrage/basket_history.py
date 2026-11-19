"""Historical holdings-basket gaps and ex-dividend receivable adjustments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from etf_monitor.arbitrage.basket import basket_fair_value, classify_lines
from etf_monitor.arbitrage.dividends import (
    DividendEvent,
    as_traded_dividends,
    dividends_on_share_basis,
    ex_dividend_value,
)
from etf_monitor.arbitrage.splits import SplitEvent
from etf_monitor.holdings.component_prices import ComponentVintage
from etf_monitor.holdings.store import HoldingsVintage
from etf_monitor.holdings.symbols import yahoo_symbol
from etf_monitor.xnys import next_session


@dataclass(frozen=True, slots=True)
class BasketHistorySnapshot:
    """Pair one verified holdings vintage with its component-price vintage."""

    holdings: HoldingsVintage
    components: ComponentVintage


def _dividend_union(
    snapshots: Sequence[BasketHistorySnapshot],
) -> tuple[DividendEvent, ...]:
    events: dict[tuple[str, date], tuple[DividendEvent, Path]] = {}
    for snapshot in snapshots:
        vintage_dividends = as_traded_dividends(
            snapshot.components.dividends,
            snapshot.components.splits,
        )
        for event in vintage_dividends:
            key = (event.ticker, event.ex_date)
            previous = events.get(key)
            if previous is not None and not math.isclose(
                previous[0].amount,
                event.amount,
                rel_tol=1e-9,
                abs_tol=0,
            ):
                raise ValueError(
                    f"Conflicting dividends for {event.ticker} on {event.ex_date} "
                    f"in {previous[1]} and {snapshot.components.path}."
                )
            events[key] = (event, snapshot.components.path)
    return tuple(events[key][0] for key in sorted(events))


def _split_union(
    snapshots: Sequence[BasketHistorySnapshot],
) -> tuple[SplitEvent, ...]:
    events: dict[tuple[str, date], tuple[SplitEvent, Path]] = {}
    for snapshot in snapshots:
        for event in snapshot.components.splits:
            key = (event.ticker, event.date)
            previous = events.get(key)
            if previous is not None and (
                previous[0].numerator * event.denominator
                != event.numerator * previous[0].denominator
            ):
                raise ValueError(
                    f"Conflicting splits for {event.ticker} on {event.date} "
                    f"in {previous[1]} and {snapshot.components.path}."
                )
            events[key] = (event, snapshot.components.path)
    return tuple(events[key][0] for key in sorted(events))


def _validate_unique_as_of(snapshots: Sequence[BasketHistorySnapshot]) -> None:
    labels: dict[tuple[str, date], Path] = {}
    for snapshot in snapshots:
        for ticker, fund in snapshot.holdings.frame.groupby("ticker", sort=True):
            as_of_values = pd.to_datetime(fund["as_of"]).dropna().unique()
            for value in as_of_values:
                label = pd.Timestamp(value).date()
                key = (str(ticker), label)
                previous = labels.get(key)
                if previous is not None:
                    raise ValueError(
                        f"Holdings vintages {previous} and {snapshot.holdings.path} "
                        f"have the same as_of label {label} for {ticker}."
                    )
                labels[key] = snapshot.holdings.path


def _finite_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _dividend_interval_covered(
    holdings: pd.DataFrame,
    components: ComponentVintage,
    after: pd.Timestamp,
    through: pd.Timestamp,
) -> bool:
    first_required = next_session(after).date()
    if components.start > first_required or components.end_exclusive <= through.date():
        return False
    priced_lines = holdings.loc[classify_lines(holdings).eq("priced").to_numpy()]
    priced_symbols = {
        symbol
        for value in priced_lines["security_ticker"]
        if (symbol := yahoo_symbol(str(value))) is not None
    }
    return priced_symbols.issubset(components.successful_symbols)


def _median_or_none(values: pd.Series) -> float | None:
    available = values.dropna()
    if available.empty:
        return None
    median = float(available.median())
    return median if math.isfinite(median) else None


def basket_history(
    snapshots: Sequence[BasketHistorySnapshot],
    nav: pd.DataFrame,
    *,
    nav_vintage: Path,
    universe_sha256: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute next-session basket gaps and ex-dividend adjustments over vintages.

    Each snapshot is valued with its own component prices and split events, as
    in ``basket-value``. Dividend events are converted to as-traded amounts
    using the splits from their own vintage before being unioned, then restated
    to the ending holdings' share basis for splits inside the interval. An
    interval without a complete component window or a successful priced symbol
    has missing ex-dividend and residual values and false dividend coverage.

    Args:
        snapshots: Ordered holdings/component vintage pairs.
        nav: Canonical issuer NAV rows covering all valuation dates.
        nav_vintage: Input NAV vintage path for the output summary.
        universe_sha256: Hash of the validated ETF universe.

    Returns:
        The sorted history frame and a JSON-serializable summary.

    Raises:
        ValueError: If no snapshots are supplied, snapshot labels repeat, or
            event vintages disagree for the same ticker and date.
    """
    if not snapshots:
        raise ValueError("At least one snapshot is required.")
    _validate_unique_as_of(snapshots)
    all_splits = _split_union(snapshots)
    dividends = _dividend_union(snapshots)

    history_parts: list[pd.DataFrame] = []
    snapshot_by_holdings_path: dict[str, BasketHistorySnapshot] = {}
    for snapshot in snapshots:
        valued = basket_fair_value(
            snapshot.holdings.frame,
            snapshot.components.frame,
            nav,
            snapshot.components.splits,
        )
        next_session_rows = valued.loc[
            valued["alignment"].eq("next_session").to_numpy()
        ].copy()
        next_session_rows["holdings_vintage"] = str(snapshot.holdings.path)
        history_parts.append(next_session_rows)
        snapshot_by_holdings_path[str(snapshot.holdings.path)] = snapshot

    history = (
        pd.concat(history_parts, ignore_index=True)
        .sort_values(["ticker", "valuation_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    for column in (
        "gap_change_bps",
        "ex_dividend_bps",
        "cash_change_bps",
        "residual_bps",
    ):
        history[column] = np.nan
    history["dividend_coverage"] = False

    intervals_without_dividend_coverage = 0
    for ticker, group_indices in history.groupby("ticker", sort=False).groups.items():
        ticker_name = str(ticker)
        indices = list(group_indices)
        for position in range(1, len(indices)):
            previous_index, current_index = indices[position - 1 : position + 1]
            previous = history.loc[previous_index]
            current = history.loc[current_index]
            snapshot = snapshot_by_holdings_path[str(current["holdings_vintage"])]
            fund_holdings = snapshot.holdings.frame.loc[
                snapshot.holdings.frame["ticker"].eq(ticker_name).to_numpy()
            ]
            has_dividend_coverage = _dividend_interval_covered(
                fund_holdings,
                snapshot.components,
                pd.Timestamp(previous["valuation_date"]),
                pd.Timestamp(current["valuation_date"]),
            )
            history.at[current_index, "dividend_coverage"] = has_dividend_coverage
            if not has_dividend_coverage:
                intervals_without_dividend_coverage += 1

            previous_gap = _finite_float(previous["basket_vs_nav_bps"])
            current_gap = _finite_float(current["basket_vs_nav_bps"])
            if (
                previous["status"] == "ok"
                and current["status"] == "ok"
                and previous_gap is not None
                and current_gap is not None
            ):
                history.at[current_index, "gap_change_bps"] = current_gap - previous_gap

            nav_value = _finite_float(current["nav"])
            shares = _finite_float(current["shares_outstanding"])
            denominator = (
                None if nav_value is None or shares is None else nav_value * shares
            )
            if (
                denominator is not None
                and math.isfinite(denominator)
                and denominator > 0
            ):
                if has_dividend_coverage:
                    interval_start = pd.Timestamp(previous["valuation_date"])
                    interval_end = pd.Timestamp(current["valuation_date"])
                    eligible_dividends = tuple(
                        event
                        for event in dividends
                        if interval_start.date() < event.ex_date <= interval_end.date()
                    )
                    valuation_basis_dividends = dividends_on_share_basis(
                        eligible_dividends,
                        all_splits,
                        interval_end.date(),
                    )
                    receivables = ex_dividend_value(
                        fund_holdings,
                        valuation_basis_dividends,
                        interval_start,
                        interval_end,
                    )
                    if ticker_name in receivables.index:
                        history.at[current_index, "ex_dividend_bps"] = (
                            10_000.0 * float(receivables.loc[ticker_name]) / denominator
                        )
                previous_cash = _finite_float(previous["cash_value_usd"])
                current_cash = _finite_float(current["cash_value_usd"])
                if previous_cash is not None and current_cash is not None:
                    history.at[current_index, "cash_change_bps"] = (
                        10_000.0 * (current_cash - previous_cash) / denominator
                    )

            gap_change = _finite_float(history.at[current_index, "gap_change_bps"])
            ex_dividend = _finite_float(history.at[current_index, "ex_dividend_bps"])
            if gap_change is not None and ex_dividend is not None:
                history.at[current_index, "residual_bps"] = gap_change + ex_dividend

    ok_rows = history.loc[history["status"].eq("ok").to_numpy()]
    gap_counts = {
        f"le_{limit}_bps": int(ok_rows["basket_vs_nav_bps"].abs().le(limit).sum())
        for limit in (5, 10, 15, 30)
    }
    ex_dividend_rows = history.loc[
        history["ex_dividend_bps"].ge(2).fillna(False).to_numpy()
    ]
    no_dividend_rows = history.loc[history["ex_dividend_bps"].eq(0).to_numpy()]
    status_counts = {
        str(status): int(count)
        for status, count in history["status"].value_counts().sort_index().items()
    }
    summary: dict[str, object] = {
        "inputs": {
            "nav_vintage": str(nav_vintage),
            "universe_sha256": universe_sha256,
            "snapshots": [
                {
                    "holdings_vintage": str(snapshot.holdings.path),
                    "component_vintage": str(snapshot.components.path),
                }
                for snapshot in snapshots
            ],
        },
        "status_counts": status_counts,
        "ok_gap_abs_counts": gap_counts,
        "intervals_without_dividend_coverage": intervals_without_dividend_coverage,
        "ex_dividend_ge_2_bps": {
            "n": len(ex_dividend_rows),
            "median_residual_bps": _median_or_none(ex_dividend_rows["residual_bps"]),
            "median_abs_residual_bps": _median_or_none(
                ex_dividend_rows["residual_bps"].abs()
            ),
        },
        "ex_dividend_eq_0_bps": {
            "n": len(no_dividend_rows),
            "median_abs_gap_change_bps": _median_or_none(
                no_dividend_rows["gap_change_bps"].abs()
            ),
        },
    }
    return history, summary
