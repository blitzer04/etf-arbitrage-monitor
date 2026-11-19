"""Fictional basket-history calculations from hand-built issuer vintages."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from etf_monitor import cli
from etf_monitor.arbitrage.basket_history import (
    BasketHistorySnapshot,
    basket_history,
)
from etf_monitor.arbitrage.dividends import DividendEvent
from etf_monitor.arbitrage.splits import SplitEvent
from etf_monitor.holdings.acquire import fetch_holdings_batch
from etf_monitor.holdings.component_prices import (
    fetch_component_prices,
    load_component_vintage,
    write_component_vintage,
)
from etf_monitor.holdings.store import load_holdings_vintage, write_holdings_vintage
from etf_monitor.issuer_nav.acquire import fetch_nav_batch
from etf_monitor.issuer_nav.sources import NavSource
from etf_monitor.issuer_nav.store import load_nav_vintage, write_nav_vintage
from etf_monitor.providers.ssga_nav import SSGA_NAV_SOURCE
from etf_monitor.universe import Universe
from tests.fixtures.ssga_holdings import build_holdings_workbook
from tests.fixtures.ssga_nav import build_ssga_workbook
from tests.fixtures.yahoo_chart import build_yahoo_chart_payload
from tests.issuer_nav.helpers import MappingNavTransport, small_universe


class _ChartTransport:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def fetch(self, symbol: str, period1: int, period2: int) -> bytes:
        assert symbol == "AAA"
        assert period1 < period2
        return self.payload


def _clock(value: datetime) -> Callable[[], datetime]:
    return lambda: value


def _holdings_vintage(
    root: Path,
    universe: Universe,
    as_of: str,
    cash: float,
    retrieved_at: datetime,
) -> Path:
    lines = (
        (
            "FICTIONAL ALPHA CORP",
            "AAA",
            "000000001",
            "0000001",
            50.0,
            "-",
            100.0,
            "USD",
        ),
        ("US DOLLAR", "-", "999USDZ92", "-", 50.0, "-", cash, "USD"),
    )
    payload = build_holdings_workbook("SPY", lines, as_of=f"As of {as_of}")
    sources = {"SPY": NavSource("SPY", SSGA_NAV_SOURCE, None)}
    batch = fetch_holdings_batch(
        universe,
        MappingNavTransport({"SPY": payload}),
        _clock(retrieved_at),
        sources,
    )
    return write_holdings_vintage(batch, root, universe)


def _component_vintage(
    root: Path,
    retrieved_at: datetime,
    dividend_amount: float | None,
) -> Path:
    chart = build_yahoo_chart_payload(
        "AAA",
        ("2024-07-03", "2024-07-05", "2024-07-08"),
        quote_values={"close": (9.0, 8.5, 8.5)},
    )
    if dividend_amount is not None:
        ex_date_open = datetime(2024, 7, 5, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        chart["chart"]["result"][0]["events"] = {
            "dividends": {
                "fictional": {
                    "date": int(ex_date_open.timestamp()),
                    "amount": dividend_amount,
                }
            }
        }
    payload = json.dumps(chart).encode()
    batch = fetch_component_prices(
        ["AAA"],
        date(2024, 7, 1),
        date(2024, 7, 9),
        _ChartTransport(payload),
        _clock(retrieved_at),
    )
    return write_component_vintage(batch, root)


def _history_inputs(
    root: Path,
) -> tuple[Universe, list[tuple[Path, Path]], Path]:
    universe = small_universe("SPY")
    first_holdings = _holdings_vintage(
        root / "holdings",
        universe,
        "02-Jul-2024",
        100.0,
        datetime(2024, 7, 3, 1, tzinfo=UTC),
    )
    second_holdings = _holdings_vintage(
        root / "holdings",
        universe,
        "05-Jul-2024",
        110.0,
        datetime(2024, 7, 8, 1, tzinfo=UTC),
    )
    first_components = _component_vintage(
        root / "components",
        datetime(2024, 7, 10, 1, tzinfo=UTC),
        None,
    )
    second_components = _component_vintage(
        root / "components",
        datetime(2024, 7, 11, 1, tzinfo=UTC),
        0.5,
    )
    nav_payload = build_ssga_workbook(
        "SPY",
        (
            ("08-Jul-2024", 10.1, 100.0, 1010.0),
            ("03-Jul-2024", 10.0, 100.0, 1000.0),
        ),
    )
    nav_batch = fetch_nav_batch(
        universe,
        date(2024, 7, 1),
        MappingNavTransport({"SPY": nav_payload}),
        _clock(datetime(2024, 7, 10, 1, tzinfo=UTC)),
        {"SPY": NavSource("SPY", SSGA_NAV_SOURCE, None)},
    )
    nav_path = write_nav_vintage(nav_batch, root / "nav", universe)
    return (
        universe,
        [
            (first_holdings, first_components),
            (second_holdings, second_components),
        ],
        nav_path,
    )


def _loaded_snapshots(
    universe: Universe, snapshot_paths: list[tuple[Path, Path]]
) -> list[BasketHistorySnapshot]:
    return [
        BasketHistorySnapshot(
            load_holdings_vintage(holdings_path, universe),
            load_component_vintage(component_path),
        )
        for holdings_path, component_path in snapshot_paths
    ]


def test_basket_history_cli_writes_hand_calculated_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe, snapshot_paths, nav_path = _history_inputs(tmp_path)
    monkeypatch.setattr(cli, "_universe", lambda _name: universe)
    out = tmp_path / "history"
    arguments = ["basket-history", "--nav-vintage", str(nav_path), "--out", str(out)]
    for holdings_path, component_path in snapshot_paths:
        arguments.extend(["--snapshot", str(holdings_path), str(component_path)])

    assert cli.main(arguments) == 0
    history = pd.read_csv(out / "basket_history.csv")
    assert history["valuation_date"].tolist() == ["2024-07-03", "2024-07-08"]
    assert history["status"].tolist() == ["ok", "ok"]
    assert history["dividend_coverage"].tolist() == [False, True]
    assert history.loc[0, "basket_vs_nav_bps"] == pytest.approx(0.0)

    row = history.iloc[1]
    expected_gap_change = 10_000.0 * (9.6 / 10.1 - 1.0)
    expected_ex_dividend = 10_000.0 * 50.0 / (10.1 * 100.0)
    expected_cash_change = 10_000.0 * 10.0 / (10.1 * 100.0)
    assert row["gap_change_bps"] == pytest.approx(expected_gap_change)
    assert row["ex_dividend_bps"] == pytest.approx(expected_ex_dividend)
    assert row["cash_change_bps"] == pytest.approx(expected_cash_change)
    assert row["residual_bps"] == pytest.approx(
        expected_gap_change + expected_ex_dividend
    )
    assert row["holdings_vintage"] == str(snapshot_paths[1][0])

    summary = json.loads((out / "summary.json").read_text())
    assert summary["status_counts"] == {"ok": 2}
    assert summary["ok_gap_abs_counts"] == {
        "le_5_bps": 1,
        "le_10_bps": 1,
        "le_15_bps": 1,
        "le_30_bps": 1,
    }
    assert summary["ex_dividend_ge_2_bps"]["n"] == 1
    assert summary["ex_dividend_ge_2_bps"]["median_residual_bps"] == pytest.approx(0.0)
    assert summary["ex_dividend_ge_2_bps"]["median_abs_residual_bps"] == pytest.approx(
        0.0
    )
    assert summary["ex_dividend_eq_0_bps"]["n"] == 0
    assert summary["ex_dividend_eq_0_bps"]["median_abs_gap_change_bps"] is None
    assert summary["intervals_without_dividend_coverage"] == 0
    assert summary["inputs"]["snapshots"][0]["holdings_vintage"] == str(
        snapshot_paths[0][0]
    )

    assert cli.main(arguments) == 2


@pytest.mark.parametrize(
    ("start", "end_exclusive"),
    [
        (date(2024, 7, 6), date(2024, 7, 9)),
        (date(2024, 7, 1), date(2024, 7, 8)),
    ],
)
def test_basket_history_marks_uncovered_component_window(
    tmp_path: Path,
    start: date,
    end_exclusive: date,
) -> None:
    universe, snapshot_paths, nav_path = _history_inputs(tmp_path)
    snapshots = _loaded_snapshots(universe, snapshot_paths)
    short_window = replace(
        snapshots[1].components,
        start=start,
        end_exclusive=end_exclusive,
    )
    history, summary = basket_history(
        [
            snapshots[0],
            BasketHistorySnapshot(snapshots[1].holdings, short_window),
        ],
        load_nav_vintage(nav_path, universe).frame,
        nav_vintage=nav_path,
        universe_sha256=universe.sha256,
    )

    row = history.iloc[1]
    assert not row["dividend_coverage"]
    assert pd.isna(row["ex_dividend_bps"])
    assert pd.isna(row["residual_bps"])
    assert summary["intervals_without_dividend_coverage"] == 1


def test_basket_history_marks_failed_priced_symbol_uncovered(
    tmp_path: Path,
) -> None:
    universe, snapshot_paths, nav_path = _history_inputs(tmp_path)
    snapshots = _loaded_snapshots(universe, snapshot_paths)
    failed_components = replace(
        snapshots[1].components,
        failed_symbols=("AAA",),
    )
    history, summary = basket_history(
        [
            snapshots[0],
            BasketHistorySnapshot(snapshots[1].holdings, failed_components),
        ],
        load_nav_vintage(nav_path, universe).frame,
        nav_vintage=nav_path,
        universe_sha256=universe.sha256,
    )

    row = history.iloc[1]
    assert not row["dividend_coverage"]
    assert pd.notna(row["gap_change_bps"])
    assert pd.isna(row["ex_dividend_bps"])
    assert pd.isna(row["residual_bps"])
    assert summary["intervals_without_dividend_coverage"] == 1


def test_basket_history_rejects_duplicate_as_of_with_both_paths(
    tmp_path: Path,
) -> None:
    universe, snapshot_paths, nav_path = _history_inputs(tmp_path)
    snapshots = _loaded_snapshots(universe, snapshot_paths)
    duplicate_path = tmp_path / "duplicate-holdings"
    duplicate_holdings = replace(
        snapshots[1].holdings,
        path=duplicate_path,
        frame=snapshots[0].holdings.frame,
    )
    duplicate = BasketHistorySnapshot(duplicate_holdings, snapshots[1].components)

    with pytest.raises(ValueError, match="same as_of label") as error:
        basket_history(
            [snapshots[0], duplicate],
            load_nav_vintage(nav_path, universe).frame,
            nav_vintage=nav_path,
            universe_sha256=universe.sha256,
        )
    assert str(snapshots[0].holdings.path) in str(error.value)
    assert str(duplicate_path) in str(error.value)


def test_basket_history_rejects_conflicting_dividends_across_vintages(
    tmp_path: Path,
) -> None:
    universe, snapshot_paths, nav_path = _history_inputs(tmp_path)
    snapshots = _loaded_snapshots(universe, snapshot_paths)
    conflicting_components = replace(
        snapshots[0].components,
        dividends=(DividendEvent("AAA", date(2024, 7, 5), 0.25),),
    )
    conflicting = BasketHistorySnapshot(snapshots[0].holdings, conflicting_components)

    with pytest.raises(ValueError, match="Conflicting dividends") as error:
        basket_history(
            [conflicting, snapshots[1]],
            load_nav_vintage(nav_path, universe).frame,
            nav_vintage=nav_path,
            universe_sha256=universe.sha256,
        )
    assert str(conflicting_components.path) in str(error.value)
    assert str(snapshots[1].components.path) in str(error.value)


def test_basket_history_unions_pre_and_post_split_amounts(tmp_path: Path) -> None:
    universe, snapshot_paths, nav_path = _history_inputs(tmp_path)
    snapshots = _loaded_snapshots(universe, snapshot_paths)
    ex_date = date(2024, 7, 5)
    first_components = replace(
        snapshots[0].components,
        dividends=(DividendEvent("AAA", ex_date, 1.0),),
        splits=(),
    )
    second_components = replace(
        snapshots[1].components,
        dividends=(DividendEvent("AAA", ex_date, 0.5),),
        splits=(SplitEvent("AAA", date(2024, 7, 8), 2, 1),),
    )
    history, _ = basket_history(
        [
            BasketHistorySnapshot(snapshots[0].holdings, first_components),
            BasketHistorySnapshot(snapshots[1].holdings, second_components),
        ],
        load_nav_vintage(nav_path, universe).frame,
        nav_vintage=nav_path,
        universe_sha256=universe.sha256,
    )

    assert history.iloc[1]["ex_dividend_bps"] == pytest.approx(
        10_000.0 * 100.0 / (10.1 * 100.0)
    )


def test_later_vintage_split_does_not_adjust_earlier_only_dividend(
    tmp_path: Path,
) -> None:
    universe, snapshot_paths, nav_path = _history_inputs(tmp_path)
    snapshots = _loaded_snapshots(universe, snapshot_paths)
    first_components = replace(
        snapshots[0].components,
        dividends=(DividendEvent("AAA", date(2024, 7, 5), 1.0),),
        splits=(),
    )
    second_components = replace(
        snapshots[1].components,
        dividends=(),
        splits=(SplitEvent("AAA", date(2024, 7, 8), 2, 1),),
    )
    history, _ = basket_history(
        [
            BasketHistorySnapshot(snapshots[0].holdings, first_components),
            BasketHistorySnapshot(snapshots[1].holdings, second_components),
        ],
        load_nav_vintage(nav_path, universe).frame,
        nav_vintage=nav_path,
        universe_sha256=universe.sha256,
    )

    assert history.iloc[1]["ex_dividend_bps"] == pytest.approx(
        10_000.0 * 100.0 / (10.1 * 100.0)
    )


def test_identical_dividends_across_vintages_are_counted_once(tmp_path: Path) -> None:
    universe, snapshot_paths, nav_path = _history_inputs(tmp_path)
    snapshots = _loaded_snapshots(universe, snapshot_paths)
    duplicate_event = replace(
        snapshots[0].components,
        dividends=snapshots[1].components.dividends,
    )
    repeated = BasketHistorySnapshot(snapshots[0].holdings, duplicate_event)
    history, _ = basket_history(
        [repeated, snapshots[1]],
        load_nav_vintage(nav_path, universe).frame,
        nav_vintage=nav_path,
        universe_sha256=universe.sha256,
    )

    assert history.iloc[1]["ex_dividend_bps"] == pytest.approx(
        10_000.0 * 50.0 / (10.1 * 100.0)
    )
