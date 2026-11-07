"""Offline tests for SSGA holdings parsing, vintages, components and baskets."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_monitor.arbitrage.basket import (
    basket_fair_value,
    classify_lines,
    required_symbols,
)
from etf_monitor.arbitrage.splits import SplitEvent
from etf_monitor.holdings.acquire import HoldingsBatchResult, fetch_holdings_batch
from etf_monitor.holdings.component_prices import (
    fetch_component_prices,
    load_component_vintage,
    write_component_vintage,
    yahoo_symbol,
)
from etf_monitor.holdings.store import load_holdings_vintage, write_holdings_vintage
from etf_monitor.holdings.vintage import VintageError
from etf_monitor.issuer_nav.sources import NavSource
from etf_monitor.providers.ssga_holdings import (
    FIXED_INCOME_HEADER,
    HoldingsPayloadError,
    parse_ssga_holdings,
)
from etf_monitor.providers.transport import ProviderTransportError
from tests.fixtures.ssga_holdings import (
    EQUITY_LINES,
    build_fixed_income_workbook,
    build_holdings_workbook,
)
from tests.fixtures.yahoo_chart import build_yahoo_chart_payload
from tests.issuer_nav.helpers import MappingNavTransport, nav_frame, small_universe

RETRIEVED = datetime(2024, 7, 4, 1, 0, tzinfo=UTC)
SOURCES = {
    "SPY": NavSource("SPY", "ssga_navhist_xlsx", None),
    "BIL": NavSource("BIL", "ssga_navhist_xlsx", None),
    "QQQ": NavSource("QQQ", None, "FICTIONAL non-SSGA reason"),
}


def test_equity_layout_keeps_lines_in_order_with_dashes_missing() -> None:
    parsed = parse_ssga_holdings(build_holdings_workbook(), "SPY")
    assert parsed.as_of == date(2024, 7, 3)
    assert parsed.layout == "equity"
    assert [line.security_ticker for line in parsed.holdings] == [
        "AAA",
        "BBB.B",
        "IXXZ6",
        None,
        None,
    ]
    assert parsed.holdings[4].quantity == 1_000.0
    assert parsed.holdings[2].sedol is None


def test_fixed_income_layout_parses_maturity_and_market_value() -> None:
    parsed = parse_ssga_holdings(build_fixed_income_workbook(), "BIL")
    assert parsed.layout == "fixed_income"
    assert parsed.holdings[0].maturity == date(2030, 1, 15)
    assert parsed.holdings[1].maturity is None
    assert parsed.holdings[2].market_value == -5.0


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"workbook_ticker": "XYZ"}, "identity_mismatch"),
        ({"as_of": "03-Jul-2024"}, "as_of"),
        ({"as_of": "As of 31-Jun-2024"}, "as_of"),
        ({"blank_fourth_row": False}, "workbook_shape"),
        ({"header": ("Name", "Ticker")}, "header_mismatch"),
        ({"extra_rows_after_footer": (EQUITY_LINES[0],)}, "malformed_row"),
        (
            {"lines": (("X", "AAA", "1", "-", "heavy", "-", 1.0, "USD"),)},
            "invalid_number",
        ),
        ({"lines": (("X", "AAA", "-", "-", 1.0, "-", 1.0, "USD"),)}, "invalid_text"),
        (
            {"lines": (("X", "AAA", "1", "-", 1.0, "-", 1.0, "usd"),)},
            "invalid_currency",
        ),
        ({"lines": ()}, "empty"),
    ],
)
def test_layout_violations_are_rejected(kwargs: dict[str, object], reason: str) -> None:
    with pytest.raises(HoldingsPayloadError) as error:
        parse_ssga_holdings(build_holdings_workbook(**kwargs), "SPY")  # type: ignore[arg-type]
    assert error.value.reason_code == reason


def test_bad_maturity_is_rejected() -> None:
    lines = (("B", "US1", "-", 1.0, 1.0, 1.0, 1.0, "USD", "2030-01-15"),)
    raw = build_holdings_workbook("BIL", lines, header=FIXED_INCOME_HEADER)
    with pytest.raises(HoldingsPayloadError, match="MM/DD/YYYY"):
        parse_ssga_holdings(raw, "BIL")


def test_not_xlsx_is_rejected() -> None:
    with pytest.raises(HoldingsPayloadError) as error:
        parse_ssga_holdings(b"<html>", "SPY")
    assert error.value.reason_code == "not_xlsx"


def _batch(responses: dict[str, bytes | Exception]) -> HoldingsBatchResult:
    return fetch_holdings_batch(
        small_universe("SPY", "BIL", "QQQ"),
        MappingNavTransport(responses),
        lambda: RETRIEVED,
        SOURCES,
    )


def test_batch_isolates_failures_and_skips_non_ssga_funds() -> None:
    batch = _batch(
        {"SPY": build_holdings_workbook(), "BIL": ProviderTransportError("x")}
    )
    assert [status.status for status in batch.statuses] == [
        "success",
        "failed",
        "unavailable",
    ]
    assert set(batch.frame["ticker"]) == {"SPY"}
    assert batch.frame["position"].tolist() == [1, 2, 3, 4, 5]


def test_holdings_dated_after_retrieval_fail() -> None:
    batch = _batch(
        {"SPY": build_holdings_workbook(as_of="As of 05-Jul-2024"), "BIL": b"x"}
    )
    assert batch.statuses[0].reason_code == "holdings_contract"
    assert "SPY" in batch.raw_payloads


def test_vintage_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    universe = small_universe("SPY", "BIL", "QQQ")
    batch = _batch(
        {"SPY": build_holdings_workbook(), "BIL": build_fixed_income_workbook()}
    )
    path = write_holdings_vintage(batch, tmp_path, universe)
    loaded = load_holdings_vintage(path, universe)
    pd.testing.assert_frame_equal(loaded.frame, batch.frame)
    with pytest.raises(VintageError, match="already exists"):
        write_holdings_vintage(batch, tmp_path, universe)
    with pytest.raises(VintageError, match="different universe"):
        load_holdings_vintage(path, small_universe("SPY", "BIL"))
    raw = path / "raw/SPY.xlsx"
    raw.chmod(0o644)
    raw.write_bytes(raw.read_bytes() + b"\0")
    with pytest.raises(VintageError, match="Digest mismatch"):
        load_holdings_vintage(path, universe)


def test_yahoo_symbol_mapping() -> None:
    assert yahoo_symbol("BRK.B") == "BRK-B"
    assert yahoo_symbol("AAPL") == "AAPL"
    assert yahoo_symbol("2602335D") is None
    assert yahoo_symbol(None) is None


def _holdings() -> pd.DataFrame:
    batch = _batch(
        {"SPY": build_holdings_workbook(), "BIL": build_fixed_income_workbook()}
    )
    return batch.frame


def test_line_classes_and_required_symbols() -> None:
    spy = _holdings().query("ticker == 'SPY'")
    assert classify_lines(spy).tolist() == [
        "priced",
        "priced",
        "futures",
        "unpriced",
        "cash",
    ]
    assert required_symbols(_holdings()) == ("AAA", "BBB-B")


def _prices(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.array([r[0] for r in rows], dtype="string"),
            "date": pd.to_datetime([r[1] for r in rows]).astype("datetime64[ns]"),
            "close": np.array([r[2] for r in rows], dtype=np.float64),
        }
    )


def test_basket_value_both_alignments_with_split_and_missing_price() -> None:
    nav = nav_frame(
        [("SPY", "2024-07-03", 4.0, 2000), ("BIL", "2024-07-03", 0.5, 2000)]
    )
    prices = _prices([("AAA", "2024-07-03", 50.0), ("BBB-B", "2024-07-03", 10.0)])
    split = (SplitEvent("BBB-B", date(2024, 7, 5), 2, 1),)
    result = basket_fair_value(_holdings(), prices, nav, split)
    spy = result.query("ticker == 'SPY' and alignment == 'same_session'").iloc[0]
    # 100*50 + 50*(10*2) + 1000 cash = 7000; per share 3.5 vs NAV 4.0.
    assert spy["basket_value_usd"] == pytest.approx(7_000.0)
    assert spy["basket_vs_nav_bps"] == pytest.approx(-1_250.0)
    assert spy["unpriced_weight_pct"] == pytest.approx(0.5)
    assert spy["implied_shares_outstanding"] == pytest.approx(1_750.0)
    assert spy["futures_lines"] == 1
    later = result.query("ticker == 'SPY' and alignment == 'next_session'").iloc[0]
    assert later["valuation_date"] == pd.Timestamp("2024-07-05")
    assert later["status"] == "missing_component_price"
    assert pd.isna(later["basket_value_usd"])
    bil = result.query("ticker == 'BIL' and alignment == 'same_session'").iloc[0]
    assert bil["valuation_method"] == "issuer_market_value"
    assert bil["basket_value_usd"] == pytest.approx(1_000.0)
    assert bil["basket_vs_nav_bps"] == pytest.approx(0.0)
    assert bil["weight_consistency_max_abs_pct"] == pytest.approx(0.0)
    assert result.query("ticker == 'BIL' and alignment == 'next_session'")[
        "status"
    ].tolist() == ["missing_nav"]


def test_non_usd_equity_fund_is_not_valued() -> None:
    lines = (("FICTIONAL EU", "EEE", "1", "-", 99.0, "-", 5.0, "EUR"),)
    holdings = fetch_holdings_batch(
        small_universe("SPY"),
        MappingNavTransport({"SPY": build_holdings_workbook(lines=lines)}),
        lambda: RETRIEVED,
        {"SPY": SOURCES["SPY"]},
    ).frame
    result = basket_fair_value(holdings, _prices([]), nav_frame([]))
    assert set(result["status"]) == {"unsupported_non_usd_securities"}
    assert required_symbols(holdings) == ()


class _ChartTransport:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses

    def fetch(self, symbol: str, period1: int, period2: int) -> bytes:
        if symbol not in self.responses:
            raise ProviderTransportError(f"FICTIONAL failure for {symbol}")
        return self.responses[symbol]


def test_component_vintage_round_trip(tmp_path: Path) -> None:
    payload = json.dumps(build_yahoo_chart_payload("AAA", ("2024-07-03",))).encode()
    batch = fetch_component_prices(
        ["AAA", "BBB-B", "AAA"],
        date(2024, 7, 1),
        date(2024, 7, 6),
        _ChartTransport({"AAA": payload}),
        lambda: RETRIEVED,
    )
    assert [s.status for s in batch.statuses] == ["success", "failed"]
    path = write_component_vintage(batch, tmp_path)
    loaded = load_component_vintage(path)
    pd.testing.assert_frame_equal(loaded.frame, batch.frame)
    assert loaded.failed_symbols == ("BBB-B",)
    manifest = json.loads((path / "manifest.json").read_text())
    assert set(manifest["files"]) == {"raw/AAA.json"}
