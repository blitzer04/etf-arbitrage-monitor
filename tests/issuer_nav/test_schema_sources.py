"""Issuer NAV schema, conversion, and source-configuration tests."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from etf_monitor.issuer_nav.schema import (
    NavValidationError,
    nav_frame_from_history,
    validate_nav_frame,
)
from etf_monitor.issuer_nav.sources import (
    NavSourceConfigError,
    load_nav_sources,
    nav_sources_for,
)
from etf_monitor.providers.ssga_nav import (
    SSGA_NAV_SOURCE,
    NavObservation,
    ParsedNavHistory,
)
from etf_monitor.universe import load_arbitrage_universe, load_universe
from tests.issuer_nav.helpers import RETRIEVED_AT, nav_frame


def _history(shares: int | None) -> ParsedNavHistory:
    return ParsedNavHistory(
        ticker="SPY",
        fund_name="FICTIONAL",
        observations=(NavObservation(date(2024, 7, 2), 100.25, shares, None),),
        rows_before_history_start=0,
    )


def test_conversion_is_lossless_above_two_pow_53() -> None:
    frame = nav_frame_from_history(_history(2**53 + 1), RETRIEVED_AT, SSGA_NAV_SOURCE)

    assert frame["shares_outstanding"].dtype == pd.Int64Dtype()
    assert int(frame["shares_outstanding"].iloc[0]) == 2**53 + 1
    assert pd.isna(frame["total_net_assets"].iloc[0])
    validate_nav_frame(frame, {"SPY"})


def test_missing_shares_stay_missing() -> None:
    frame = nav_frame_from_history(_history(None), RETRIEVED_AT, SSGA_NAV_SOURCE)
    assert frame["shares_outstanding"].isna().all()


def test_naive_retrieval_instant_is_rejected() -> None:
    with pytest.raises(NavValidationError):
        nav_frame_from_history(_history(1), datetime(2024, 7, 9), SSGA_NAV_SOURCE)


@pytest.mark.parametrize(
    ("rows", "allowed"),
    [
        ([("SPY", "2024-07-06", 1.0, 1)], {"SPY"}),  # Saturday
        ([("SPY", "2024-07-04", 1.0, 1)], {"SPY"}),  # Independence Day
        ([("SPY", "2024-07-02", 1.0, 1)], {"DIA"}),
        ([("SPY", "2024-07-02", 0.0, 1)], {"SPY"}),
        ([("SPY", "2024-07-02", 1.0, -1)], {"SPY"}),
        ([("SPY", "2024-07-03", 1.0, 1), ("SPY", "2024-07-02", 1.0, 1)], {"SPY"}),
        ([("SPY", "2024-07-02", 1.0, 1), ("SPY", "2024-07-02", 1.0, 1)], {"SPY"}),
    ],
)
def test_contract_violations_are_rejected(
    rows: list[tuple[str, str, float, int | None]], allowed: set[str]
) -> None:
    with pytest.raises(NavValidationError):
        validate_nav_frame(nav_frame(rows), allowed)


def test_mixed_sources_are_rejected() -> None:
    frame = pd.concat(
        [
            nav_frame([("SPY", "2024-07-02", 1.0, 1)]),
            nav_frame([("SPY", "2024-07-03", 1.0, 1)], source=SSGA_NAV_SOURCE),
        ],
        ignore_index=True,
    )
    with pytest.raises(NavValidationError):
        validate_nav_frame(frame, {"SPY"})


def test_every_universe_etf_has_a_configured_source() -> None:
    sources = load_nav_sources()
    core = nav_sources_for(load_universe(), sources)
    arbitrage = nav_sources_for(load_arbitrage_universe(), sources)

    assert sum(source.source == SSGA_NAV_SOURCE for source in core) == 13
    assert all(source.source == SSGA_NAV_SOURCE for source in arbitrage)
    assert all(source.reason for source in core if source.source is None)


def test_source_without_reason_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text("sources_version: 1\nfunds:\n  QQQ: {source: null, reason: null}\n")
    with pytest.raises(NavSourceConfigError):
        load_nav_sources(path)


def test_unknown_source_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        "sources_version: 1\nfunds:\n  QQQ: {source: invesco_api, reason: null}\n"
    )
    with pytest.raises(NavSourceConfigError):
        load_nav_sources(path)
