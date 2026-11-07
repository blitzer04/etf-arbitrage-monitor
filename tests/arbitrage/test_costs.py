from __future__ import annotations

from pathlib import Path

import pytest

from etf_monitor.arbitrage.costs import CostConfigError, load_fund_costs

PACKAGED = Path("src/etf_monitor/config")


def test_packaged_costs_cover_every_ssga_fund() -> None:
    costs, classes = load_fund_costs()
    assert len(costs) == 25
    assert costs["JNK"].creation_unit_shares == 200_000
    assert costs["SPDW"].stale_nav and not costs["SPY"].stale_nav
    assert classes["EBND"] == "em_local_bond"


def test_unknown_source_is_rejected(tmp_path: Path) -> None:
    text = (PACKAGED / "arbitrage_costs.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "costs.yaml"
    bad.write_text(
        text.replace("fee_source: ncen_dia", "fee_source: nowhere"), encoding="utf-8"
    )
    with pytest.raises(CostConfigError, match="nowhere"):
        load_fund_costs(bad)


def test_negative_basket_cost_is_rejected(tmp_path: Path) -> None:
    text = (PACKAGED / "arbitrage_costs.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "costs.yaml"
    bad.write_text(
        text.replace("basket_cost_bps: 2.0", "basket_cost_bps: -2.0"), encoding="utf-8"
    )
    with pytest.raises(CostConfigError, match="basket_cost_bps"):
        load_fund_costs(bad)
