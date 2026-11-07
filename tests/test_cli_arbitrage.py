"""Offline CLI tests for issuer NAV and premium/flow commands."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from etf_monitor import cli


def test_fetch_issuer_nav_requires_authorization(tmp_path: Path) -> None:
    assert cli.main(["fetch-issuer-nav", "--data-root", str(tmp_path)]) == 2
    assert not any(tmp_path.iterdir())


def test_fetch_prices_rejects_unknown_universe(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(
            [
                "fetch-prices",
                "--end-exclusive",
                "2024-07-10",
                "--data-root",
                str(tmp_path),
                "--universe",
                "everything",
            ]
        )


def test_premium_flows_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.issuer_nav.helpers import nav_frame
    from tests.prices.helpers import canonical_frame

    prices = canonical_frame(source="FICTIONAL_synthetic")
    nav = nav_frame([("SPY", "2024-07-02", 100.0, 10)])

    class _Vintage:
        frame = nav

    monkeypatch.setattr(cli, "_load_prices", lambda path: prices)
    monkeypatch.setattr(cli, "load_nav_vintage", lambda path, universe: _Vintage())
    out = tmp_path / "out"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "SPY.json").write_text(
        '{"chart": {"result": [{"meta": {"symbol": "SPY"}}], "error": null}}'
    )

    code = cli.main(
        [
            "premium-flows",
            "--prices",
            "p.parquet",
            "--nav-vintage",
            "v",
            "--price-raw-dir",
            str(raw_dir),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    premium = pd.read_parquet(out / "premium_discount.parquet")
    assert premium["premium_discount_bps"].tolist() == pytest.approx([100.0])
    assert (out / "flow_proxy.parquet").is_file()
    assert (out / "summary.json").is_file()


@pytest.mark.parametrize("command", ["fetch-holdings", "fetch-component-prices"])
def test_gate9_live_commands_refuse_without_authorization(
    tmp_path: Path, command: str
) -> None:
    arguments = [command, "--data-root", str(tmp_path)]
    if command == "fetch-component-prices":
        arguments += [
            "--holdings-vintage",
            str(tmp_path / "missing"),
            "--start",
            "2024-07-01",
            "--end-exclusive",
            "2024-07-06",
        ]
    assert cli.main(arguments) == 2
    assert not any(tmp_path.iterdir())


def test_arbitrage_band_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from tests.prices.helpers import canonical_frame

    prices = canonical_frame(source="FICTIONAL_synthetic")
    monkeypatch.setattr(cli, "_load_prices", lambda path: prices)
    dates = pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-03"])
    flows_dir = tmp_path / "flows"
    flows_dir.mkdir()
    pd.DataFrame(
        {
            "ticker": "SPY",
            "date": dates,
            "premium_discount_bps": [50.0, 50.0, 0.0],
            "nav": 100.0,
            "status": "ok",
        }
    ).to_parquet(flows_dir / "premium_discount.parquet", index=False)
    pd.DataFrame(
        {
            "ticker": "SPY",
            "date": dates,
            "shares_outstanding": pd.array([10, 12, 12], dtype="Int64"),
            "status": "ok",
        }
    ).to_parquet(flows_dir / "flow_proxy.parquet", index=False)
    out = tmp_path / "band"

    code = cli.main(
        [
            "arbitrage-band",
            "--prices",
            "p.parquet",
            "--premium-flows",
            str(flows_dir),
            "--horizon",
            "1",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    band = pd.read_parquet(out / "arbitrage_band.parquet")
    assert band["side"].tolist() == ["premium", "premium", "inside"]
    response = pd.read_parquet(out / "forward_share_response.parquet")
    assert response["forward_rel_change"].iloc[0] == pytest.approx(0.2)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["dislocations"]["SPY"]["persistent"] == 1
