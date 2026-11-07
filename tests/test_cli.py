"""CLI authorization boundary tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import etf_monitor.cli as cli
from etf_monitor.cli import main
from etf_monitor.prices.store import StorePaths


def test_cli_refuses_live_request_without_authorization() -> None:
    result = main(
        [
            "fetch-prices",
            "--start",
            "2024-07-01",
            "--end-exclusive",
            "2024-07-10",
            "--data-root",
            "unused",
        ]
    )

    assert result == 2


def test_cli_requires_data_root() -> None:
    try:
        main(["fetch-prices", "--end-exclusive", "2024-07-10"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("CLI accepted a missing --data-root.")


def test_cli_roots_raw_and_processed_files_at_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class FakeTransport:
        pass

    def fake_fetch(**kwargs: object) -> SimpleNamespace:
        captured["raw_dir"] = kwargs["raw_dir"]
        return SimpleNamespace(statuses=[SimpleNamespace(status="success")])

    def fake_persist(
        batch: object, paths: StorePaths, universe: object = None
    ) -> SimpleNamespace:
        captured["paths"] = paths
        captured["universe"] = universe
        return SimpleNamespace(persisted=True)

    monkeypatch.setattr(cli, "UrllibChartTransport", FakeTransport)
    monkeypatch.setattr(cli, "fetch_price_batch", fake_fetch)
    monkeypatch.setattr(cli, "_log_batch", lambda batch: None)
    monkeypatch.setattr(cli, "persist_batch", fake_persist)

    result = main(
        [
            "fetch-prices",
            "--start",
            "2024-07-01",
            "--end-exclusive",
            "2024-07-10",
            "--data-root",
            str(tmp_path),
            "--authorize-live-request",
        ]
    )

    assert result == 0
    assert captured["raw_dir"] == tmp_path / "raw/prices"
    assert captured["paths"] == StorePaths.under(tmp_path)
