"""Arbitrage universe loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from etf_monitor.universe import (
    UniverseConfigError,
    load_arbitrage_universe,
    load_universe,
)


def test_arbitrage_universe_is_separate_and_fixed() -> None:
    arbitrage = load_arbitrage_universe()
    core = load_universe()

    assert len(arbitrage.etfs) == 12
    assert arbitrage.etfs[0].ticker == "JNK"
    assert not {etf.ticker for etf in arbitrage.etfs} & {
        etf.ticker for etf in core.etfs
    }
    assert arbitrage.sha256 != core.sha256
    assert len(core.etfs) == 24


def test_arbitrage_universe_rejects_crowding_categories(tmp_path: Path) -> None:
    lines = ["universe_version: 1", "etfs:"]
    for position in range(12):
        lines += [
            f"  - ticker: T{chr(65 + position)}",
            "    name: FICTIONAL",
            "    category: Broad equity",
        ]
    path = tmp_path / "arbitrage.yaml"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(UniverseConfigError):
        load_arbitrage_universe(path)
