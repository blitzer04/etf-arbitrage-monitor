"""Validation tests for the owner-approved ETF universe."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
import yaml

from etf_monitor.universe import (
    CATEGORIES,
    UniverseConfigError,
    load_universe,
)

EXPECTED_TICKERS = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "XLK",
    "VGT",
    "SMH",
    "SOXX",
    "IGV",
    "XLF",
    "KRE",
    "XLE",
    "XOP",
    "XLV",
    "XBI",
    "XLY",
    "XLP",
    "XLI",
    "XLB",
    "ARKK",
    "TAN",
    "ICLN",
    "LIT",
)
EXPECTED_CATEGORIES = (
    "Broad market",
    "Broad market",
    "Broad market",
    "Broad market",
    "Broad market",
    "Technology / growth",
    "Technology / growth",
    "Technology / growth",
    "Technology / growth",
    "Technology / growth",
    "Financials",
    "Financials",
    "Energy",
    "Energy",
    "Healthcare",
    "Healthcare",
    "Consumer",
    "Consumer",
    "Industrials / materials",
    "Industrials / materials",
    "Thematic",
    "Thematic",
    "Thematic",
    "Thematic",
)


def _source_data() -> dict[str, Any]:
    """Return a mutable copy of the packaged universe document."""
    resource = files("etf_monitor.config").joinpath("universe.yaml")
    content = yaml.safe_load(resource.read_text(encoding="utf-8"))
    assert isinstance(content, dict)
    return content


def _write_config(path: Path, content: object) -> None:
    """Serialize a test configuration to a temporary YAML file."""
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def test_packaged_universe_has_exact_approved_order_and_categories() -> None:
    universe = load_universe()

    assert tuple(etf.ticker for etf in universe.etfs) == EXPECTED_TICKERS
    assert tuple(etf.category for etf in universe.etfs) == EXPECTED_CATEGORIES
    assert set(EXPECTED_CATEGORIES) == set(CATEGORIES)
    assert len(universe.etfs) == 24


def test_universe_hash_is_stable_and_matches_canonical_json() -> None:
    content = _source_data()
    expected = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert load_universe().sha256 == expected
    assert load_universe().sha256 == load_universe().sha256


@pytest.mark.parametrize("size", [23, 25])
def test_universe_rejects_wrong_entry_count(tmp_path: Path, size: int) -> None:
    content = _source_data()
    entries = content["etfs"]
    assert isinstance(entries, list)
    if size == 23:
        entries.pop()
    else:
        entries.append(dict(entries[-1]))
        entries[-1]["ticker"] = "ABC"
    path = tmp_path / "universe.yaml"
    _write_config(path, content)

    with pytest.raises(UniverseConfigError, match="exactly 24"):
        load_universe(path)


def test_universe_rejects_duplicate_ticker(tmp_path: Path) -> None:
    content = _source_data()
    entries = content["etfs"]
    assert isinstance(entries, list)
    entries[1]["ticker"] = entries[0]["ticker"]
    path = tmp_path / "universe.yaml"
    _write_config(path, content)

    with pytest.raises(UniverseConfigError, match="Duplicate ETF ticker"):
        load_universe(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ticker", "spy", "invalid ticker"),
        ("ticker", "TOOLONG", "invalid ticker"),
        ("category", "Broad Market", "unknown category"),
        ("name", "  ", "non-empty name"),
    ],
)
def test_universe_rejects_invalid_values(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    content = _source_data()
    entries = content["etfs"]
    assert isinstance(entries, list)
    entries[0][field] = value
    path = tmp_path / "universe.yaml"
    _write_config(path, content)

    with pytest.raises(UniverseConfigError, match=message):
        load_universe(path)


def test_universe_rejects_missing_and_extra_keys(tmp_path: Path) -> None:
    content = _source_data()
    content["unexpected"] = True
    path = tmp_path / "universe.yaml"
    _write_config(path, content)
    with pytest.raises(UniverseConfigError, match="only universe_version and etfs"):
        load_universe(path)

    content = _source_data()
    entries = content["etfs"]
    assert isinstance(entries, list)
    del entries[0]["name"]
    _write_config(path, content)
    with pytest.raises(UniverseConfigError, match="ticker, name, and category"):
        load_universe(path)


def test_universe_rejects_duplicate_yaml_mapping_key(tmp_path: Path) -> None:
    resource = files("etf_monitor.config").joinpath("universe.yaml")
    source = resource.read_text(encoding="utf-8")
    duplicate = source.replace(
        "universe_version: 1", "universe_version: 1\nuniverse_version: 1", 1
    )
    path = tmp_path / "universe.yaml"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(UniverseConfigError, match="duplicate key"):
        load_universe(path)
