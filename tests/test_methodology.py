"""Validation tests for methodology constants and approval pins."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
import yaml

from etf_monitor.methodology import MethodologyConfigError, load_methodology

APPROVED_CONSTANTS = {
    "methodology_version": "1.0.0",
    "calendar": "XNYS",
    "exchange_calendars_version": "4.13.2",
    "price_history_start": "2018-01-01",
    "momentum_horizon_sessions": 252,
    "volatility_return_count": 21,
    "volatility_ddof": 1,
    "annualization_sessions": 252,
    "reference_window_sessions": 756,
    "minimum_reference_observations": 252,
    "percentile_method": "midrank",
    "tie_rule": "exact_float_equality",
}


def _source_data() -> dict[str, Any]:
    """Return a mutable copy of the packaged methodology document."""
    resource = files("etf_monitor.config").joinpath("methodology.yaml")
    content = yaml.safe_load(resource.read_text(encoding="utf-8"))
    assert isinstance(content, dict)
    return content


def _write_config(path: Path, content: object) -> None:
    """Serialize a test configuration to a temporary YAML file."""
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def test_methodology_constants_are_approved_and_hash_is_stable() -> None:
    methodology = load_methodology()
    methodology.assert_approved_constants()

    assert methodology.methodology_version == APPROVED_CONSTANTS["methodology_version"]
    assert methodology.calendar == APPROVED_CONSTANTS["calendar"]
    assert methodology.exchange_calendars_version == "4.13.2"
    assert methodology.price_history_start.isoformat() == "2018-01-01"
    assert methodology.momentum_horizon_sessions == 252
    assert methodology.volatility_return_count == 21
    assert methodology.volatility_ddof == 1
    assert methodology.annualization_sessions == 252
    assert methodology.reference_window_sessions == 756
    assert methodology.minimum_reference_observations == 252
    assert methodology.percentile_method == "midrank"
    assert methodology.tie_rule == "exact_float_equality"
    assert methodology.sha256 == load_methodology().sha256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("momentum_horizon_sessions", 0, "positive integer"),
        ("volatility_return_count", True, "positive integer"),
        ("price_history_start", "2018-1-1", "YYYY-MM-DD"),
        ("calendar", "", "non-empty string"),
        ("reference_window_sessions", 755, "owner-approved"),
    ],
)
def test_methodology_rejects_invalid_or_unapproved_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    content = _source_data()
    content[field] = value
    path = tmp_path / "methodology.yaml"
    _write_config(path, content)

    with pytest.raises(MethodologyConfigError, match=message):
        load_methodology(path)


def test_methodology_rejects_missing_extra_and_duplicate_yaml_keys(
    tmp_path: Path,
) -> None:
    content = _source_data()
    del content["calendar"]
    path = tmp_path / "methodology.yaml"
    _write_config(path, content)
    with pytest.raises(MethodologyConfigError, match="missing or unexpected"):
        load_methodology(path)

    resource = files("etf_monitor.config").joinpath("methodology.yaml")
    source = resource.read_text(encoding="utf-8")
    duplicate = source.replace("calendar: XNYS", "calendar: XNYS\ncalendar: XNYS", 1)
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(MethodologyConfigError, match="duplicate key"):
        load_methodology(path)
