"""Load and validate the owner-approved methodology constants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from etf_monitor.configuration import load_yaml_bytes, read_config_bytes

_APPROVED_CONSTANTS: dict[str, object] = {
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
_INTEGER_FIELDS = (
    "momentum_horizon_sessions",
    "volatility_return_count",
    "volatility_ddof",
    "annualization_sessions",
    "reference_window_sessions",
    "minimum_reference_observations",
)


class MethodologyConfigError(ValueError):
    """Indicate that methodology configuration is invalid or unapproved."""


@dataclass(frozen=True, slots=True)
class Methodology:
    """Hold validated calculation constants and their configuration hash."""

    methodology_version: str
    calendar: str
    exchange_calendars_version: str
    price_history_start: date
    momentum_horizon_sessions: int
    volatility_return_count: int
    volatility_ddof: int
    annualization_sessions: int
    reference_window_sessions: int
    minimum_reference_observations: int
    percentile_method: str
    tie_rule: str
    sha256: str

    def assert_approved_constants(self) -> None:
        """Reject changes to owner-approved methodology defaults."""
        current = {
            "methodology_version": self.methodology_version,
            "calendar": self.calendar,
            "exchange_calendars_version": self.exchange_calendars_version,
            "price_history_start": self.price_history_start.isoformat(),
            "momentum_horizon_sessions": self.momentum_horizon_sessions,
            "volatility_return_count": self.volatility_return_count,
            "volatility_ddof": self.volatility_ddof,
            "annualization_sessions": self.annualization_sessions,
            "reference_window_sessions": self.reference_window_sessions,
            "minimum_reference_observations": self.minimum_reference_observations,
            "percentile_method": self.percentile_method,
            "tie_rule": self.tie_rule,
        }
        if current != _APPROVED_CONSTANTS:
            raise MethodologyConfigError(
                "Methodology configuration differs from the owner-approved constants."
            )


def _canonical_sha256(content: object) -> str:
    """Hash parsed YAML content using canonical JSON serialization."""
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_methodology(path: Path | None = None) -> Methodology:
    """Load, validate, and approve the packaged methodology configuration.

    Args:
        path: Optional YAML path, primarily for validation tests.

    Returns:
        The approved methodology constants and SHA-256 of parsed YAML content.

    Raises:
        MethodologyConfigError: If fields have invalid types or values.
    """
    try:
        payload, source = read_config_bytes(path, "methodology.yaml")
        content = load_yaml_bytes(payload, source=source)
    except (OSError, ValueError) as error:
        raise MethodologyConfigError(str(error)) from error

    if not isinstance(content, dict) or set(content) != set(_APPROVED_CONSTANTS):
        raise MethodologyConfigError(
            "Methodology configuration has missing or unexpected fields."
        )
    for field in ("methodology_version", "calendar", "exchange_calendars_version"):
        if not isinstance(content[field], str) or not content[field]:
            raise MethodologyConfigError(f"{field} must be a non-empty string.")
    start_value = content["price_history_start"]
    if not isinstance(start_value, str):
        raise MethodologyConfigError("price_history_start must be an ISO date string.")
    try:
        start_date = date.fromisoformat(start_value)
    except ValueError as error:
        raise MethodologyConfigError(
            "price_history_start must use YYYY-MM-DD format."
        ) from error
    if start_date.isoformat() != start_value:
        raise MethodologyConfigError("price_history_start must use YYYY-MM-DD format.")
    for field in _INTEGER_FIELDS:
        value = content[field]
        if type(value) is not int or value <= 0:
            raise MethodologyConfigError(f"{field} must be a positive integer.")
    for field in ("percentile_method", "tie_rule"):
        if not isinstance(content[field], str) or not content[field]:
            raise MethodologyConfigError(f"{field} must be a non-empty string.")

    methodology = Methodology(
        methodology_version=str(content["methodology_version"]),
        calendar=str(content["calendar"]),
        exchange_calendars_version=str(content["exchange_calendars_version"]),
        price_history_start=start_date,
        momentum_horizon_sessions=int(content["momentum_horizon_sessions"]),
        volatility_return_count=int(content["volatility_return_count"]),
        volatility_ddof=int(content["volatility_ddof"]),
        annualization_sessions=int(content["annualization_sessions"]),
        reference_window_sessions=int(content["reference_window_sessions"]),
        minimum_reference_observations=int(content["minimum_reference_observations"]),
        percentile_method=str(content["percentile_method"]),
        tie_rule=str(content["tie_rule"]),
        sha256=_canonical_sha256(content),
    )
    methodology.assert_approved_constants()
    return methodology
