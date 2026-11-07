"""Load and validate the fixed ETF universe."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from etf_monitor.configuration import load_yaml_bytes, read_config_bytes

EXPECTED_UNIVERSE_SIZE = 24
CATEGORIES = (
    "Broad market",
    "Technology / growth",
    "Financials",
    "Energy",
    "Healthcare",
    "Consumer",
    "Industrials / materials",
    "Thematic",
)
ARBITRAGE_UNIVERSE_SIZE = 12
ARBITRAGE_CATEGORIES = (
    "U.S. fixed income",
    "International fixed income",
    "International equity",
    "Global equity",
)
_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")


class UniverseConfigError(ValueError):
    """Indicate that the configured ETF universe violates its contract."""


@dataclass(frozen=True, slots=True)
class Etf:
    """Represent one ETF in the version-controlled research universe."""

    ticker: str
    name: str
    category: str


@dataclass(frozen=True, slots=True)
class Universe:
    """Represent a validated, ordered ETF universe and its content hash."""

    version: int
    etfs: tuple[Etf, ...]
    sha256: str


def _canonical_sha256(content: object) -> str:
    """Hash parsed YAML content using canonical JSON serialization."""
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_universe(path: Path | None = None) -> Universe:
    """Load and validate the packaged 24-ETF crowding universe.

    Args:
        path: Optional YAML path, primarily for validation tests.

    Returns:
        The fixed ordered ETF universe and SHA-256 of its parsed content.

    Raises:
        UniverseConfigError: If the document, entries, or identifiers are invalid.
    """
    return _load_universe_file(
        path, "universe.yaml", EXPECTED_UNIVERSE_SIZE, CATEGORIES
    )


def load_arbitrage_universe(path: Path | None = None) -> Universe:
    """Load and validate the packaged ETF arbitrage-monitor universe.

    The arbitrage universe is a separate, fixed list of fixed-income and
    international funds whose issuer publishes daily NAV and shares
    outstanding. It never changes the crowding universe or its hash.

    Args:
        path: Optional YAML path, primarily for validation tests.

    Returns:
        The fixed ordered arbitrage universe and SHA-256 of its parsed content.

    Raises:
        UniverseConfigError: If the document, entries, or identifiers are invalid.
    """
    return _load_universe_file(
        path,
        "arbitrage_universe.yaml",
        ARBITRAGE_UNIVERSE_SIZE,
        ARBITRAGE_CATEGORIES,
    )


def _load_universe_file(
    path: Path | None,
    resource_name: str,
    expected_size: int,
    categories: tuple[str, ...],
) -> Universe:
    try:
        payload, source = read_config_bytes(path, resource_name)
        content = load_yaml_bytes(payload, source=source)
    except (OSError, ValueError) as error:
        raise UniverseConfigError(str(error)) from error

    if not isinstance(content, dict) or set(content) != {"universe_version", "etfs"}:
        raise UniverseConfigError(
            "Universe configuration must contain only universe_version and etfs."
        )
    version = content["universe_version"]
    entries = content["etfs"]
    if type(version) is not int or version < 1:
        raise UniverseConfigError("universe_version must be a positive integer.")
    if not isinstance(entries, list):
        raise UniverseConfigError("etfs must be a list.")
    if len(entries) != expected_size:
        raise UniverseConfigError(
            f"Universe must contain exactly {expected_size} ETFs; "
            f"received {len(entries)}."
        )

    parsed_etfs: list[Etf] = []
    tickers: set[str] = set()
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "ticker",
            "name",
            "category",
        }:
            raise UniverseConfigError(
                f"ETF at position {position} must contain only ticker, name, "
                "and category."
            )
        ticker = entry["ticker"]
        name = entry["name"]
        category = entry["category"]
        if not isinstance(ticker, str) or _TICKER_PATTERN.fullmatch(ticker) is None:
            raise UniverseConfigError(
                f"ETF at position {position} has an invalid ticker."
            )
        if ticker in tickers:
            raise UniverseConfigError(f"Duplicate ETF ticker {ticker!r}.")
        if not isinstance(name, str) or not name.strip():
            raise UniverseConfigError(
                f"ETF {ticker} must have a non-empty name string."
            )
        if not isinstance(category, str) or category not in categories:
            raise UniverseConfigError(
                f"ETF {ticker} has an unknown category {category!r}."
            )
        tickers.add(ticker)
        parsed_etfs.append(Etf(ticker=ticker, name=name, category=category))

    return Universe(
        version=version,
        etfs=tuple(parsed_etfs),
        sha256=_canonical_sha256(content),
    )
