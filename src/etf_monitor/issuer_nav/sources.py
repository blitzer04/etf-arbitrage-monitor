"""Configured issuer NAV-history source for every monitored ETF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from etf_monitor.configuration import load_yaml_bytes, read_config_bytes
from etf_monitor.providers.ssga_nav import SSGA_NAV_SOURCE
from etf_monitor.universe import Universe

SUPPORTED_NAV_SOURCES = (SSGA_NAV_SOURCE,)


class NavSourceConfigError(ValueError):
    """Indicate that the issuer NAV source configuration violates its contract."""


@dataclass(frozen=True, slots=True)
class NavSource:
    """The NAV-history source for one ETF, or the reason none is available."""

    ticker: str
    source: str | None
    reason: str | None


def load_nav_sources(path: Path | None = None) -> dict[str, NavSource]:
    """Load the packaged per-ETF issuer NAV source mapping.

    Args:
        path: Optional YAML path, primarily for validation tests.

    Returns:
        Mapping from ticker to its configured source.

    Raises:
        NavSourceConfigError: If the document or any entry is invalid.
    """
    try:
        payload, source_name = read_config_bytes(path, "issuer_nav_sources.yaml")
        content = load_yaml_bytes(payload, source=source_name)
    except (OSError, ValueError) as error:
        raise NavSourceConfigError(str(error)) from error
    if not isinstance(content, dict) or set(content) != {"sources_version", "funds"}:
        raise NavSourceConfigError(
            "NAV source configuration must contain only sources_version and funds."
        )
    funds = content["funds"]
    if not isinstance(funds, dict) or not funds:
        raise NavSourceConfigError("funds must be a non-empty mapping.")
    sources: dict[str, NavSource] = {}
    for ticker, entry in funds.items():
        if not isinstance(ticker, str) or not isinstance(entry, dict):
            raise NavSourceConfigError(f"Invalid NAV source entry {ticker!r}.")
        if set(entry) != {"source", "reason"}:
            raise NavSourceConfigError(f"{ticker} must define only source and reason.")
        source, reason = entry["source"], entry["reason"]
        if source is None:
            if not isinstance(reason, str) or not reason.strip():
                raise NavSourceConfigError(
                    f"{ticker} has no source and must state a reason."
                )
        elif source not in SUPPORTED_NAV_SOURCES or reason is not None:
            raise NavSourceConfigError(
                f"{ticker} must use a supported source with a null reason."
            )
        sources[ticker] = NavSource(ticker=ticker, source=source, reason=reason)
    return sources


def nav_sources_for(
    universe: Universe, sources: dict[str, NavSource] | None = None
) -> tuple[NavSource, ...]:
    """Return the configured NAV source for each universe ETF in universe order.

    Args:
        universe: Validated ETF universe.
        sources: Optional preloaded mapping; defaults to the packaged file.

    Returns:
        One source entry per universe ETF.

    Raises:
        NavSourceConfigError: If any universe ETF has no configured entry.
    """
    mapping = load_nav_sources() if sources is None else sources
    missing = [etf.ticker for etf in universe.etfs if etf.ticker not in mapping]
    if missing:
        raise NavSourceConfigError(f"No NAV source configured for {missing}.")
    return tuple(mapping[etf.ticker] for etf in universe.etfs)
