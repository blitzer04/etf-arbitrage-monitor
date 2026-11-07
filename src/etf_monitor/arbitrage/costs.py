"""Load the packaged creation-unit, fee and basket-cost inputs for the band."""

from __future__ import annotations

from pathlib import Path

from etf_monitor.arbitrage.band import FundCosts
from etf_monitor.configuration import load_yaml_bytes, read_config_bytes

_FUND_KEYS = {
    "asset_class",
    "creation_unit_shares",
    "unit_source",
    "creation_fee_usd",
    "fee_source",
}


class CostConfigError(ValueError):
    """Indicate that the arbitrage cost configuration violates its contract."""


def load_fund_costs(
    path: Path | None = None,
) -> tuple[dict[str, FundCosts], dict[str, str]]:
    """Return cost inputs by ticker and the asset class of each ticker."""
    try:
        payload, name = read_config_bytes(path, "arbitrage_costs.yaml")
        content = load_yaml_bytes(payload, source=name)
    except (OSError, ValueError) as error:
        raise CostConfigError(str(error)) from error
    if not isinstance(content, dict) or set(content) != {
        "costs_version",
        "sources",
        "asset_classes",
        "funds",
    }:
        raise CostConfigError(
            "Cost config needs costs_version, sources, asset_classes, funds."
        )
    sources, classes, funds = (
        content["sources"],
        content["asset_classes"],
        content["funds"],
    )
    if not all(isinstance(x, dict) and x for x in (sources, classes, funds)):
        raise CostConfigError(
            "sources, asset_classes and funds must be non-empty mappings."
        )
    for cls, entry in classes.items():
        if not isinstance(entry, dict) or set(entry) != {
            "basket_cost_bps",
            "etf_half_spread_bps",
            "stale_nav",
        }:
            raise CostConfigError(
                f"Asset class {cls!r} needs basket_cost_bps, etf_half_spread_bps, stale_nav."
            )
        for key in ("basket_cost_bps", "etf_half_spread_bps"):
            value = entry[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or value < 0
            ):
                raise CostConfigError(f"Asset class {cls!r} {key} must be >= 0.")
        bps = entry["basket_cost_bps"]
        if isinstance(bps, bool) or not isinstance(bps, int | float) or bps < 0:
            raise CostConfigError(f"Asset class {cls!r} basket_cost_bps must be >= 0.")
        if not isinstance(entry["stale_nav"], bool):
            raise CostConfigError(f"Asset class {cls!r} stale_nav must be boolean.")
    costs: dict[str, FundCosts] = {}
    asset_class: dict[str, str] = {}
    for ticker, entry in funds.items():
        if not isinstance(entry, dict) or set(entry) != _FUND_KEYS:
            raise CostConfigError(f"{ticker} must define exactly {sorted(_FUND_KEYS)}.")
        cls = entry["asset_class"]
        if cls not in classes:
            raise CostConfigError(f"{ticker} has unknown asset class {cls!r}.")
        unit, fee = entry["creation_unit_shares"], entry["creation_fee_usd"]
        if isinstance(unit, bool) or not isinstance(unit, int) or unit <= 0:
            raise CostConfigError(
                f"{ticker} creation_unit_shares must be a positive integer."
            )
        if fee is not None and (
            isinstance(fee, bool) or not isinstance(fee, int | float) or fee < 0
        ):
            raise CostConfigError(f"{ticker} creation_fee_usd must be null or >= 0.")
        for key in ("unit_source", "fee_source"):
            if entry[key] not in sources:
                raise CostConfigError(
                    f"{ticker} {key} {entry[key]!r} is not a listed source."
                )
        costs[ticker] = FundCosts(
            ticker=ticker,
            creation_unit_shares=unit,
            creation_fee_usd=None if fee is None else float(fee),
            basket_cost_bps=float(classes[cls]["basket_cost_bps"]),
            half_spread_bps=float(classes[cls]["etf_half_spread_bps"]),
            stale_nav=classes[cls]["stale_nav"],
        )
        asset_class[ticker] = cls
    return costs, asset_class
