"""Command-line interface for offline evaluation and the guarded data pipelines."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from etf_monitor.arbitrage.band import arbitrage_band
from etf_monitor.arbitrage.basket import basket_fair_value, required_symbols
from etf_monitor.arbitrage.costs import load_fund_costs
from etf_monitor.arbitrage.flows import (
    creation_redemption_flow_proxy,
    premium_flow_lead_lag,
)
from etf_monitor.arbitrage.premium import premium_discount
from etf_monitor.arbitrage.response import forward_share_response, summarize_response
from etf_monitor.arbitrage.splits import SplitEvent, load_split_events
from etf_monitor.arbitrage.spread import abdi_ranaldo_spread
from etf_monitor.bundles import (
    VintageComparison,
    compare_bundles,
    verify_bundle,
    write_bundle,
)
from etf_monitor.demo import build_demo_prices
from etf_monitor.evaluation import EvaluationError, evaluate
from etf_monitor.holdings.acquire import fetch_holdings_batch
from etf_monitor.holdings.component_prices import (
    empty_component_frame,
    fetch_component_prices,
    load_component_vintage,
    write_component_vintage,
)
from etf_monitor.holdings.store import load_holdings_vintage, write_holdings_vintage
from etf_monitor.issuer_nav.acquire import NavBatchResult, fetch_nav_batch
from etf_monitor.issuer_nav.store import load_nav_vintage, write_nav_vintage
from etf_monitor.methodology import load_methodology
from etf_monitor.prices.acquire import BatchResult, fetch_price_batch
from etf_monitor.prices.schema import validate_canonical
from etf_monitor.prices.store import (
    StorePaths,
    _read_canonical_parquet,
    persist_batch,
)
from etf_monitor.providers.transport import (
    UrllibChartTransport,
    UrllibSsgaHoldingsTransport,
    UrllibSsgaNavHistoryTransport,
)
from etf_monitor.universe import Universe, load_arbitrage_universe, load_universe

_LOGGER = logging.getLogger(__name__)
_UNIVERSES = ("core", "arbitrage")


def _universe(name: str) -> Universe:
    return load_arbitrage_universe() if name == "arbitrage" else load_universe()


def _universe_root(data_root: Path, name: str) -> Path:
    """Keep the arbitrage universe in its own store so stores never mix."""
    return data_root if name == "core" else data_root / name


def _iso_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from error
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="etf-monitor")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--prices", required=True, type=Path)
    evaluation.add_argument("--instant", required=True)
    evaluation.add_argument("--out", required=True, type=Path)

    verify = commands.add_parser("verify")
    verify.add_argument("bundle", type=Path)

    compare = commands.add_parser("compare")
    compare.add_argument("bundle_a", type=Path)
    compare.add_argument("bundle_b", type=Path)

    demo = commands.add_parser("demo-bundle")
    demo.add_argument("--out", required=True, type=Path)
    demo.add_argument("--instant", required=True)

    fetch = commands.add_parser("fetch-prices")
    fetch.add_argument("--end-exclusive", required=True, type=_iso_date)
    fetch.add_argument("--start", default="2018-01-01", type=_iso_date)
    fetch.add_argument("--data-root", required=True, type=Path)
    fetch.add_argument("--authorize-live-request", action="store_true")
    fetch.add_argument("--universe", choices=_UNIVERSES, default="core")

    fetch_nav = commands.add_parser("fetch-issuer-nav")
    fetch_nav.add_argument("--data-root", required=True, type=Path)
    fetch_nav.add_argument("--universe", choices=_UNIVERSES, default="core")
    fetch_nav.add_argument("--authorize-live-request", action="store_true")

    arbitrage = commands.add_parser("premium-flows")
    arbitrage.add_argument("--prices", required=True, type=Path)
    arbitrage.add_argument("--nav-vintage", required=True, type=Path)
    arbitrage.add_argument("--price-raw-dir", required=True, type=Path)
    arbitrage.add_argument("--universe", choices=_UNIVERSES, default="core")
    arbitrage.add_argument("--out", required=True, type=Path)

    band = commands.add_parser("arbitrage-band")
    band.add_argument("--prices", required=True, type=Path)
    band.add_argument("--premium-flows", required=True, type=Path)
    band.add_argument("--universe", choices=_UNIVERSES, default="core")
    band.add_argument("--horizon", type=int, default=5)
    band.add_argument("--out", required=True, type=Path)

    fetch_holdings = commands.add_parser("fetch-holdings")
    fetch_holdings.add_argument("--data-root", required=True, type=Path)
    fetch_holdings.add_argument("--universe", choices=_UNIVERSES, default="core")
    fetch_holdings.add_argument("--authorize-live-request", action="store_true")

    components = commands.add_parser("fetch-component-prices")
    components.add_argument("--holdings-vintage", required=True, type=Path)
    components.add_argument("--universe", choices=_UNIVERSES, default="core")
    components.add_argument("--start", required=True, type=_iso_date)
    components.add_argument("--end-exclusive", required=True, type=_iso_date)
    components.add_argument("--data-root", required=True, type=Path)
    components.add_argument("--authorize-live-request", action="store_true")

    basket = commands.add_parser("basket-value")
    basket.add_argument("--holdings-vintage", required=True, type=Path)
    basket.add_argument("--component-vintage", type=Path)
    basket.add_argument("--nav-vintage", required=True, type=Path)
    basket.add_argument("--universe", choices=_UNIVERSES, default="core")
    basket.add_argument("--out", required=True, type=Path)
    return parser


def _parse_instant(value: str) -> datetime | None:
    if value == "now":
        return None
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--instant must be an ISO-8601 timestamp with an offset or 'now'."
        ) from error
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise argparse.ArgumentTypeError("--instant must include a UTC offset.")
    return instant


def _capture_instant(value: str) -> datetime:
    parsed = _parse_instant(value)
    return datetime.now(UTC) if parsed is None else parsed


def _load_prices(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise EvaluationError(f"Canonical prices file does not exist: {path}.")
    return _read_canonical_parquet(path)


def _compare_json(comparison: VintageComparison) -> str:
    data = {
        "input_row_counts": comparison.input_row_counts.to_dict(orient="records"),
        "input_field_changes": comparison.input_field_changes.to_dict(orient="records"),
        "input_changes_by_ticker": comparison.input_changes_by_ticker.to_dict(
            orient="records"
        ),
        "signal_changes_by_ticker": comparison.signal_changes_by_ticker.to_dict(
            orient="records"
        ),
        "target_session_changes": comparison.target_session_changes.to_dict(
            orient="records"
        ),
    }
    return json.dumps(data, default=str, allow_nan=False, sort_keys=True)


def _log_batch(batch: BatchResult) -> None:
    counts = {
        outcome: sum(status.status == outcome for status in batch.statuses)
        for outcome in ("success", "empty", "failed")
    }
    _LOGGER.info(
        "Price batch %s: %d success, %d empty, %d failed",
        batch.batch_id,
        counts["success"],
        counts["empty"],
        counts["failed"],
    )
    for status in batch.statuses:
        if status.status != "success":
            _LOGGER.warning("%s: %s (%s)", status.ticker, status.status, status.reason)


def _log_nav_batch(batch: NavBatchResult) -> None:
    for status in batch.statuses:
        _LOGGER.info(
            "%s %s rows=%d missing_sessions=%d missing_shares=%d %s",
            status.ticker,
            status.status,
            status.row_count,
            len(status.missing_sessions),
            status.missing_shares_outstanding,
            status.reason_code or "",
        )


def _write_premium_flows(
    prices_path: Path,
    vintage_path: Path,
    price_raw_dir: Path,
    universe: Universe,
    out: Path,
) -> Path:
    prices = _load_prices(prices_path)
    validate_canonical(prices, universe)
    nav = load_nav_vintage(vintage_path, universe).frame
    splits = load_split_events(price_raw_dir, nav["ticker"].unique())
    premium = premium_discount(prices, nav, splits)
    flows = creation_redemption_flow_proxy(nav, splits)
    lead_lag = premium_flow_lead_lag(premium, flows)
    out.mkdir(parents=True, exist_ok=False)
    premium.to_parquet(out / "premium_discount.parquet", index=False)
    flows.to_parquet(out / "flow_proxy.parquet", index=False)
    lead_lag.to_csv(out / "premium_flow_lead_lag.csv", index=False)
    summary = {
        "prices": str(prices_path),
        "nav_vintage": str(vintage_path),
        "price_raw_dir": str(price_raw_dir),
        "split_events": [
            f"{event.ticker} {event.date.isoformat()} "
            f"{event.numerator}:{event.denominator}"
            for event in splits
        ],
        "universe_sha256": universe.sha256,
        "premium_status_counts": premium["status"].value_counts().to_dict(),
        "flow_status_counts": flows["status"].value_counts().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return out


def _write_arbitrage_band(
    prices_path: Path,
    premium_flows: Path,
    universe: Universe,
    horizon: int,
    out: Path,
) -> Path:
    prices = _load_prices(prices_path)
    validate_canonical(prices, universe)
    premium = pd.read_parquet(premium_flows / "premium_discount.parquet")
    flows = pd.read_parquet(premium_flows / "flow_proxy.parquet")
    costs, asset_class = load_fund_costs()
    spread = abdi_ranaldo_spread(prices)
    band = arbitrage_band(premium, spread, costs)
    response = forward_share_response(band, flows, horizon=horizon)
    by_class = summarize_response(response, asset_class)
    out.mkdir(parents=True, exist_ok=False)
    band.to_parquet(out / "arbitrage_band.parquet", index=False)
    response.to_parquet(out / "forward_share_response.parquet", index=False)
    by_class.to_csv(out / "response_by_asset_class.csv", index=False)
    by_fund = summarize_response(response, {t: t for t in asset_class})
    by_fund.to_csv(out / "response_by_fund.csv", index=False)
    ok = band[band["status"] == "ok"]
    summary = {
        "prices": str(prices_path),
        "premium_flows": str(premium_flows),
        "horizon_sessions": horizon,
        "universe_sha256": universe.sha256,
        "status_counts": band["status"].value_counts().to_dict(),
        "side_counts": {
            str(ticker): rows["side"].value_counts().to_dict()
            for ticker, rows in ok.groupby("ticker", sort=True)
        },
        "dislocations": {
            str(ticker): {
                "outside_band": int(rows["side"].isin(["premium", "discount"]).sum()),
                "persistent": int(rows["persistent"].fillna(False).sum()),
                "stale_nav": bool(rows["stale_nav"].iloc[0]),
            }
            for ticker, rows in ok.groupby("ticker", sort=True)
        },
        "median_band_bps": {
            str(ticker): float(rows["band_bps"].median())
            for ticker, rows in ok.groupby("ticker", sort=True)
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return out


def _write_basket_value(
    holdings_path: Path,
    component_path: Path | None,
    nav_path: Path,
    universe: Universe,
    out: Path,
) -> Path:
    holdings = load_holdings_vintage(holdings_path, universe).frame
    nav = load_nav_vintage(nav_path, universe).frame
    prices = empty_component_frame()
    splits: tuple[SplitEvent, ...] = ()
    failed: tuple[str, ...] = ()
    if component_path is not None:
        components = load_component_vintage(component_path)
        prices, splits = components.frame, components.splits
        failed = components.failed_symbols
    elif required_symbols(holdings):
        raise ValueError("--component-vintage is required for equity baskets.")
    basket = basket_fair_value(holdings, prices, nav, splits)
    out.mkdir(parents=True, exist_ok=False)
    basket.to_csv(out / "basket_value.csv", index=False)
    summary: dict[str, object] = {
        "holdings_vintage": str(holdings_path),
        "component_vintage": None if component_path is None else str(component_path),
        "nav_vintage": str(nav_path),
        "universe_sha256": universe.sha256,
        "failed_component_symbols": list(failed),
        "status_counts": basket["status"].value_counts().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run offline evaluation, bundle, or explicitly authorized fetch commands.

    Args:
        argv: Optional command-line arguments, primarily for tests.

    Returns:
        Exit code 0 on success, 3 on partial fetch success, or 2 on failure.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "evaluate":
            instant = _capture_instant(args.instant)
            result = evaluate(_load_prices(args.prices), instant)
            print(write_bundle(result, args.out))
            return 0
        if args.command == "verify":
            bundle = verify_bundle(args.bundle)
            print(f"verified {bundle.run_id}")
            return 0
        if args.command == "compare":
            comparison = compare_bundles(
                verify_bundle(args.bundle_a),
                verify_bundle(args.bundle_b),
            )
            print(_compare_json(comparison))
            return 0
        if args.command == "demo-bundle":
            instant = _capture_instant(args.instant)
            result = evaluate(build_demo_prices(instant), instant)
            print(write_bundle(result, args.out))
            return 0
        if args.command == "fetch-prices":
            if not args.authorize_live_request:
                _LOGGER.error(
                    "Refusing live Yahoo requests without --authorize-live-request."
                )
                return 2
            if args.start >= args.end_exclusive:
                _LOGGER.error("--start must be earlier than --end-exclusive.")
                return 2
            universe = _universe(args.universe)
            price_root = _universe_root(args.data_root, args.universe)
            transport = UrllibChartTransport()
            batch = fetch_price_batch(
                universe=universe,
                start=args.start,
                end_exclusive=args.end_exclusive,
                transport=transport,
                clock=lambda: datetime.now(UTC),
                raw_dir=price_root / "raw/prices",
            )
            _log_batch(batch)
            persisted = persist_batch(batch, StorePaths.under(price_root), universe)
            if not persisted.persisted:
                return 2
            return (
                0 if all(status.status == "success" for status in batch.statuses) else 3
            )
        if args.command == "fetch-issuer-nav":
            if not args.authorize_live_request:
                _LOGGER.error(
                    "Refusing live issuer requests without --authorize-live-request."
                )
                return 2
            universe = _universe(args.universe)
            nav_batch = fetch_nav_batch(
                universe=universe,
                history_start=load_methodology().price_history_start,
                transport=UrllibSsgaNavHistoryTransport(),
                clock=lambda: datetime.now(UTC),
            )
            _log_nav_batch(nav_batch)
            print(
                write_nav_vintage(
                    nav_batch, args.data_root / "issuer_nav" / args.universe, universe
                )
            )
            requested = [s for s in nav_batch.statuses if s.status != "unavailable"]
            succeeded = [s for s in requested if s.status == "success"]
            if not succeeded:
                return 2
            return 0 if len(succeeded) == len(requested) else 3
        if args.command in ("fetch-holdings", "fetch-component-prices") and (
            not args.authorize_live_request
        ):
            _LOGGER.error("Refusing live requests without --authorize-live-request.")
            return 2
        if args.command == "fetch-holdings":
            universe = _universe(args.universe)
            holdings_batch = fetch_holdings_batch(
                universe=universe,
                transport=UrllibSsgaHoldingsTransport(),
                clock=lambda: datetime.now(UTC),
            )
            for holding_status in holdings_batch.statuses:
                _LOGGER.info(
                    "%s %s as_of=%s rows=%d %s",
                    holding_status.ticker,
                    holding_status.status,
                    holding_status.as_of,
                    holding_status.row_count,
                    holding_status.reason_code or "",
                )
            print(
                write_holdings_vintage(
                    holdings_batch,
                    args.data_root / "holdings" / args.universe,
                    universe,
                )
            )
            requested_holdings = [
                s for s in holdings_batch.statuses if s.status != "unavailable"
            ]
            if not any(s.status == "success" for s in requested_holdings):
                return 2
            return 0 if all(s.status == "success" for s in requested_holdings) else 3
        if args.command == "fetch-component-prices":
            holdings = load_holdings_vintage(
                args.holdings_vintage, _universe(args.universe)
            ).frame
            component_batch = fetch_component_prices(
                required_symbols(holdings),
                args.start,
                args.end_exclusive,
                UrllibChartTransport(),
                lambda: datetime.now(UTC),
            )
            failed_components = [
                s.symbol for s in component_batch.statuses if s.status != "success"
            ]
            _LOGGER.info(
                "Component batch %s: %d requested, %d failed %s",
                component_batch.batch_id,
                len(component_batch.statuses),
                len(failed_components),
                failed_components,
            )
            print(
                write_component_vintage(
                    component_batch, args.data_root / "component_prices"
                )
            )
            return 0 if not failed_components else 3
        if args.command == "basket-value":
            print(
                _write_basket_value(
                    args.holdings_vintage,
                    args.component_vintage,
                    args.nav_vintage,
                    _universe(args.universe),
                    args.out,
                )
            )
            return 0
        if args.command == "arbitrage-band":
            print(
                _write_arbitrage_band(
                    args.prices,
                    args.premium_flows,
                    _universe(args.universe),
                    args.horizon,
                    args.out,
                )
            )
            return 0
        if args.command == "premium-flows":
            print(
                _write_premium_flows(
                    args.prices,
                    args.nav_vintage,
                    args.price_raw_dir,
                    _universe(args.universe),
                    args.out,
                )
            )
            return 0
    except (OSError, ValueError) as error:
        _LOGGER.error("%s", error)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
