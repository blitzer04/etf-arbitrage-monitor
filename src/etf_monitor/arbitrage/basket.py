"""Holdings-based basket value per ETF share compared with official NAV.

For an equity fund with holdings published ``As of`` session D::

    basket_value_usd = sum(shares_held * close_as_traded) + cash_usd
    basket_value_per_share = basket_value_usd / shares_outstanding
    basket_vs_nav_bps = 10_000 * (basket_value_per_share / nav - 1)

``close_as_traded`` is the component's Yahoo close on the valuation session
times its later split factor. Two alignments are reported because the date
semantics of the snapshot are not documented: ``same_session`` values the
D holdings at D closes against NAV and shares outstanding for D;
``next_session`` values them at the next XNYS session's closes against that
session's NAV, as if the snapshot were the opening position for it.

Line classes:

- ``cash``: no ticker and a USD cash or State Street USD government money
  market identifier; valued at the published amount (unit value 1).
- ``futures``: index futures lines; valued at zero because variation margin
  settles daily into cash. Their open notional is not part of NAV.
- ``priced``: a USD line with a Yahoo-mappable ticker.
- ``unpriced``: everything else (rights, earn-outs, non-USD cash); excluded
  from the basket and reported as ``unpriced_weight_pct`` so the gap stays
  visible instead of being filled from the issuer's own weights.

``implied_shares_outstanding`` (basket value / NAV) separates a share-count
timing mismatch between the snapshot and the reported shares outstanding
from a valuation gap.

Accrued fees and other liabilities are not in the holdings file, so a small
positive basket gap is expected. A fund with any non-USD security is not
valued (no FX or foreign-close alignment yet). Fixed-income funds have no
independent price source; they get an ``issuer_market_value`` consistency row
(sum of the issuer's own market values per share against NAV), which is not
an independent fair value.

A priced line without a close on the valuation session makes the basket
null with status ``missing_component_price``; no price is carried forward.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from etf_monitor.arbitrage.splits import SplitEvent, split_factors
from etf_monitor.holdings.component_prices import yahoo_symbol
from etf_monitor.xnys import next_session

CASH_IDENTIFIERS = {"999USDZ92", "924QSGII3"}
_FUTURES_NAME = re.compile(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}$")
ALIGNMENTS = ("same_session", "next_session")
BASKET_COLUMNS = (
    "ticker",
    "as_of",
    "alignment",
    "valuation_date",
    "valuation_method",
    "status",
    "nav",
    "shares_outstanding",
    "security_value_usd",
    "cash_value_usd",
    "lines",
    "priced_lines",
    "futures_lines",
    "unpriced_lines",
    "missing_price_lines",
    "unpriced_weight_pct",
    "weight_consistency_max_abs_pct",
    "basket_value_usd",
    "basket_value_per_share",
    "basket_vs_nav_bps",
    "implied_shares_outstanding",
)


def classify_lines(holdings: pd.DataFrame) -> pd.Series:
    """Return ``cash``/``futures``/``priced``/``unpriced`` per equity line."""
    ticker = holdings["security_ticker"]
    currency = holdings["local_currency"]
    usd = currency.eq("USD").fillna(False).to_numpy(dtype=bool)
    no_ticker = ticker.isna().to_numpy()
    cash = no_ticker & usd & holdings["identifier"].isin(CASH_IDENTIFIERS).to_numpy()
    futures = ~no_ticker & np.array(
        [
            _FUTURES_NAME.search(str(name).strip()) is not None
            for name in holdings["name"]
        ]
    )
    mappable = np.array(
        [
            yahoo_symbol(None if pd.isna(value) else str(value)) is not None
            for value in ticker
        ]
    )
    priced = usd & mappable & ~futures
    return pd.Series(
        np.select([cash, futures, priced], ["cash", "futures", "priced"], "unpriced"),
        index=holdings.index,
        dtype="string",
    )


def required_symbols(holdings: pd.DataFrame) -> tuple[str, ...]:
    """Return the Yahoo symbols needed to value every supported equity fund."""
    symbols: set[str] = set()
    for _, fund in holdings.groupby("ticker", sort=True):
        if _unsupported_reason(fund) is not None:
            continue
        priced = fund.loc[classify_lines(fund).eq("priced").to_numpy()]
        symbols.update(
            str(yahoo_symbol(str(value))) for value in priced["security_ticker"]
        )
    return tuple(sorted(symbols))


def _unsupported_reason(fund: pd.DataFrame) -> str | None:
    if fund["layout"].iloc[0] != "equity":
        return "fixed_income"
    securities = fund.loc[fund["security_ticker"].notna().to_numpy()]
    if securities["local_currency"].ne("USD").fillna(True).any():
        return "unsupported_non_usd_securities"
    return None


def _nav_on(nav: pd.DataFrame, ticker: str, day: pd.Timestamp) -> tuple[float, float]:
    row = nav.loc[nav["ticker"].eq(ticker).to_numpy() & nav["date"].eq(day).to_numpy()]
    if row.empty:
        return np.nan, np.nan
    shares = row["shares_outstanding"].iloc[0]
    return float(row["nav"].iloc[0]), np.nan if pd.isna(shares) else float(shares)


def _empty_record(
    ticker: str, as_of: pd.Timestamp, alignment: str
) -> dict[str, object]:
    record: dict[str, object] = dict.fromkeys(BASKET_COLUMNS, np.nan)
    record.update(ticker=ticker, as_of=as_of, alignment=alignment)
    return record


def _per_share(record: dict[str, object], total: float) -> None:
    nav, shares = float(record["nav"]), float(record["shares_outstanding"])  # type: ignore[arg-type]
    if np.isnan(nav) or np.isnan(shares) or shares <= 0:
        record["status"] = "missing_nav"
        return
    record["basket_value_usd"] = total
    record["basket_value_per_share"] = total / shares
    record["basket_vs_nav_bps"] = 10_000.0 * (total / shares / nav - 1.0)
    record["implied_shares_outstanding"] = total / nav
    record["status"] = "ok"


def basket_fair_value(
    holdings: pd.DataFrame,
    component_prices: pd.DataFrame,
    nav: pd.DataFrame,
    splits: Sequence[SplitEvent] = (),
) -> pd.DataFrame:
    """Value every fund's published holdings under both alignments.

    Args:
        holdings: Canonical holdings rows (one snapshot per fund).
        component_prices: ``symbol``, ``date``, ``close`` rows from a
            component vintage.
        nav: Canonical issuer NAV rows.
        splits: Component split events from the same component vintage.

    Returns:
        One row per fund and alignment with columns :data:`BASKET_COLUMNS`.
    """
    records: list[dict[str, object]] = []
    for ticker, fund in holdings.groupby("ticker", sort=True):
        fund = fund.reset_index(drop=True)
        as_of = pd.Timestamp(fund["as_of"].iloc[0])
        reason = _unsupported_reason(fund)
        for alignment in ALIGNMENTS:
            day = as_of if alignment == "same_session" else next_session(as_of)
            record = _empty_record(str(ticker), as_of, alignment)
            record["valuation_date"] = day
            record["lines"] = len(fund)
            record["nav"], record["shares_outstanding"] = _nav_on(nav, str(ticker), day)
            if reason == "fixed_income":
                _fixed_income(record, fund)
            elif reason is not None:
                record["valuation_method"] = "none"
                record["status"] = reason
            else:
                _equity(record, fund, component_prices, splits, day)
            records.append(record)
    frame = pd.DataFrame.from_records(records, columns=list(BASKET_COLUMNS))
    for column in ("as_of", "valuation_date"):
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    for column in ("ticker", "alignment", "valuation_method", "status"):
        frame[column] = frame[column].astype("string")
    return frame


def _fixed_income(record: dict[str, object], fund: pd.DataFrame) -> None:
    values = fund["market_value"].to_numpy(dtype=np.float64)
    total = float(values.sum())
    record["valuation_method"] = "issuer_market_value"
    record["security_value_usd"] = total
    record["cash_value_usd"] = 0.0
    implied = 100.0 * values / total if total else np.full(len(values), np.nan)
    record["weight_consistency_max_abs_pct"] = float(
        np.max(np.abs(implied - fund["weight_pct"].to_numpy(dtype=np.float64)))
    )
    _per_share(record, total)


def _equity(
    record: dict[str, object],
    fund: pd.DataFrame,
    component_prices: pd.DataFrame,
    splits: Sequence[SplitEvent],
    day: pd.Timestamp,
) -> None:
    record["valuation_method"] = "component_close"
    classes = classify_lines(fund).to_numpy()
    quantity = fund["quantity"].to_numpy(dtype=np.float64)
    priced = fund.loc[classes == "priced"]
    symbols = pd.Series(
        [yahoo_symbol(str(value)) for value in priced["security_ticker"]],
        dtype="string",
    )
    closes = component_prices.loc[component_prices["date"].eq(day).to_numpy()]
    close_by_symbol = dict(zip(closes["symbol"], closes["close"], strict=True))
    close = np.array([close_by_symbol.get(s, np.nan) for s in symbols], dtype=float)
    factor = split_factors(symbols, pd.Series([day] * len(symbols)), splits)
    missing = int(np.isnan(close).sum())
    record.update(
        priced_lines=int((classes == "priced").sum()),
        futures_lines=int((classes == "futures").sum()),
        unpriced_lines=int((classes == "unpriced").sum()),
        missing_price_lines=missing,
        unpriced_weight_pct=float(
            fund["weight_pct"].to_numpy()[classes == "unpriced"].sum()
        ),
        cash_value_usd=float(quantity[classes == "cash"].sum()),
    )
    if missing:
        record["status"] = "missing_component_price"
        return
    security = float((quantity[classes == "priced"] * close * factor).sum())
    record["security_value_usd"] = security
    _per_share(record, security + float(record["cash_value_usd"]))  # type: ignore[arg-type]
