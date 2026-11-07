"""Descriptive exact-date dependence calculations for standalone signals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from etf_monitor.universe import Universe


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    centered_first = first.astype(np.float64) - np.mean(first, dtype=np.float64)
    centered_second = second.astype(np.float64) - np.mean(second, dtype=np.float64)
    numerator = float(np.sum(centered_first * centered_second, dtype=np.float64))
    first_square = float(np.sum(centered_first * centered_first, dtype=np.float64))
    second_square = float(np.sum(centered_second * centered_second, dtype=np.float64))
    estimate = numerator / math.sqrt(first_square * second_square)
    if not math.isfinite(estimate):
        raise ValueError("Dependence calculation produced a non-finite estimate.")
    return estimate


def _estimates(first: np.ndarray, second: np.ndarray) -> tuple[float, float, str]:
    if len(first) < 3:
        return float("nan"), float("nan"), "insufficient_pairs"
    if np.unique(first).size < 2 or np.unique(second).size < 2:
        return float("nan"), float("nan"), "constant_input"
    pearson = _pearson(first, second)
    ranked_first = pd.Series(first).rank(method="average").to_numpy(dtype=np.float64)
    ranked_second = pd.Series(second).rank(method="average").to_numpy(dtype=np.float64)
    spearman = _pearson(ranked_first, ranked_second)
    return pearson, spearman, "available"


def calculate_dependence(
    momentum: pd.DataFrame,
    volatility: pd.DataFrame,
    universe: Universe,
    sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate descriptive per-session and per-ETF percentile correlations.

    Pairs are exact ticker/session inner joins with both normalized percentiles
    available. Pearson is calculated from explicitly centered float64 vectors;
    Spearman is Pearson correlation of average pandas ranks. No inference or
    causal claim is made.

    Args:
        momentum: Momentum signal output for every configured ETF and session.
        volatility: Volatility signal output for every configured ETF/session.
        universe: Ordered configured ETF universe.
        sessions: Full XNYS evaluation grid.

    Returns:
        Cross-sectional-by-session and full-history-per-ETF tables.
    """
    momentum_pairs = momentum.loc[
        momentum["percentile"].notna(),
        ["ticker", "signal_date", "percentile"],
    ].rename(columns={"percentile": "momentum_percentile"})
    volatility_pairs = volatility.loc[
        volatility["percentile"].notna(),
        ["ticker", "signal_date", "percentile"],
    ].rename(columns={"percentile": "volatility_percentile"})
    paired = momentum_pairs.merge(
        volatility_pairs,
        on=["ticker", "signal_date"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    allowed_tickers = [etf.ticker for etf in universe.etfs]
    paired = paired.loc[paired["ticker"].isin(allowed_tickers)]

    cross_rows: list[dict[str, object]] = []
    for session in sessions:
        group = paired.loc[paired["signal_date"].eq(session)]
        first = group["momentum_percentile"].to_numpy(dtype=np.float64)
        second = group["volatility_percentile"].to_numpy(dtype=np.float64)
        pearson, spearman, status = _estimates(first, second)
        cross_rows.append(
            {
                "signal_date": session,
                "n_pairs": len(group),
                "pearson": pearson,
                "spearman": spearman,
                "status": status,
            }
        )

    ticker_rows: list[dict[str, object]] = []
    for ticker in allowed_tickers:
        group = paired.loc[paired["ticker"].eq(ticker)].sort_values(
            "signal_date", kind="mergesort"
        )
        first = group["momentum_percentile"].to_numpy(dtype=np.float64)
        second = group["volatility_percentile"].to_numpy(dtype=np.float64)
        pearson, spearman, status = _estimates(first, second)
        ticker_rows.append(
            {
                "ticker": ticker,
                "n_pairs": len(group),
                "pearson": pearson,
                "spearman": spearman,
                "status": status,
            }
        )

    cross_section = pd.DataFrame(
        cross_rows,
        columns=["signal_date", "n_pairs", "pearson", "spearman", "status"],
    )
    per_etf = pd.DataFrame(
        ticker_rows,
        columns=["ticker", "n_pairs", "pearson", "spearman", "status"],
    )
    cross_section["signal_date"] = pd.to_datetime(cross_section["signal_date"])
    cross_section["n_pairs"] = cross_section["n_pairs"].astype("int64")
    cross_section["status"] = cross_section["status"].astype("string")
    per_etf["ticker"] = per_etf["ticker"].astype("string")
    per_etf["n_pairs"] = per_etf["n_pairs"].astype("int64")
    per_etf["status"] = per_etf["status"].astype("string")
    return cross_section, per_etf
