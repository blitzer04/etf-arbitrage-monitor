"""As-of evaluation, complete session coverage, and signal staleness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from etf_monitor.evaluation.dependence import calculate_dependence
from etf_monitor.methodology import Methodology, load_methodology
from etf_monitor.prices.schema import ALLOWED_SOURCES, validate_canonical
from etf_monitor.signals.momentum import calculate_momentum
from etf_monitor.signals.panel import build_session_panel
from etf_monitor.signals.volatility import calculate_volatility
from etf_monitor.universe import Universe, load_universe
from etf_monitor.xnys import resolve_target_session, xnys_sessions


class EvaluationError(ValueError):
    """Indicate an invalid as-of evaluation request or input frame."""


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Hold immutable evaluation metadata and its in-memory artifact frames."""

    evaluation_instant_utc: datetime
    target_session: pd.Timestamp
    requested_interval: tuple[date, date]
    data_classification: str
    input_prices: pd.DataFrame
    coverage: pd.DataFrame
    momentum: pd.DataFrame
    volatility: pd.DataFrame
    dependence_cross_section: pd.DataFrame
    dependence_per_etf: pd.DataFrame
    universe: Universe
    methodology: Methodology


def _assert_methodology(methodology: Methodology) -> None:
    assert methodology.momentum_horizon_sessions == 252
    assert methodology.volatility_return_count == 21
    assert methodology.reference_window_sessions == 756
    assert methodology.minimum_reference_observations == 252
    assert methodology.volatility_ddof == 1
    assert methodology.annualization_sessions == 252


def _last_eligible_position(signal: pd.DataFrame, value_column: str) -> int | None:
    eligible = signal[value_column].notna().to_numpy()
    positions = np.flatnonzero(eligible)
    return None if not len(positions) else int(positions[-1])


def _coverage_row(
    ticker: str,
    ticker_prices: pd.DataFrame,
    panel: pd.DataFrame,
    momentum: pd.DataFrame,
    volatility: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    rows_after_target: int,
) -> dict[str, object]:
    row_present = panel["row_present"].to_numpy(dtype=np.bool_)
    adjusted_close = panel["adjusted_close"].to_numpy(dtype=np.float64)
    adjusted_present = np.isfinite(adjusted_close) & (adjusted_close > 0.0)
    missing_rows = [
        session.date()
        for session, present in zip(sessions, row_present, strict=True)
        if not present
    ]
    missing_adjusted = [
        session.date()
        for session, present, price in zip(
            sessions, row_present, adjusted_present, strict=True
        )
        if present and not price
    ]
    eligible_adjusted_positions = np.flatnonzero(row_present & adjusted_present)
    last_price_position = (
        None
        if not len(eligible_adjusted_positions)
        else int(eligible_adjusted_positions[-1])
    )
    target_position = len(sessions) - 1
    target_row_present = bool(row_present[target_position])
    target_adjusted_present = bool(adjusted_present[target_position])

    target_momentum = momentum.iloc[-1]
    target_volatility = volatility.iloc[-1]
    last_momentum_raw = _last_eligible_position(momentum, "raw_log_return")
    last_momentum_normalized = _last_eligible_position(momentum, "percentile")
    last_volatility_raw = _last_eligible_position(volatility, "raw_annualized_vol")
    last_volatility_normalized = _last_eligible_position(volatility, "percentile")

    def staleness(position: int | None) -> object:
        return pd.NA if position is None else target_position - position

    def row_date(position: int | None) -> pd.Timestamp | None:
        return None if position is None else sessions[position]

    return {
        "ticker": ticker,
        "expected_session_count": len(sessions),
        "present_row_count": int(np.count_nonzero(row_present)),
        "missing_row_count": len(missing_rows),
        "missing_row_dates": missing_rows,
        "missing_adjusted_close_count": len(missing_adjusted),
        "missing_adjusted_close_dates": missing_adjusted,
        "rows_after_target": rows_after_target,
        "first_row_date": (
            pd.NaT if ticker_prices.empty else pd.Timestamp(ticker_prices["date"].min())
        ),
        "last_row_date": (
            pd.NaT if ticker_prices.empty else pd.Timestamp(ticker_prices["date"].max())
        ),
        "last_eligible_adjusted_close_date": row_date(last_price_position),
        "price_staleness_sessions": staleness(last_price_position),
        "target_row_present": target_row_present,
        "target_adjusted_close_present": target_adjusted_present,
        "retrieved_at_min": (
            pd.NaT if ticker_prices.empty else ticker_prices["retrieved_at"].min()
        ),
        "retrieved_at_max": (
            pd.NaT if ticker_prices.empty else ticker_prices["retrieved_at"].max()
        ),
        "momentum_target_raw": target_momentum["raw_log_return"],
        "momentum_target_display": target_momentum["simple_return_pct"],
        "momentum_target_percentile": target_momentum["percentile"],
        "momentum_target_reference_count": int(target_momentum["reference_count"]),
        "momentum_target_status": target_momentum["status"],
        "momentum_target_normalization_status": target_momentum["normalization_status"],
        "momentum_last_raw_eligible_date": row_date(last_momentum_raw),
        "momentum_last_normalized_date": row_date(last_momentum_normalized),
        "momentum_raw_staleness_sessions": staleness(last_momentum_raw),
        "momentum_normalized_staleness_sessions": staleness(last_momentum_normalized),
        "volatility_target_raw": target_volatility["raw_annualized_vol"],
        "volatility_target_display": target_volatility["display_vol_pct"],
        "volatility_target_percentile": target_volatility["percentile"],
        "volatility_target_reference_count": int(target_volatility["reference_count"]),
        "volatility_target_status": target_volatility["status"],
        "volatility_target_normalization_status": target_volatility[
            "normalization_status"
        ],
        "volatility_last_raw_eligible_date": row_date(last_volatility_raw),
        "volatility_last_normalized_date": row_date(last_volatility_normalized),
        "volatility_raw_staleness_sessions": staleness(last_volatility_raw),
        "volatility_normalized_staleness_sessions": staleness(
            last_volatility_normalized
        ),
    }


def _coverage_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    string_columns = (
        "ticker",
        "momentum_target_status",
        "momentum_target_normalization_status",
        "volatility_target_status",
        "volatility_target_normalization_status",
    )
    for column in string_columns:
        frame[column] = frame[column].astype("string")
    for column in (
        "expected_session_count",
        "present_row_count",
        "missing_row_count",
        "missing_adjusted_close_count",
        "rows_after_target",
        "momentum_target_reference_count",
        "volatility_target_reference_count",
    ):
        frame[column] = frame[column].astype("int64")
    for column in (
        "price_staleness_sessions",
        "momentum_raw_staleness_sessions",
        "momentum_normalized_staleness_sessions",
        "volatility_raw_staleness_sessions",
        "volatility_normalized_staleness_sessions",
    ):
        frame[column] = frame[column].astype("Int64")
    for column in (
        "target_row_present",
        "target_adjusted_close_present",
    ):
        frame[column] = frame[column].astype("bool")
    for column in (
        "first_row_date",
        "last_row_date",
        "last_eligible_adjusted_close_date",
        "momentum_last_raw_eligible_date",
        "momentum_last_normalized_date",
        "volatility_last_raw_eligible_date",
        "volatility_last_normalized_date",
    ):
        frame[column] = pd.to_datetime(frame[column])
    for column in ("retrieved_at_min", "retrieved_at_max"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _combine_results(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "signal_date"], kind="mergesort")
        .reset_index(drop=True)
    )


def evaluate(
    canonical_prices: pd.DataFrame,
    evaluation_instant: datetime,
) -> EvaluationResult:
    """Evaluate both standalone signals against a single captured UTC instant.

    Rows after the resolved target are explicitly counted per ticker and
    excluded before panels and signal calculations are built. Older-than-
    methodology rows are rejected to avoid silently expanding the interval.

    Args:
        canonical_prices: Gate 2 canonical rows, possibly including later
            sessions that must not enter this evaluation.
        evaluation_instant: One timezone-aware instant captured by the caller.

    Returns:
        Full-history signal outputs, 24-row coverage, and dependence tables.

    Raises:
        EvaluationError: If timing, source classification, or interval is invalid.
        PriceValidationError: If canonical rows violate the Gate 2 contract.
    """
    if evaluation_instant.tzinfo is None or evaluation_instant.utcoffset() is None:
        raise EvaluationError("evaluation_instant must be timezone-aware.")
    instant_utc = evaluation_instant.astimezone(UTC)
    methodology = load_methodology()
    universe = load_universe()
    _assert_methodology(methodology)
    validate_canonical(canonical_prices)
    sources = tuple(canonical_prices["source"].dropna().unique())
    if len(sources) != 1:
        raise EvaluationError("Evaluation requires exactly one input source.")
    source = str(sources[0])
    if source not in ALLOWED_SOURCES:
        raise EvaluationError(f"Unsupported canonical source {source!r}.")
    classification = (
        "FICTIONAL" if source == "FICTIONAL_synthetic" else "OBSERVED_PROVIDER_DATA"
    )

    target = resolve_target_session(instant_utc)
    if target.date() < methodology.price_history_start:
        raise EvaluationError("Target session predates the methodology history start.")
    target_end = target.date() + timedelta(days=1)
    sessions = xnys_sessions(methodology.price_history_start, target.date())
    pre_history = canonical_prices["date"].lt(
        pd.Timestamp(methodology.price_history_start)
    )
    if pre_history.any():
        raise EvaluationError("Input prices contain rows before price_history_start.")

    after_target_mask = canonical_prices["date"].gt(target)
    after_target_counts = (
        canonical_prices.loc[after_target_mask]
        .groupby("ticker", sort=False)
        .size()
        .to_dict()
    )
    calculation_prices = canonical_prices.loc[~after_target_mask].copy()
    momentum_frames: list[pd.DataFrame] = []
    volatility_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    for etf in universe.etfs:
        ticker_prices = calculation_prices.loc[
            calculation_prices["ticker"].eq(etf.ticker)
        ].reset_index(drop=True)
        panel = build_session_panel(ticker_prices, etf.ticker, target, grid_start=None)
        momentum = calculate_momentum(
            ticker_prices,
            etf.ticker,
            target,
            grid_start=None,
        )
        volatility = calculate_volatility(
            ticker_prices,
            etf.ticker,
            target,
            grid_start=None,
        )
        momentum_frames.append(momentum)
        volatility_frames.append(volatility)
        coverage_rows.append(
            _coverage_row(
                etf.ticker,
                ticker_prices,
                panel,
                momentum,
                volatility,
                sessions,
                int(after_target_counts.get(etf.ticker, 0)),
            )
        )

    all_momentum = _combine_results(momentum_frames)
    all_volatility = _combine_results(volatility_frames)
    cross_section, per_etf = calculate_dependence(
        all_momentum, all_volatility, universe, sessions
    )
    return EvaluationResult(
        evaluation_instant_utc=instant_utc,
        target_session=target,
        requested_interval=(methodology.price_history_start, target_end),
        data_classification=classification,
        input_prices=canonical_prices.copy(deep=True).reset_index(drop=True),
        coverage=_coverage_frame(coverage_rows),
        momentum=all_momentum,
        volatility=all_volatility,
        dependence_cross_section=cross_section,
        dependence_per_etf=per_etf,
        universe=universe,
        methodology=methodology,
    )
