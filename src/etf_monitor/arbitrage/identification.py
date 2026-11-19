"""Causal-evidence checks for cost-band AP-response estimates."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from etf_monitor.xnys import xnys_sessions

BOOTSTRAP_DRAWS = 500
BOOTSTRAP_SEED = 20261118
EVENT_OFFSETS = tuple(range(-5, 6))
DOSE_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, np.inf)
RD_CUTOFFS = (0.5, 1.0, 1.5, 2.0)
US_BOND_CLASSES = {
    "us_high_yield_bond",
    "us_investment_grade_bond",
    "us_treasury_bills",
}


@dataclass(frozen=True, slots=True)
class MonthBootstrapPlan:
    """Hold one shared calendar-month resampling plan for a test."""

    months: tuple[str, ...]
    counts: np.ndarray


@dataclass(frozen=True, slots=True)
class ClusteredOLSResult:
    """Return OLS coefficients and month-clustered CR1 inference."""

    coefficients: np.ndarray
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    observations: int
    clusters: int


def daily_share_change(flows: pd.DataFrame) -> pd.Series:
    """Compute daily relative share changes in decimal units.

    The value at session ``t`` uses shares reported at ``t`` and the previous
    row on that fund's NAV-session grid. A missing/nonpositive share count or
    a ``split_session`` or ``missing_previous_session`` at ``t`` makes the
    value missing.

    Args:
        flows: Rows with ``ticker``, ``date``, ``shares_outstanding`` and
            ``status``.

    Returns:
        ``c_t = SO_t / SO_(t-1) - 1`` in decimal units, aligned to ``flows``.
    """
    _require_columns(flows, ("ticker", "date", "shares_outstanding", "status"), "flows")
    ordered = flows.sort_values(["ticker", "date"], kind="stable")
    shares = ordered["shares_outstanding"]
    previous = shares.groupby(ordered["ticker"], sort=False).shift(1)
    current_values = shares.to_numpy(dtype=object)
    previous_values = previous.to_numpy(dtype=object)
    status = ordered["status"].astype("string").fillna("").to_numpy(dtype=object)
    change = np.full(len(ordered), np.nan, dtype=np.float64)
    for index, (current, prior, row_status) in enumerate(
        zip(current_values, previous_values, status, strict=True)
    ):
        if (
            row_status in {"split_session", "missing_previous_session"}
            or not _is_positive_share_count(current)
            or not _is_positive_share_count(prior)
        ):
            continue
        change[index] = _relative_share_change(current, prior)
    return pd.Series(change, index=ordered.index, name="c_t").reindex(flows.index)


def find_onsets(band: pd.DataFrame) -> pd.DataFrame:
    """Find first premium/discount rows after a different valid band side.

    Args:
        band: Band rows with ``ticker``, ``date``, ``status``, ``band_bps`` and
            ``side``. Only ``ok`` rows with a non-null band enter the sequence.

    Returns:
        Onset rows with ``ticker``, ``date`` and ``side``. The first valid
        observation for a fund is not an onset because it has no prior side.
    """
    _require_columns(band, ("ticker", "date", "status", "band_bps", "side"), "band")
    valid = band.loc[
        (band["status"] == "ok") & band["band_bps"].notna() & band["side"].notna()
    ].sort_values(["ticker", "date"], kind="stable")
    previous = valid.groupby("ticker", sort=False)["side"].shift(1)
    onset = (
        valid["side"].isin(["premium", "discount"])
        & previous.notna()
        & valid["side"].ne(previous)
    )
    return valid.loc[onset, ["ticker", "date", "side"]].reset_index(drop=True)


def month_block_bootstrap(
    months: Sequence[str] | pd.Series,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> MonthBootstrapPlan:
    """Draw calendar-month clusters with replacement.

    Each draw samples as many distinct calendar months as occur in ``months``;
    all funds' rows in a selected month share that month's multiplicity.

    Args:
        months: ``YYYY-MM`` cluster labels for the test's full row set.
        draws: Number of bootstrap draws.
        seed: Random seed reused for all statistics within a test.

    Returns:
        A plan containing sorted unique month labels and a draw-by-month
        multiplicity matrix.
    """
    if draws < 1:
        raise ValueError("draws must be positive.")
    unique_months = tuple(
        sorted(pd.Series(months, dtype="string").dropna().astype(str).unique())
    )
    if not unique_months:
        raise ValueError("At least one month cluster is required.")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(unique_months), size=(draws, len(unique_months)), dtype=np.int64
    )
    counts = np.zeros((draws, len(unique_months)), dtype=np.int64)
    rows = np.arange(draws)[:, None]
    np.add.at(counts, (rows, sampled), 1)
    return MonthBootstrapPlan(unique_months, counts)


def two_way_demean(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    fund_column: str = "ticker",
    date_column: str = "date",
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> pd.DataFrame:
    """Absorb fund and date fixed effects by alternating projections.

    Args:
        frame: Complete observations with fund and date identifiers.
        columns: Numeric outcome/regressor columns to residualize.
        fund_column: Fund fixed-effect identifier.
        date_column: Date fixed-effect identifier.
        tolerance: Maximum elementwise change between projections.
        max_iterations: Maximum alternating-projection iterations.

    Returns:
        A data frame of demeaned columns, retaining the input index.
    """
    _require_columns(frame, (*columns, fund_column, date_column), "panel")
    if not columns:
        raise ValueError("At least one column is required.")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive.")
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Fixed-effect inputs must be finite.")
    fund_codes, fund_levels = pd.factorize(frame[fund_column], sort=True)
    date_codes, date_levels = pd.factorize(frame[date_column], sort=True)
    if (fund_codes < 0).any() or (date_codes < 0).any():
        raise ValueError("Fixed-effect identifiers cannot be missing.")
    for _ in range(max_iterations):
        previous = values.copy()
        values = _demean_codes(values, fund_codes, len(fund_levels))
        values = _demean_codes(values, date_codes, len(date_levels))
        if float(np.max(np.abs(values - previous))) < tolerance:
            return pd.DataFrame(values, index=frame.index, columns=list(columns))
    raise ValueError(
        f"Two-way fixed-effect demeaning did not converge in {max_iterations} "
        "iterations."
    )


def ols_month_clustered_cr1(
    outcome: Sequence[float] | np.ndarray,
    regressors: pd.DataFrame | np.ndarray,
    months: Sequence[str] | np.ndarray,
) -> ClusteredOLSResult:
    """Fit OLS and month-clustered CR1 standard errors.

    Args:
        outcome: Finite outcome values in their stated units.
        regressors: Finite design matrix, without absorbed fixed effects.
        months: Calendar-month cluster labels for each observation.

    Returns:
        Coefficients, CR1 standard errors, t-statistics, observation count and
        month-cluster count. The finite-sample factor is
        ``G/(G-1) * (N-1)/(N-K)``.
    """
    y = np.asarray(outcome, dtype=np.float64).reshape(-1)
    x = np.asarray(regressors, dtype=np.float64)
    clusters = np.asarray(months, dtype=object).reshape(-1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or len(y) != len(x) or len(clusters) != len(y):
        raise ValueError("Outcome, regressors and month labels must align.")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("OLS inputs must be finite.")
    observations, parameters = x.shape
    cluster_levels, cluster_codes = np.unique(clusters, return_inverse=True)
    cluster_count = len(cluster_levels)
    if observations <= parameters or cluster_count < 2:
        raise ValueError("CR1 requires N > K and at least two month clusters.")
    if np.linalg.matrix_rank(x) < parameters:
        raise ValueError("The OLS regressor matrix is rank deficient.")
    cross_product = x.T @ x
    bread = np.linalg.inv(cross_product)
    coefficients = bread @ (x.T @ y)
    residuals = y - x @ coefficients
    scores = np.zeros((cluster_count, parameters), dtype=np.float64)
    np.add.at(scores, cluster_codes, x * residuals[:, None])
    meat = scores.T @ scores
    correction = (
        cluster_count
        / (cluster_count - 1)
        * (observations - 1)
        / (observations - parameters)
    )
    covariance = correction * bread @ meat @ bread
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0,
    )
    return ClusteredOLSResult(
        coefficients,
        standard_errors,
        t_statistics,
        observations,
        cluster_count,
    )


def rolling_past_beta(
    panel: pd.DataFrame,
    window: int = 252,
    min_observations: int = 126,
) -> pd.DataFrame:
    """Estimate past-only premium sensitivity and market-mechanical residuals.

    The rolling OLS uses premium and SPY NAV returns from the previous
    ``window`` valid band sessions, excluding the current row. Premium and
    market returns are in basis points; ``mech_bps = alpha + beta * market_bps``.

    Args:
        panel: Rows with ``ticker``, ``date``, ``premium_discount_bps`` and
            ``market_return_bps``.
        window: Number of prior valid band sessions.
        min_observations: Minimum complete premium/market pairs in the window.

    Returns:
        ``ticker``, ``date``, ``alpha_bps``, ``beta``, ``mech_bps`` and
        ``resid_bps`` for each input row. Estimates are missing until enough
        prior observations and a current market return are available.
    """
    _require_columns(
        panel,
        ("ticker", "date", "premium_discount_bps", "market_return_bps"),
        "rolling panel",
    )
    if window < 1 or min_observations < 2 or min_observations > window:
        raise ValueError("Require window >= min_observations >= 2.")
    ordered = panel.sort_values(["ticker", "date"], kind="stable")
    records: list[dict[str, object]] = []
    for ticker, fund in ordered.groupby("ticker", sort=False, observed=True):
        premium = fund["premium_discount_bps"].to_numpy(dtype=np.float64)
        market = fund["market_return_bps"].to_numpy(dtype=np.float64)
        dates = fund["date"].to_numpy()
        for position in range(len(fund)):
            start = max(0, position - window)
            history_premium = premium[start:position]
            history_market = market[start:position]
            complete = np.isfinite(history_premium) & np.isfinite(history_market)
            alpha = beta = mech = residual = np.nan
            if int(complete.sum()) >= min_observations and np.isfinite(
                market[position]
            ):
                design = np.column_stack(
                    (np.ones(int(complete.sum())), history_market[complete])
                )
                if np.linalg.matrix_rank(design) == 2:
                    alpha_beta = np.linalg.lstsq(
                        design, history_premium[complete], rcond=None
                    )[0]
                    alpha = float(alpha_beta[0])
                    beta = float(alpha_beta[1])
                    mech = alpha + beta * market[position]
                    residual = premium[position] - mech
            records.append(
                {
                    "ticker": ticker,
                    "date": dates[position],
                    "alpha_bps": alpha,
                    "beta": beta,
                    "mech_bps": mech,
                    "resid_bps": residual,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "ticker",
            "date",
            "alpha_bps",
            "beta",
            "mech_bps",
            "resid_bps",
        ],
    )


def local_linear_rd_jump(
    frame: pd.DataFrame,
    *,
    cutoff: float,
    outcome_column: str = "outcome",
    running_column: str = "s",
    fund_column: str = "ticker",
    window: float = 0.5,
) -> dict[str, float | int]:
    """Estimate a local-linear discontinuity with fund fixed effects.

    Args:
        frame: Rows with outcome, nonnegative running variable and fund ID.
        cutoff: Cost-band multiple at the discontinuity.
        outcome_column: Outcome column.
        running_column: Running-variable column ``s``.
        fund_column: Fund fixed-effect identifier.
        window: Uniform-kernel half-width around ``cutoff``.

    Returns:
        The jump ``tau`` and left/right observation counts. The outcome units
        are inherited from ``outcome_column``; the running variable is in
        multiples of the cost band.
    """
    _require_columns(frame, (outcome_column, running_column, fund_column), "RD frame")
    result = _local_linear_rd_fit(
        frame,
        cutoff=cutoff,
        outcome_column=outcome_column,
        running_column=running_column,
        fund_column=fund_column,
        window=window,
        weights=np.ones(len(frame), dtype=np.float64),
        raise_on_rank=True,
    )
    return {
        "tau": result[0],
        "n_left": result[1],
        "n_right": result[2],
    }


def event_study(
    band_rows: pd.DataFrame,
    flow_rows: pd.DataFrame,
    group_memberships: Mapping[str, Collection[str]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate creation/redemption changes around cost-band onsets.

    Offsets ``k <= 0`` are the pre-period because SSGA reports creations one
    session late; ``k = 1`` is the first session that can reflect a response
    to the onset-day close. Share changes and baseline rates are proportions.

    Args:
        band_rows: Valid band rows with ``ticker``, ``date``, ``side``,
            ``band_bps`` and ``c_t``.
        flow_rows: Rows with ``ticker``, ``date`` and ``c_t``. Missing XNYS
            sessions within each fund's date range are inserted before offsets
            are applied.
        group_memberships: Group names mapped to their member tickers.
        draws: Number of calendar-month bootstrap draws.
        seed: Seed shared by all event-study statistics.

    Returns:
        Event-study rows ``(group, side, k, onsets, uplift, lo, hi)`` and
        pre/post rows containing mean pre-period uplift, mean post-period
        uplift, their difference, and its bootstrap interval.
    """
    _require_columns(
        band_rows, ("ticker", "date", "status", "band_bps", "side", "c_t"), "band"
    )
    _require_columns(flow_rows, ("ticker", "date", "c_t"), "flows")
    flow_rows = _expand_to_xnys_sessions(flow_rows)
    plan = month_block_bootstrap(_month_labels(band_rows["date"]), draws, seed)
    onsets = find_onsets(band_rows)
    onsets["month"] = _month_labels(onsets["date"])
    flow_lookup: dict[tuple[str, pd.Timestamp], tuple[int, float]] = {}
    values_by_ticker: dict[str, np.ndarray] = {}
    for ticker, fund in flow_rows.sort_values(
        ["ticker", "grid_position"], kind="stable"
    ).groupby("ticker", sort=False, observed=True):
        values_by_ticker[str(ticker)] = fund["c_t"].to_numpy(dtype=np.float64)
        fund_tickers = fund["ticker"].astype(str).to_numpy(dtype=str)
        fund_dates = fund["date"].to_numpy(dtype="datetime64[ns]")
        fund_positions = fund["grid_position"].to_numpy(dtype=np.int64)
        fund_changes = fund["c_t"].to_numpy(dtype=np.float64, na_value=np.nan)
        for row_ticker, row_date, row_position, row_change in zip(
            fund_tickers,
            fund_dates,
            fund_positions,
            fund_changes,
            strict=True,
        ):
            flow_lookup[(row_ticker, pd.Timestamp(row_date))] = (
                int(row_position),
                float(row_change),
            )

    event_records: list[dict[str, object]] = []
    pre_post_records: list[dict[str, object]] = []
    for group_name, members in sorted(group_memberships.items()):
        group_band = band_rows.loc[band_rows["ticker"].isin(members)]
        for side in ("premium", "discount"):
            onset_rows = onsets.loc[
                onsets["ticker"].isin(members) & onsets["side"].eq(side)
            ].reset_index(drop=True)
            baseline_rows = group_band.loc[group_band["side"].eq("inside")]
            baseline_changes = baseline_rows["c_t"].to_numpy(dtype=np.float64)
            baseline_valid = np.isfinite(baseline_changes)
            baseline_values = (
                (baseline_changes[baseline_valid] > 0)
                if side == "premium"
                else (baseline_changes[baseline_valid] < 0)
            ).astype(np.float64)
            baseline_months = _month_labels(baseline_rows.loc[baseline_valid, "date"])
            baseline_point = _finite_mean(baseline_values)
            baseline_boot = _bootstrap_mean(
                baseline_values,
                baseline_months,
                plan,
            )
            uplift_draws: dict[int, np.ndarray] = {}
            point_uplifts: dict[int, float] = {}
            for offset in EVENT_OFFSETS:
                observed: list[float] = []
                observed_months: list[str] = []
                onset_tickers = onset_rows["ticker"].astype(str).to_numpy(dtype=str)
                onset_dates = onset_rows["date"].to_numpy(dtype="datetime64[ns]")
                onset_months = onset_rows["month"].astype(str).to_numpy(dtype=str)
                for ticker, onset_date, onset_month in zip(
                    onset_tickers,
                    onset_dates,
                    onset_months,
                    strict=True,
                ):
                    key = (ticker, pd.Timestamp(onset_date))
                    if key not in flow_lookup:
                        raise ValueError(f"Onset row is missing from flows: {key}.")
                    position, _ = flow_lookup[key]
                    shifted = position + offset
                    ticker_values = values_by_ticker[ticker]
                    if shifted < 0 or shifted >= len(ticker_values):
                        continue
                    change = ticker_values[shifted]
                    if not np.isfinite(change):
                        continue
                    observed.append(
                        float(change > 0) if side == "premium" else float(change < 0)
                    )
                    observed_months.append(onset_month)
                event_values = np.asarray(observed, dtype=np.float64)
                event_boot = _bootstrap_mean(
                    event_values,
                    observed_months,
                    plan,
                )
                event_point = _finite_mean(event_values)
                uplift = event_point - baseline_point
                draws_for_uplift = event_boot - baseline_boot
                lo, hi = _percentile_interval(draws_for_uplift)
                point_uplifts[offset] = uplift
                uplift_draws[offset] = draws_for_uplift
                event_records.append(
                    {
                        "group": group_name,
                        "side": side,
                        "k": offset,
                        "onsets": len(onset_rows),
                        "uplift": uplift,
                        "lo": lo,
                        "hi": hi,
                    }
                )
            pre_offsets = tuple(range(-4, 1))
            post_offsets = tuple(range(1, 6))
            pre_values = np.asarray(
                [point_uplifts[offset] for offset in pre_offsets], dtype=np.float64
            )
            post_values = np.asarray(
                [point_uplifts[offset] for offset in post_offsets], dtype=np.float64
            )
            pre_draws = np.vstack([uplift_draws[offset] for offset in pre_offsets])
            post_draws = np.vstack([uplift_draws[offset] for offset in post_offsets])
            pre_mean = _finite_mean(pre_values)
            post_mean = _finite_mean(post_values)
            difference_draws = _row_nanmean(post_draws) - _row_nanmean(pre_draws)
            lo, hi = _percentile_interval(difference_draws)
            pre_post_records.append(
                {
                    "group": group_name,
                    "side": side,
                    "onsets": len(onset_rows),
                    "pre_mean": pre_mean,
                    "post_mean": post_mean,
                    "diff": post_mean - pre_mean,
                    "lo": lo,
                    "hi": hi,
                }
            )
    return (
        pd.DataFrame.from_records(
            event_records,
            columns=["group", "side", "k", "onsets", "uplift", "lo", "hi"],
        ),
        pd.DataFrame.from_records(
            pre_post_records,
            columns=[
                "group",
                "side",
                "onsets",
                "pre_mean",
                "post_mean",
                "diff",
                "lo",
                "hi",
            ],
        ),
    )


def dose_response_bins(
    panel: pd.DataFrame,
    group_memberships: Mapping[str, Collection[str]],
    *,
    bootstrap_plan: MonthBootstrapPlan | None = None,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Estimate creation/redemption probabilities by cost-band multiple.

    Premium rows use ``s = r_t`` and ``Y_create``; discount rows use
    ``s = -r_t`` and ``Y_redeem``. The bins are left-closed in cost-band
    multiples, and confidence limits use calendar-month block resampling.

    Args:
        panel: Valid band rows with ``ticker``, ``date``, ``r_t``,
            ``Y_create`` and ``Y_redeem``.
        group_memberships: Group names mapped to their member tickers.
        bootstrap_plan: Optional month resampling plan shared with RD.
        draws: Number of draws when a plan is not supplied.
        seed: Seed used when a plan is not supplied.

    Returns:
        Group/side/bin counts, mean outcome and 95% month-bootstrap intervals.
    """
    _require_columns(
        panel,
        ("ticker", "date", "r_t", "Y_create", "Y_redeem"),
        "dose-response panel",
    )
    plan = bootstrap_plan or month_block_bootstrap(
        _month_labels(panel["date"]), draws, seed
    )
    records: list[dict[str, object]] = []
    for group_name, members in sorted(group_memberships.items()):
        group_rows = panel.loc[panel["ticker"].isin(members)]
        for side in ("premium", "discount"):
            side_rows = _side_sample(group_rows, side)
            running = side_rows["s"].to_numpy(dtype=np.float64)
            outcome = side_rows["outcome"].to_numpy(dtype=np.float64)
            month_values = _month_labels(side_rows["date"])
            valid = np.isfinite(running) & np.isfinite(outcome) & (running >= 0)
            running = running[valid]
            outcome = outcome[valid]
            month_values = month_values.loc[valid].reset_index(drop=True)
            bin_indices = (
                np.searchsorted(np.asarray(DOSE_EDGES), running, side="right") - 1
            )
            for bin_index, (lower, upper) in enumerate(
                zip(DOSE_EDGES[:-1], DOSE_EDGES[1:], strict=True)
            ):
                selected = bin_indices == bin_index
                values = outcome[selected]
                selected_months = month_values.loc[selected].reset_index(drop=True)
                draws_for_mean = _bootstrap_mean(values, selected_months, plan)
                lo, hi = _percentile_interval(draws_for_mean)
                records.append(
                    {
                        "group": group_name,
                        "side": side,
                        "bin": _bin_label(lower, upper),
                        "n": len(values),
                        "mean_outcome": _finite_mean(values),
                        "lo": lo,
                        "hi": hi,
                    }
                )
    return pd.DataFrame.from_records(
        records,
        columns=["group", "side", "bin", "n", "mean_outcome", "lo", "hi"],
    )


def _side_sample(panel: pd.DataFrame, side: str) -> pd.DataFrame:
    if side == "premium":
        selected = panel.loc[panel["r_t"] >= 0].copy()
        selected["s"] = selected["r_t"]
        selected["outcome"] = selected["Y_create"]
        return selected
    selected = panel.loc[panel["r_t"] <= 0].copy()
    selected["s"] = -selected["r_t"]
    selected["outcome"] = selected["Y_redeem"]
    return selected


def _rd_jump_table(
    panel: pd.DataFrame,
    group_memberships: Mapping[str, Collection[str]],
    plan: MonthBootstrapPlan,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for group_name, members in sorted(group_memberships.items()):
        group_rows = panel.loc[panel["ticker"].isin(members)]
        for side in ("premium", "discount"):
            sample = _side_sample(group_rows, side)
            sample = sample.loc[
                np.isfinite(sample["s"].to_numpy(dtype=np.float64))
                & np.isfinite(sample["outcome"].to_numpy(dtype=np.float64))
            ].reset_index(drop=True)
            months = _month_labels(sample["date"])
            month_codes = _month_codes(months, plan)
            for cutoff in RD_CUTOFFS:
                point_tau, n_left, n_right = _local_linear_rd_fit(
                    sample,
                    cutoff=cutoff,
                    outcome_column="outcome",
                    running_column="s",
                    fund_column="ticker",
                    window=0.5,
                    weights=np.ones(len(sample), dtype=np.float64),
                    raise_on_rank=False,
                )
                boot_taus = np.full(len(plan.counts), np.nan, dtype=np.float64)
                for draw_index, month_counts in enumerate(plan.counts):
                    row_weights = month_counts[month_codes]
                    draw_tau, _, _ = _local_linear_rd_fit(
                        sample,
                        cutoff=cutoff,
                        outcome_column="outcome",
                        running_column="s",
                        fund_column="ticker",
                        window=0.5,
                        weights=row_weights.astype(np.float64),
                        raise_on_rank=False,
                    )
                    boot_taus[draw_index] = draw_tau
                valid_draws = int(np.isfinite(boot_taus).sum())
                insufficient_support = (
                    n_left < 20 or n_right < 20 or not np.isfinite(point_tau)
                )
                if insufficient_support:
                    status = "insufficient_support"
                    lo = hi = np.nan
                    point_tau = np.nan
                elif valid_draws < 400:
                    status = "unstable_bootstrap"
                    lo = hi = np.nan
                else:
                    status = "ok"
                    lo, hi = _percentile_interval(boot_taus)
                records.append(
                    {
                        "group": group_name,
                        "side": side,
                        "cutoff": cutoff,
                        "tau": point_tau,
                        "n_left": n_left,
                        "n_right": n_right,
                        "lo": lo,
                        "hi": hi,
                        "valid_draws": valid_draws,
                        "status": status,
                    }
                )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "group",
            "side",
            "cutoff",
            "tau",
            "n_left",
            "n_right",
            "lo",
            "hi",
            "valid_draws",
            "status",
        ],
    )


def stale_placebo_regression(
    panel: pd.DataFrame,
    stale_tickers: Collection[str],
) -> tuple[pd.DataFrame, int]:
    """Regress stale-fund outcomes on market-mechanical and residual premiums.

    Outcomes are five-session share changes in basis points and the creation
    indicator. Both regressions absorb fund and date effects and use month-
    clustered CR1 standard errors.

    Args:
        panel: Valid band rows with rolling ``mech_bps``/``resid_bps``,
            ``F_t_bps``, ``Y_create`` and calendar-month labels.
        stale_tickers: Funds marked ``stale_nav`` in the cost configuration.

    Returns:
        A coefficient table and the number of complete rows used for the
        forward-share-change regression. The mechanical-fit R-squared is the
        squared correlation of premium and mechanical premium. Coefficient
        rows are marked ``insufficient_support`` with missing estimates when
        they have fewer than three funds, fewer than 24 months, or a
        rank-deficient within-fund/date design.
    """
    required = (
        "ticker",
        "date",
        "month",
        "premium_discount_bps",
        "mech_bps",
        "resid_bps",
        "beta",
        "F_t_bps",
        "Y_create",
    )
    _require_columns(panel, required, "stale-placebo panel")
    stale = panel.loc[panel["ticker"].isin(stale_tickers)].copy()
    fit_rows = stale.loc[
        np.isfinite(stale["premium_discount_bps"].to_numpy(dtype=np.float64))
        & np.isfinite(stale["mech_bps"].to_numpy(dtype=np.float64))
    ]
    premium_values = fit_rows["premium_discount_bps"].to_numpy(dtype=np.float64)
    mechanical_values = fit_rows["mech_bps"].to_numpy(dtype=np.float64)
    if (
        len(fit_rows) >= 2
        and np.std(premium_values) > 0
        and np.std(mechanical_values) > 0
    ):
        correlation = float(np.corrcoef(premium_values, mechanical_values)[0, 1])
        mech_r_squared = correlation**2
    else:
        mech_r_squared = np.nan
    mean_beta = float(fit_rows["beta"].mean(skipna=True))
    records: list[dict[str, object]] = []
    share_sample_size = 0
    for outcome_column, outcome_name in (
        ("F_t_bps", "forward_change_bps"),
        ("Y_create", "Y_create"),
    ):
        complete = stale.dropna(
            subset=[outcome_column, "mech_bps", "resid_bps", "month"]
        ).copy()
        if outcome_column == "F_t_bps":
            share_sample_size = len(complete)
        funds = int(complete["ticker"].nunique())
        months = int(complete["month"].nunique())
        coefficients = np.full(2, np.nan, dtype=np.float64)
        standard_errors = np.full(2, np.nan, dtype=np.float64)
        t_statistics = np.full(2, np.nan, dtype=np.float64)
        status = "insufficient_support"
        if funds >= 3 and months >= 24:
            demeaned = two_way_demean(
                complete,
                [outcome_column, "mech_bps", "resid_bps"],
            )
            regressors = demeaned[["mech_bps", "resid_bps"]].to_numpy(dtype=np.float64)
            if np.linalg.matrix_rank(regressors) == regressors.shape[1]:
                result = ols_month_clustered_cr1(
                    demeaned[outcome_column].to_numpy(dtype=np.float64),
                    regressors,
                    complete["month"].astype(str).to_numpy(),
                )
                coefficients = result.coefficients
                standard_errors = result.standard_errors
                t_statistics = result.t_statistics
                status = "ok"
        for index, term in enumerate(("mech_bps", "resid_bps")):
            records.append(
                {
                    "outcome": outcome_name,
                    "term": term,
                    "coefficient": coefficients[index],
                    "standard_error": standard_errors[index],
                    "t_statistic": t_statistics[index],
                    "n": len(complete),
                    "funds": funds,
                    "months": months,
                    "mech_r_squared": mech_r_squared,
                    "mean_beta": mean_beta,
                    "status": status,
                }
            )
    return (
        pd.DataFrame.from_records(
            records,
            columns=[
                "outcome",
                "term",
                "coefficient",
                "standard_error",
                "t_statistic",
                "n",
                "funds",
                "months",
                "mech_r_squared",
                "mean_beta",
                "status",
            ],
        ),
        share_sample_size,
    )


def panel_regression(
    panel: pd.DataFrame,
    group_memberships: Mapping[str, Collection[str]],
) -> tuple[pd.DataFrame, int]:
    """Estimate two-way fixed-effect panel regressions with CR1 inference.

    Outcomes and regressors are in basis points except ``F_t``'s underlying
    share change, which is converted to basis points. Date effects absorb
    market returns and volatility common to a date.

    Args:
        panel: Valid band rows with share-change, premium, NAV-return and
            five-session outcome columns.
        group_memberships: Group names mapped to their member tickers.

    Returns:
        One row per group/regressor with the coefficient and standard error
        scaled to a 10-basis-point regressor move, t-statistic, sample size,
        number of funds and number of month clusters; plus the total number of
        complete group-observations across eligible regressions. Eligible
        groups below the three-fund/24-month thresholds or with a
        rank-deficient design remain in the table with missing estimates and
        ``insufficient_support`` status.
    """
    required = (
        "ticker",
        "date",
        "month",
        "premium_discount_bps",
        "band_bps",
        "c_t_bps",
        "prior_5_c_sum_bps",
        "nav_return_5_bps",
        "F_t_bps",
    )
    _require_columns(panel, required, "panel regression")
    records: list[dict[str, object]] = []
    observations_total = 0
    regressors = (
        "excess_prem_bps",
        "excess_disc_bps",
        "inside_prem_bps",
        "c_t_bps",
        "prior_5_c_sum_bps",
        "nav_return_5_bps",
    )
    prepared = panel.copy()
    premium = prepared["premium_discount_bps"].to_numpy(dtype=np.float64)
    band = prepared["band_bps"].to_numpy(dtype=np.float64)
    prepared["excess_prem_bps"] = np.maximum(premium - band, 0.0)
    prepared["excess_disc_bps"] = np.maximum(-premium - band, 0.0)
    prepared["inside_prem_bps"] = np.clip(premium, -band, band)
    for group_name, members in sorted(group_memberships.items()):
        if (
            group_name not in {"us_bond", "intl_stale", "us_equity"}
            and len(members) < 2
        ):
            continue
        group_rows = prepared.loc[prepared["ticker"].isin(members)].copy()
        complete = group_rows.dropna(subset=["F_t_bps", "month", *regressors]).copy()
        funds = int(complete["ticker"].nunique())
        months = int(complete["month"].nunique())
        observations_total += len(complete)
        coefficients = np.full(len(regressors), np.nan, dtype=np.float64)
        standard_errors = np.full(len(regressors), np.nan, dtype=np.float64)
        t_statistics = np.full(len(regressors), np.nan, dtype=np.float64)
        status = "insufficient_support"
        if funds >= 3 and months >= 24:
            demeaned = two_way_demean(complete, ["F_t_bps", *regressors])
            regressors_matrix = demeaned[list(regressors)].to_numpy(dtype=np.float64)
            if np.linalg.matrix_rank(regressors_matrix) == len(regressors):
                result = ols_month_clustered_cr1(
                    demeaned["F_t_bps"].to_numpy(dtype=np.float64),
                    regressors_matrix,
                    complete["month"].astype(str).to_numpy(),
                )
                coefficients = result.coefficients
                standard_errors = result.standard_errors
                t_statistics = result.t_statistics
                status = "ok"
        for index, term in enumerate(regressors):
            records.append(
                {
                    "group": group_name,
                    "term": term,
                    "coefficient_per_10_bps": coefficients[index] * 10.0,
                    "standard_error_per_10_bps": standard_errors[index] * 10.0,
                    "t_statistic": t_statistics[index],
                    "n": len(complete),
                    "funds": funds,
                    "months": months,
                    "status": status,
                }
            )
    return (
        pd.DataFrame.from_records(
            records,
            columns=[
                "group",
                "term",
                "coefficient_per_10_bps",
                "standard_error_per_10_bps",
                "t_statistic",
                "n",
                "funds",
                "months",
                "status",
            ],
        ),
        observations_total,
    )


def run_identification(
    arbitrage_band: pd.DataFrame,
    core_band: pd.DataFrame,
    arbitrage_flows: pd.DataFrame,
    core_flows: pd.DataFrame,
    asset_classes: Mapping[str, str],
    stale_tickers: Collection[str],
    core_tickers: Collection[str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Run the four identification checks on validated in-memory inputs.

    Args:
        arbitrage_band: V2 arbitrage-universe band rows.
        core_band: V2 core-universe band rows, including the SPY NAV proxy.
        arbitrage_flows: Flow-proxy rows paired with ``arbitrage_band``.
        core_flows: Flow-proxy rows paired with ``core_band``.
        asset_classes: ``ticker -> asset_class`` mapping from ``load_fund_costs``.
        stale_tickers: Tickers marked ``stale_nav`` in the cost configuration.
        core_tickers: Tickers in the configured core universe.
        draws: Number of calendar-month bootstrap draws.
        seed: Bootstrap seed shared within each test.

    Returns:
        A mapping of output CSV names to data frames and summary row counts.
    """
    if draws < 1:
        raise ValueError("draws must be positive.")
    _require_columns(core_band, ("ticker", "date", "nav"), "core band")
    if "SPY" not in set(core_band["ticker"]):
        raise ValueError("SPY is missing from the core band.")
    spy_rows = core_band.loc[
        core_band["ticker"].eq("SPY"), ["date", "nav"]
    ].sort_values("date", kind="stable")
    if spy_rows.empty or spy_rows["date"].duplicated().any():
        raise ValueError("Core band must contain unique SPY NAV rows.")
    spy_nav = spy_rows["nav"].astype("Float64")
    previous_spy_nav = spy_nav.shift(1)
    spy_values = spy_nav.to_numpy(dtype=np.float64, na_value=np.nan)
    previous_spy_values = previous_spy_nav.to_numpy(dtype=np.float64, na_value=np.nan)
    market_return = np.full(len(spy_rows), np.nan, dtype=np.float64)
    valid_spy = (
        np.isfinite(spy_values)
        & np.isfinite(previous_spy_values)
        & (spy_values > 0)
        & (previous_spy_values > 0)
    )
    market_return[valid_spy] = (
        spy_values[valid_spy] / previous_spy_values[valid_spy] - 1.0
    ) * 10_000.0
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(spy_rows["date"]).to_numpy(),
            "market_return_bps": market_return,
        }
    )
    arbitrage_panel, arbitrage_grid = _prepare_universe(
        arbitrage_band, arbitrage_flows, market, "arbitrage"
    )
    core_panel, core_grid = _prepare_universe(core_band, core_flows, market, "core")
    panel = pd.concat([arbitrage_panel, core_panel], ignore_index=True).sort_values(
        ["ticker", "date"], kind="stable"
    )
    panel = panel.reset_index(drop=True)
    flows = pd.concat([arbitrage_grid, core_grid], ignore_index=True).sort_values(
        ["ticker", "date"], kind="stable"
    )
    flows = flows.reset_index(drop=True)
    if flows.duplicated(["ticker", "date"]).any():
        raise ValueError("Input flow grids overlap on ticker/date keys.")
    known_tickers = set(asset_classes)
    unknown = sorted(set(panel["ticker"]) - known_tickers)
    if unknown:
        raise ValueError(f"Cost configuration has no asset class for {unknown}.")
    memberships: dict[str, set[str]] = {
        asset_class: {
            ticker
            for ticker, configured in asset_classes.items()
            if configured == asset_class
        }
        for asset_class in sorted(set(asset_classes.values()))
    }
    memberships["us_bond"] = {
        ticker
        for ticker, asset_class in asset_classes.items()
        if asset_class in US_BOND_CLASSES
    }
    memberships["intl_stale"] = set(stale_tickers)
    memberships["us_equity"] = set(core_tickers)
    rolling = rolling_past_beta(
        panel[["ticker", "date", "premium_discount_bps", "market_return_bps"]]
    )
    panel = panel.merge(
        rolling,
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )
    panel["F_t_bps"] = panel["F_t"] * 10_000.0
    panel["c_t_bps"] = panel["c_t"] * 10_000.0
    event_rows, pre_post = event_study(
        panel,
        flows,
        memberships,
        draws=draws,
        seed=seed,
    )
    dose_plan = month_block_bootstrap(_month_labels(panel["date"]), draws, seed)
    dose_response = dose_response_bins(
        panel,
        memberships,
        bootstrap_plan=dose_plan,
    )
    rd_jumps = _rd_jump_table(panel, memberships, dose_plan)
    stale_placebo, stale_rows = stale_placebo_regression(panel, stale_tickers)
    panel_table, panel_rows = panel_regression(panel, memberships)
    outputs = {
        "event_study": event_rows,
        "pre_post": pre_post,
        "dose_response": dose_response,
        "rd_jumps": rd_jumps,
        "stale_placebo": stale_placebo,
        "panel": panel_table,
    }
    row_counts: dict[str, object] = {
        "arbitrage_band_sample": len(arbitrage_panel),
        "core_band_sample": len(core_panel),
        "event_study": {
            "onset_rows": len(find_onsets(panel)),
            "output_rows": len(event_rows),
        },
        "dose_response": {
            "eligible_rows": int(
                (panel["Y_create"].notna() | panel["Y_redeem"].notna()).sum()
            ),
            "output_rows": len(dose_response),
        },
        "rd_jumps": {
            "eligible_rows": int(
                (panel["Y_create"].notna() | panel["Y_redeem"].notna()).sum()
            ),
            "output_rows": len(rd_jumps),
        },
        "stale_placebo": {
            "regression_rows": stale_rows,
            "output_rows": len(stale_placebo),
        },
        "panel": {
            "regression_observations": panel_rows,
            "output_rows": len(panel_table),
        },
    }
    return outputs, row_counts


def _prepare_universe(
    band: pd.DataFrame,
    flows: pd.DataFrame,
    market: pd.DataFrame,
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        band,
        (
            "ticker",
            "date",
            "premium_discount_bps",
            "nav",
            "band_bps",
            "side",
            "stale_nav",
            "status",
        ),
        f"{name} band",
    )
    flow_grid = _prepare_flows(flows)
    valid_band = band.copy()
    valid_band["date"] = pd.to_datetime(valid_band["date"])
    valid_band = valid_band.loc[
        (valid_band["status"] == "ok")
        & (valid_band["date"] >= pd.Timestamp("2018-01-01"))
        & valid_band["band_bps"].notna()
    ].copy()
    valid_band["band_bps"] = valid_band["band_bps"].astype("Float64")
    valid_band["premium_discount_bps"] = valid_band["premium_discount_bps"].astype(
        "Float64"
    )
    band_values = valid_band["band_bps"].to_numpy(dtype=np.float64, na_value=np.nan)
    premium_values = valid_band["premium_discount_bps"].to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if not np.isfinite(band_values).all() or (band_values <= 0).any():
        raise ValueError(f"{name} band contains nonpositive band_bps.")
    if not np.isfinite(premium_values).all():
        raise ValueError(f"{name} ok band rows have missing premium values.")
    if valid_band.duplicated(["ticker", "date"]).any():
        raise ValueError(f"{name} band contains duplicate ticker/date rows.")
    metric_columns = [
        "ticker",
        "date",
        "c_t",
        "F_t",
        "Y_create",
        "Y_redeem",
        "grid_position",
        "prior_5_c_sum_bps",
        "nav_return_5_bps",
    ]
    merged = valid_band.merge(
        flow_grid[metric_columns],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        missing = merged.loc[merged["_merge"] != "both", ["ticker", "date"]].to_dict(
            orient="records"
        )
        raise ValueError(
            f"{name} band rows are missing from the flow grid: {missing[:5]}."
        )
    merged = merged.drop(columns="_merge")
    premium_values = merged["premium_discount_bps"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    band_values = merged["band_bps"].to_numpy(dtype=np.float64, na_value=np.nan)
    merged["r_t"] = premium_values / band_values
    merged["month"] = _month_labels(merged["date"])
    merged = merged.merge(
        market,
        on="date",
        how="left",
        validate="many_to_one",
    )
    return merged.sort_values(["ticker", "date"], kind="stable"), flow_grid


def _prepare_flows(flows: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        flows,
        ("ticker", "date", "shares_outstanding", "nav", "status"),
        "flow proxy",
    )
    ordered = flows.copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    if ordered.duplicated(["ticker", "date"]).any():
        raise ValueError("Flow proxy contains duplicate ticker/date rows.")
    ordered = ordered.sort_values(["ticker", "date"], kind="stable").reset_index(
        drop=True
    )
    ordered = _expand_to_xnys_sessions(ordered)
    ordered["c_t"] = daily_share_change(ordered).to_numpy(dtype=np.float64)
    ordered["F_t"] = _forward_share_change(ordered)
    ordered["Y_create"] = np.where(
        ordered["F_t"].notna(),
        (ordered["F_t"] > 0).astype(np.float64),
        np.nan,
    )
    ordered["Y_redeem"] = np.where(
        ordered["F_t"].notna(),
        (ordered["F_t"] < 0).astype(np.float64),
        np.nan,
    )
    ordered["grid_position"] = ordered.groupby(
        "ticker", sort=False, observed=True
    ).cumcount()
    prior_change = ordered.groupby("ticker", sort=False, observed=True)[
        "c_t"
    ].transform(lambda values: values.shift(1).rolling(5, min_periods=5).sum())
    ordered["prior_5_c_sum_bps"] = prior_change * 10_000.0
    nav = ordered["nav"].astype("Float64")
    previous_nav = nav.groupby(ordered["ticker"], sort=False).shift(5)
    nav_values = nav.to_numpy(dtype=np.float64, na_value=np.nan)
    previous_values = previous_nav.to_numpy(dtype=np.float64, na_value=np.nan)
    valid_nav = (
        np.isfinite(nav_values)
        & np.isfinite(previous_values)
        & (nav_values > 0)
        & (previous_values > 0)
    )
    nav_return = np.full(len(ordered), np.nan, dtype=np.float64)
    nav_return[valid_nav] = (
        nav_values[valid_nav] / previous_values[valid_nav] - 1.0
    ) * 10_000.0
    ordered["nav_return_5_bps"] = nav_return
    return ordered


def _forward_share_change(flows: pd.DataFrame, horizon: int = 5) -> np.ndarray:
    result = np.full(len(flows), np.nan, dtype=np.float64)
    for _, indices in flows.groupby(
        "ticker", sort=False, observed=True
    ).indices.items():
        positions = np.asarray(indices, dtype=np.int64)
        shares = flows.loc[positions, "shares_outstanding"].to_numpy(dtype=object)
        status = (
            flows.loc[positions, "status"]
            .astype("string")
            .fillna("")
            .to_numpy(dtype=object)
        )
        for position in range(len(positions) - horizon):
            end = position + horizon
            start_shares = shares[position]
            end_shares = shares[end]
            if not all(
                _is_positive_share_count(value) for value in shares[position : end + 1]
            ) or np.any(status[position + 1 : end + 1] == "split_session"):
                continue
            result[positions[position]] = _relative_share_change(
                end_shares,
                start_shares,
            )
    return result


def _is_positive_share_count(value: object) -> bool:
    if pd.isna(cast(Any, value)):
        return False
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value) > 0
    number = float(cast(Any, value))
    return np.isfinite(number) and number > 0


def _relative_share_change(current: object, previous: object) -> float:
    if _is_integral_share_count(current) and _is_integral_share_count(previous):
        current_integer = int(cast(Any, current))
        previous_integer = int(cast(Any, previous))
        return (current_integer - previous_integer) / previous_integer
    current_number = float(cast(Any, current))
    previous_number = float(cast(Any, previous))
    return current_number / previous_number - 1.0


def _is_integral_share_count(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _expand_to_xnys_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex each fund's rows onto every XNYS session in its date range."""
    if frame.empty:
        return frame.copy()
    expanded_funds: list[pd.DataFrame] = []
    for ticker, fund in frame.groupby("ticker", sort=False, observed=True):
        fund = fund.copy()
        fund["date"] = pd.to_datetime(fund["date"])
        dates = pd.DatetimeIndex(fund["date"])
        sessions = xnys_sessions(dates.min().date(), dates.max().date())
        if not dates.isin(sessions).all():
            raise ValueError(f"Flow rows for {ticker} contain a non-XNYS date.")
        if fund["date"].duplicated().any():
            raise ValueError(f"Flow rows for {ticker} contain duplicate dates.")
        if "shares_outstanding" in fund:
            share_dtype = fund["shares_outstanding"].dtype
            if pd.api.types.is_unsigned_integer_dtype(share_dtype):
                fund["shares_outstanding"] = fund["shares_outstanding"].astype("UInt64")
            elif pd.api.types.is_integer_dtype(share_dtype):
                fund["shares_outstanding"] = fund["shares_outstanding"].astype("Int64")
        expanded = fund.set_index("date").reindex(sessions)
        expanded["ticker"] = ticker
        expanded["date"] = sessions
        expanded["grid_position"] = np.arange(len(sessions), dtype=np.int64)
        expanded_funds.append(expanded.reset_index(drop=True))
    return pd.concat(expanded_funds, ignore_index=True)


def _month_labels(dates: Sequence[object] | pd.Series) -> pd.Series:
    parsed = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    return parsed.dt.to_period("M").astype("string")


def _month_codes(
    months: Sequence[str] | pd.Series, plan: MonthBootstrapPlan
) -> np.ndarray:
    lookup = {month: index for index, month in enumerate(plan.months)}
    labels = pd.Series(months, dtype="string").astype(str).to_numpy()
    codes = np.asarray([lookup.get(month, -1) for month in labels], dtype=np.int64)
    if (codes < 0).any():
        raise ValueError("Bootstrap row month is absent from the shared plan.")
    return codes


def _bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    months: Sequence[str] | pd.Series,
    plan: MonthBootstrapPlan,
) -> np.ndarray:
    observations = np.asarray(values, dtype=np.float64).reshape(-1)
    codes = _month_codes(months, plan)
    if len(observations) != len(codes):
        raise ValueError("Bootstrap values and months must align.")
    valid = np.isfinite(observations)
    sums = np.bincount(
        codes[valid],
        weights=observations[valid],
        minlength=len(plan.months),
    )
    counts = np.bincount(codes[valid], minlength=len(plan.months)).astype(np.float64)
    numerator = plan.counts @ sums
    denominator = plan.counts @ counts
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(plan.counts), np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _finite_mean(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else np.nan


def _row_nanmean(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _percentile_interval(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return np.nan, np.nan
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper)


def _bin_label(lower: float, upper: float) -> str:
    return f"[{lower:g},{upper:g})"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}.")


def _demean_codes(values: np.ndarray, codes: np.ndarray, groups: int) -> np.ndarray:
    sums = np.zeros((groups, values.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, values)
    counts = np.bincount(codes, minlength=groups).astype(np.float64)
    return np.asarray(values - sums[codes] / counts[codes, None], dtype=np.float64)


def _weighted_within(
    values: np.ndarray, codes: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    residualized = values.copy()
    for code in np.unique(codes):
        rows = codes == code
        total_weight = float(weights[rows].sum())
        if total_weight > 0:
            mean = (residualized[rows] * weights[rows, None]).sum(axis=0)
            residualized[rows] -= mean / total_weight
    return residualized


def _local_linear_rd_fit(
    frame: pd.DataFrame,
    *,
    cutoff: float,
    outcome_column: str,
    running_column: str,
    fund_column: str,
    window: float,
    weights: np.ndarray,
    raise_on_rank: bool,
) -> tuple[float, int, int]:
    outcome = frame[outcome_column].to_numpy(dtype=np.float64)
    running = frame[running_column].to_numpy(dtype=np.float64)
    complete = (
        np.isfinite(outcome)
        & np.isfinite(running)
        & (np.abs(running - cutoff) <= window)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if not complete.any():
        return np.nan, 0, 0
    selected = frame.loc[complete]
    selected_outcome = outcome[complete]
    selected_running = running[complete]
    selected_weights = weights[complete]
    right = selected_running > cutoff
    distance = selected_running - cutoff
    regressors = np.column_stack((right, distance, distance * right)).astype(np.float64)
    fund_codes, fund_levels = pd.factorize(selected[fund_column], sort=True)
    if (fund_codes < 0).any():
        raise ValueError("RD fund identifiers cannot be missing.")
    within_outcome = _weighted_within(
        selected_outcome[:, None], fund_codes, selected_weights
    ).ravel()
    within_regressors = _weighted_within(regressors, fund_codes, selected_weights)
    square_root_weights = np.sqrt(selected_weights)
    weighted_x = within_regressors * square_root_weights[:, None]
    weighted_y = within_outcome * square_root_weights
    if np.linalg.matrix_rank(weighted_x) < weighted_x.shape[1]:
        if raise_on_rank:
            raise ValueError(f"RD design is rank deficient at cutoff {cutoff}.")
        return np.nan, int((~right).sum()), int(right.sum())
    coefficients = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)[0]
    return (
        float(coefficients[0]),
        int((~right).sum()),
        int(right.sum()),
    )
