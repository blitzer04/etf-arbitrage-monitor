"""Difference-in-differences checks for the effective date of Rule 6c-11."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd

from etf_monitor.arbitrage.identification import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    MonthBootstrapPlan,
    _percentile_interval,
    _prepare_flows,
    month_block_bootstrap,
    ols_month_clustered_cr1,
    two_way_demean,
)

MAIN_START = pd.Timestamp("2018-01-02")
RULE_DATE = pd.Timestamp("2019-12-23")
MAIN_END = pd.Timestamp("2021-12-31")
COVID_START = pd.Timestamp("2020-02-15")
COVID_END = pd.Timestamp("2020-06-30")
PLACEBO_DATE = pd.Timestamp("2018-12-24")
PLACEBO_END = pd.Timestamp("2019-12-13")
BOND_TICKERS = frozenset({"JNK", "SJNK", "SPAB", "SPSB", "BIL", "BWX", "EBND"})
US_BOND_TICKERS = frozenset({"JNK", "SJNK", "SPAB", "SPSB", "BIL"})
EXCLUDED_TICKERS = frozenset({"FEZ", "GII", "GNR", "SPDW", "SPEM"})
CORE_EQUITY_COUNT = 13
HALF_YEARS = tuple(f"{year}H{half}" for year in range(2018, 2022) for half in (1, 2))
BASE_HALF_YEAR = "2018H1"
MIN_VALID_BOOTSTRAP_DRAWS = 400

MAIN_REGRESSORS = (
    ("b0", "s"),
    ("b1", "s_post"),
    ("b2", "s_bond"),
    ("b3", "s_bond_post"),
)


def prepare_rule_6c11_panel(
    arbitrage_band: pd.DataFrame,
    core_band: pd.DataFrame,
    arbitrage_flows: pd.DataFrame,
    core_flows: pd.DataFrame,
) -> pd.DataFrame:
    """Join specified bond and core-equity bands to their flow outcomes.

    The returned rows use the signed, clipped cost-band multiple ``s`` and
    forward five-session share changes in decimal units and basis points.
    Every selected fund must have a band row on 2018-01-02 and valid
    pre-effective-date observations; otherwise the input does not fit the
    specified sample.

    Args:
        arbitrage_band: V2 band rows for the arbitrage universe.
        core_band: V2 band rows for the 13 core U.S. equity funds.
        arbitrage_flows: Matching arbitrage flow-proxy rows.
        core_flows: Matching core flow-proxy rows.

    Returns:
        A sorted panel containing eligible rows through the end of the input
        data, including treatment, timing, and flow-outcome columns.
    """
    _require_columns(arbitrage_band, ("ticker", "date"), "arbitrage band")
    _require_columns(core_band, ("ticker", "date"), "core band")
    core_tickers = frozenset(core_band["ticker"].dropna().astype(str).unique())
    if len(core_tickers) != CORE_EQUITY_COUNT:
        raise ValueError(
            "The core band must contain exactly 13 U.S. equity funds; "
            f"found {len(core_tickers)}."
        )
    if BOND_TICKERS & core_tickers:
        raise ValueError("Treated bond funds overlap the core control funds.")

    bond_rows = _prepare_fund_rows(
        arbitrage_band, arbitrage_flows, BOND_TICKERS, "arbitrage", bond=1
    )
    control_rows = _prepare_fund_rows(
        core_band, core_flows, core_tickers, "core", bond=0
    )
    panel = pd.concat([bond_rows, control_rows], ignore_index=True)
    panel["s"] = signed_band_multiple(panel["premium_discount_bps"], panel["band_bps"])
    panel["F_t_bps"] = panel["F_t"] * 10_000.0
    panel["month"] = panel["date"].dt.strftime("%Y-%m")
    panel = panel.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    return panel


def signed_band_multiple(
    premium_discount_bps: pd.Series,
    band_bps: pd.Series,
) -> pd.Series:
    """Return the signed premium-to-band multiple clipped to [-3, 3].

    Args:
        premium_discount_bps: Premium or discount in basis points.
        band_bps: Positive arbitrage band in basis points.

    Returns:
        A dimensionless signed multiple, clipped at three band widths.
    """
    return (premium_discount_bps / band_bps).clip(lower=-3.0, upper=3.0)


def run_rule_6c11_did(
    arbitrage_band: pd.DataFrame,
    core_band: pd.DataFrame,
    arbitrage_flows: pd.DataFrame,
    core_flows: pd.DataFrame,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Estimate the Rule 6c-11 main, dynamic, and robustness specifications.

    The primary outcome is the forward five-session share-change proxy in
    basis points. Month blocks are resampled jointly across funds, and the
    fund-by-post fixed effect absorbs fund-specific level changes.

    Args:
        arbitrage_band: V2 band rows for the arbitrage universe.
        core_band: V2 band rows for the core universe.
        arbitrage_flows: Matching arbitrage flow-proxy rows.
        core_flows: Matching core flow-proxy rows.
        draws: Number of month-block bootstrap draws.
        seed: Seed shared by the bootstrap tests.

    Returns:
        The three output tables and a JSON-serializable summary.
    """
    if draws < MIN_VALID_BOOTSTRAP_DRAWS:
        raise ValueError(
            f"At least {MIN_VALID_BOOTSTRAP_DRAWS} bootstrap draws are required."
        )
    panel = prepare_rule_6c11_panel(
        arbitrage_band, core_band, arbitrage_flows, core_flows
    )
    main_window = _study_window(panel, MAIN_START, MAIN_END, RULE_DATE)
    main_sample, excluded_crossing_rows = _exclude_forward_windows_crossing(
        main_window, RULE_DATE
    )
    main_plan = month_block_bootstrap(main_sample["month"], draws, seed)
    main_design = _main_design(main_sample)

    main_forward = _fit_model(
        main_sample,
        "F_t_bps",
        main_design,
        [term for term, _ in MAIN_REGRESSORS],
        ci_terms={"b3"},
        bootstrap_plan=main_plan,
        draws=draws,
        seed=seed,
    )
    creation_sample = main_sample.loc[main_sample["s"] >= 0].copy()
    main_creation = _fit_model(
        creation_sample,
        "Y_create",
        main_design.reindex(creation_sample.index),
        [term for term, _ in MAIN_REGRESSORS],
        ci_terms={"b3"},
        bootstrap_plan=main_plan,
        draws=draws,
        seed=seed,
    )
    main_rows = pd.concat(
        [
            main_forward.assign(outcome="F_t_bps"),
            main_creation.assign(outcome="Y_create"),
        ],
        ignore_index=True,
    )
    main_rows = main_rows[
        [
            "outcome",
            "term",
            "coefficient",
            "standard_error",
            "t_statistic",
            "lo",
            "hi",
            "n",
            "funds",
            "months",
            "valid_draws",
            "status",
        ]
    ]

    dynamic_design, dynamic_terms = _dynamic_design(main_sample)
    dynamic_rows = _fit_model(
        main_sample,
        "F_t_bps",
        dynamic_design,
        dynamic_terms,
        ci_terms=set(dynamic_terms),
        bootstrap_plan=main_plan,
        draws=draws,
        seed=seed,
    )
    dynamic_rows = dynamic_rows[
        [
            "term",
            "coefficient",
            "standard_error",
            "t_statistic",
            "lo",
            "hi",
            "n",
            "funds",
            "months",
            "valid_draws",
            "status",
        ]
    ]

    robust_rows = _robustness_rows(panel, main_sample, main_plan, draws, seed)
    main_counts = _bond_post_counts(main_sample)
    summary: dict[str, object] = {
        "seed": int(seed),
        "draws": int(draws),
        "windows": {
            "main": {
                "start": MAIN_START.date().isoformat(),
                "end": MAIN_END.date().isoformat(),
                "post_date": RULE_DATE.date().isoformat(),
            },
            "full_sample": {
                "start": MAIN_START.date().isoformat(),
                "end": panel["date"].max().date().isoformat(),
                "post_date": RULE_DATE.date().isoformat(),
            },
            "covid_exclusion": {
                "start": COVID_START.date().isoformat(),
                "end": COVID_END.date().isoformat(),
            },
            "placebo": {
                "start": MAIN_START.date().isoformat(),
                "end": PLACEBO_END.date().isoformat(),
                "post_date": PLACEBO_DATE.date().isoformat(),
            },
        },
        "treated_funds": sorted(BOND_TICKERS),
        "control_funds": sorted(
            set(panel.loc[panel["bond"].eq(0), "ticker"].astype(str))
        ),
        "excluded_funds": sorted(EXCLUDED_TICKERS),
        "row_counts": {
            "main_f_t": int(len(main_sample)),
            "main_y_create_nonnegative_s": int(len(creation_sample)),
            "main_by_bond_post": main_counts,
            "excluded_forward_windows_crossing_rule_date": excluded_crossing_rows,
            "dynamic": int(len(main_sample)),
            "robustness": {
                str(row["variant"]): int(row["n"])
                for row in robust_rows.to_dict(orient="records")
            },
        },
    }
    return (
        {
            "did_main": main_rows,
            "did_event": dynamic_rows,
            "did_robust": robust_rows,
        },
        summary,
    )


def _prepare_fund_rows(
    band: pd.DataFrame,
    flows: pd.DataFrame,
    tickers: Collection[str],
    name: str,
    *,
    bond: int,
) -> pd.DataFrame:
    _require_columns(
        band,
        (
            "ticker",
            "date",
            "premium_discount_bps",
            "band_bps",
            "status",
        ),
        f"{name} band",
    )
    raw = band.loc[band["ticker"].isin(tickers)].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    missing = sorted(set(tickers) - set(raw["ticker"].astype(str)))
    if missing:
        raise ValueError(f"{name} band is missing specified funds: {missing}.")
    starts = raw.groupby("ticker", observed=True)["date"].min()
    late_starts = starts.loc[starts > MAIN_START]
    if not late_starts.empty:
        raise ValueError(
            f"{name} band starts after {MAIN_START.date()} for "
            f"{late_starts.index.tolist()}."
        )
    flow_grid = _add_forward_end_session(_prepare_flows(flows))
    valid_band = raw.loc[
        (raw["status"] == "ok")
        & raw["band_bps"].notna()
        & (raw["band_bps"] > 0)
        & (raw["date"] >= MAIN_START)
    ].copy()
    band_values = valid_band["band_bps"].to_numpy(dtype=np.float64)
    premium_values = valid_band["premium_discount_bps"].to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if not np.isfinite(band_values).all() or (band_values <= 0).any():
        raise ValueError(f"{name} band contains nonpositive or non-finite band_bps.")
    if not np.isfinite(premium_values).all():
        raise ValueError(f"{name} ok band rows have missing premiums.")
    merged = valid_band.merge(
        flow_grid[["ticker", "date", "F_t", "F_t_date", "Y_create"]],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if merged["_merge"].ne("both").any():
        missing_rows = merged.loc[
            merged["_merge"].ne("both"), ["ticker", "date"]
        ].to_dict(orient="records")
        raise ValueError(
            f"{name} band rows are missing from the flow grid: {missing_rows[:5]}."
        )
    merged = merged.drop(columns="_merge")
    merged = merged.loc[merged["F_t"].notna()].copy()
    pre_funds = set(merged.loc[merged["date"] < RULE_DATE, "ticker"].astype(str))
    post_funds = set(merged.loc[merged["date"] >= RULE_DATE, "ticker"].astype(str))
    missing_pre = sorted(set(tickers) - pre_funds)
    missing_post = sorted(set(tickers) - post_funds)
    if missing_pre or missing_post:
        raise ValueError(
            f"{name} funds lack non-null forward outcomes before or after "
            f"{RULE_DATE.date()}: pre={missing_pre}, post={missing_post}."
        )
    merged["bond"] = bond
    return merged


def _study_window(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    post_date: pd.Timestamp,
) -> pd.DataFrame:
    dates = pd.to_datetime(panel["date"])
    within = dates >= start
    if end is not None:
        within &= dates <= end
    sample = panel.loc[within].copy()
    sample["post"] = (pd.to_datetime(sample["date"]) >= post_date).astype(np.int64)
    sample["ticker_post"] = [
        f"{ticker}|post={post}"
        for ticker, post in zip(
            sample["ticker"].astype(str), sample["post"], strict=True
        )
    ]
    return sample.reset_index(drop=True)


def _add_forward_end_session(flow_grid: pd.DataFrame) -> pd.DataFrame:
    """Attach each XNYS row's five-session outcome date to its flow row."""
    forward_grid = flow_grid.copy()
    forward_grid["F_t_date"] = forward_grid.groupby(
        "ticker", sort=False, observed=True
    )["date"].shift(-5)
    return forward_grid


def _exclude_forward_windows_crossing(
    sample: pd.DataFrame,
    post_date: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    crossing = (sample["date"] < post_date) & (sample["F_t_date"] >= post_date)
    return sample.loc[~crossing].copy(), int(crossing.sum())


def _main_design(frame: pd.DataFrame) -> pd.DataFrame:
    s = frame["s"].to_numpy(dtype=np.float64)
    post = frame["post"].to_numpy(dtype=np.float64)
    bond = frame["bond"].to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "s": s,
            "s_post": s * post,
            "s_bond": s * bond,
            "s_bond_post": s * bond * post,
        },
        index=frame.index,
    )


def _dynamic_design(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dates = pd.to_datetime(frame["date"])
    half_year = pd.Series(
        [f"{date.year}H{(date.month - 1) // 6 + 1}" for date in dates],
        index=frame.index,
    )
    s = frame["s"].to_numpy(dtype=np.float64)
    bond = frame["bond"].to_numpy(dtype=np.float64)
    values: dict[str, np.ndarray] = {}
    terms: list[str] = []
    for period in HALF_YEARS:
        term = f"common_s_{period}"
        values[term] = s * half_year.eq(period).to_numpy(dtype=np.float64)
        terms.append(term)
    values["bond_s_base_2018H1"] = s * bond
    terms.append("bond_s_base_2018H1")
    for period in HALF_YEARS:
        if period == BASE_HALF_YEAR:
            continue
        term = f"bond_s_{period}"
        values[term] = s * bond * half_year.eq(period).to_numpy(dtype=np.float64)
        terms.append(term)
    return pd.DataFrame(values, index=frame.index), terms


def _robustness_rows(
    panel: pd.DataFrame,
    main_sample: pd.DataFrame,
    main_plan: MonthBootstrapPlan,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    covid_sample = main_sample.loc[
        ~pd.to_datetime(main_sample["date"]).between(COVID_START, COVID_END)
    ].copy()
    covid_plan = month_block_bootstrap(covid_sample["month"], draws, seed)
    full_window = _study_window(panel, MAIN_START, None, RULE_DATE)
    full_sample, _ = _exclude_forward_windows_crossing(full_window, RULE_DATE)
    full_plan = month_block_bootstrap(full_sample["month"], draws, seed)
    us_bond_sample = main_sample.loc[
        main_sample["ticker"].isin(
            US_BOND_TICKERS | set(panel.loc[panel["bond"].eq(0), "ticker"].astype(str))
        )
    ].copy()
    placebo_sample = _study_window(panel, MAIN_START, PLACEBO_END, PLACEBO_DATE)
    placebo_plan = month_block_bootstrap(placebo_sample["month"], draws, seed)
    creation_ex_covid = covid_sample.loc[covid_sample["s"] >= 0].copy()

    specifications = [
        ("exclude_covid", covid_sample, "F_t_bps", covid_plan),
        ("full_sample", full_sample, "F_t_bps", full_plan),
        ("us_bonds_only", us_bond_sample, "F_t_bps", main_plan),
        ("placebo_2018-12-24", placebo_sample, "F_t_bps", placebo_plan),
        ("create_exclude_covid", creation_ex_covid, "Y_create", covid_plan),
    ]
    records: list[dict[str, object]] = []
    for variant, sample, outcome, plan in specifications:
        design = _main_design(sample)
        result = _fit_model(
            sample,
            outcome,
            design,
            [term for term, _ in MAIN_REGRESSORS],
            ci_terms={"b3"},
            bootstrap_plan=plan,
            draws=draws,
            seed=seed,
        )
        row = result.loc[result["term"].eq("b3")].iloc[0].to_dict()
        row["variant"] = variant
        row["outcome"] = outcome
        records.append({str(key): value for key, value in row.items()})
    return pd.DataFrame.from_records(
        records,
        columns=[
            "variant",
            "outcome",
            "term",
            "coefficient",
            "standard_error",
            "t_statistic",
            "lo",
            "hi",
            "n",
            "funds",
            "months",
            "valid_draws",
            "status",
        ],
    )


def _fit_model(
    sample: pd.DataFrame,
    outcome: str,
    design: pd.DataFrame,
    terms: Sequence[str],
    *,
    ci_terms: set[str],
    bootstrap_plan: MonthBootstrapPlan | None,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    if len(terms) != design.shape[1]:
        raise ValueError("Regression term labels do not match the design.")
    complete = sample.copy()
    for column in design:
        complete[column] = design[column]
    complete = complete.dropna(
        subset=[outcome, "month", "ticker_post", *design.columns]
    ).reset_index(drop=True)
    funds = int(complete["ticker"].nunique())
    months = int(complete["month"].nunique())
    count = len(complete)
    estimates = np.full(len(terms), np.nan, dtype=np.float64)
    standard_errors = np.full(len(terms), np.nan, dtype=np.float64)
    t_statistics = np.full(len(terms), np.nan, dtype=np.float64)
    intervals = np.full((len(terms), 2), np.nan, dtype=np.float64)
    status = "insufficient_support"
    valid_draws = 0
    support_ok = funds >= 3 and months >= 24 and count > len(terms)
    if support_ok:
        demeaned = _demeaned_design(complete, outcome, list(design.columns))
        regressors = demeaned[list(design.columns)].to_numpy(dtype=np.float64)
        if np.linalg.matrix_rank(regressors) == regressors.shape[1]:
            result = ols_month_clustered_cr1(
                demeaned[outcome].to_numpy(dtype=np.float64),
                regressors,
                complete["month"].astype(str).to_numpy(),
            )
            estimates = result.coefficients
            standard_errors = result.standard_errors
            t_statistics = result.t_statistics
            status = "ok"
            if ci_terms and draws > 0:
                plan = bootstrap_plan or month_block_bootstrap(
                    complete["month"], draws, seed
                )
                bootstrap_estimates = _bootstrap_coefficients(
                    complete,
                    outcome,
                    list(design.columns),
                    plan,
                )
                valid_draws = len(bootstrap_estimates)
                if valid_draws >= MIN_VALID_BOOTSTRAP_DRAWS:
                    for index, term in enumerate(terms):
                        if term in ci_terms:
                            intervals[index] = _percentile_interval(
                                bootstrap_estimates[:, index]
                            )
                else:
                    status = "unstable_bootstrap"
    records = []
    for index, term in enumerate(terms):
        if status == "unstable_bootstrap" and term in ci_terms:
            row_status = "unstable_bootstrap"
        elif status == "unstable_bootstrap":
            row_status = "ok"
        else:
            row_status = status
        records.append(
            {
                "term": term,
                "coefficient": estimates[index],
                "standard_error": standard_errors[index],
                "t_statistic": t_statistics[index],
                "lo": intervals[index, 0],
                "hi": intervals[index, 1],
                "n": count,
                "funds": funds,
                "months": months,
                "valid_draws": valid_draws,
                "status": row_status,
            }
        )
    return pd.DataFrame.from_records(records)


def _demeaned_design(
    frame: pd.DataFrame,
    outcome: str,
    regressors: Sequence[str],
) -> pd.DataFrame:
    return two_way_demean(
        frame,
        [outcome, *regressors],
        fund_column="ticker_post",
        date_column="date",
    )


def _bootstrap_coefficients(
    sample: pd.DataFrame,
    outcome: str,
    regressors: Sequence[str],
    plan: MonthBootstrapPlan,
) -> np.ndarray:
    month_groups = {
        month: group
        for month, group in sample.groupby("month", sort=False, observed=True)
    }
    estimates: list[np.ndarray] = []
    for draw_counts in plan.counts:
        blocks: list[pd.DataFrame] = []
        for month, multiplicity in zip(plan.months, draw_counts, strict=True):
            block = month_groups.get(month)
            if block is not None and multiplicity:
                blocks.extend([block] * int(multiplicity))
        if not blocks:
            continue
        bootstrap_sample = pd.concat(blocks, ignore_index=True)
        demeaned = _demeaned_design(bootstrap_sample, outcome, regressors)
        x = demeaned[list(regressors)].to_numpy(dtype=np.float64)
        if np.linalg.matrix_rank(x) < x.shape[1]:
            continue
        if bootstrap_sample["month"].nunique() < 2:
            continue
        result = ols_month_clustered_cr1(
            demeaned[outcome].to_numpy(dtype=np.float64),
            x,
            bootstrap_sample["month"].astype(str).to_numpy(),
        )
        estimates.append(result.coefficients)
    if not estimates:
        return np.empty((0, len(regressors)), dtype=np.float64)
    return np.vstack(estimates)


def _bond_post_counts(frame: pd.DataFrame) -> list[dict[str, int]]:
    counts = frame.groupby(["bond", "post"], observed=True).size()
    return [
        {"bond": bond, "post": post, "rows": int(counts.get((bond, post), 0))}
        for bond in (0, 1)
        for post in (0, 1)
    ]


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}.")
