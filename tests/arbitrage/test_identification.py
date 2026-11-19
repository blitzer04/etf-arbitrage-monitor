"""Hand-calculated causal-evidence checks using fictional observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_monitor.arbitrage.identification import (
    _prepare_flows,
    _rd_jump_table,
    daily_share_change,
    dose_response_bins,
    event_study,
    find_onsets,
    local_linear_rd_jump,
    month_block_bootstrap,
    ols_month_clustered_cr1,
    panel_regression,
    rolling_past_beta,
    stale_placebo_regression,
    two_way_demean,
)
from etf_monitor.cli import _validate_premium_flow_source
from etf_monitor.xnys import xnys_sessions


def test_onsets_require_a_change_into_premium_or_discount() -> None:
    band = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "status": "ok",
            "band_bps": 10.0,
            "side": ["inside", "premium", "premium", "discount", "discount", "inside"],
        }
    )

    onsets = find_onsets(band)

    assert list(
        zip(onsets["date"].dt.strftime("%Y-%m-%d"), onsets["side"], strict=True)
    ) == [
        ("2024-01-02", "premium"),
        ("2024-01-04", "discount"),
    ]


def test_daily_share_change_is_missing_on_a_split_session() -> None:
    flows = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "shares_outstanding": [100.0, 200.0, 220.0, 230.0],
            "status": [
                "ok",
                "split_session",
                "ok",
                "missing_previous_session",
            ],
        }
    )

    change = daily_share_change(flows)

    assert pd.isna(change.iloc[0])
    assert pd.isna(change.iloc[1])
    assert change.iloc[2] == pytest.approx(0.1)
    assert pd.isna(change.iloc[3])


def test_flow_windows_use_xnys_sessions_when_an_input_session_is_missing() -> None:
    sessions = xnys_sessions(
        pd.Timestamp("2024-07-01").date(),
        pd.Timestamp("2024-07-15").date(),
    )
    missing_date = pd.Timestamp("2024-07-09")
    observed_sessions = sessions[sessions != missing_date]
    flows = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": observed_sessions,
            "shares_outstanding": np.arange(100, 100 + len(observed_sessions)),
            "nav": 100.0,
            "status": "ok",
        }
    )

    prepared = _prepare_flows(flows)
    by_date = prepared.set_index("date")

    assert len(prepared) == len(sessions)
    assert pd.isna(by_date.loc[missing_date, "shares_outstanding"])
    assert pd.isna(by_date.loc[missing_date, "c_t"])
    assert pd.isna(by_date.loc[pd.Timestamp("2024-07-10"), "c_t"])
    assert pd.isna(by_date.loc[pd.Timestamp("2024-07-08"), "F_t"])
    assert pd.isna(by_date.loc[pd.Timestamp("2024-07-15"), "prior_5_c_sum_bps"])

    band = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": [pd.Timestamp("2024-07-03"), pd.Timestamp("2024-07-08")],
            "status": "ok",
            "band_bps": 10.0,
            "side": ["inside", "premium"],
            "c_t": 0.1,
        }
    )
    complete_flow_grid = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": sessions,
            "c_t": 0.1,
        }
    )
    sparse_flow_rows = complete_flow_grid.loc[
        complete_flow_grid["date"].ne(missing_date)
    ]
    complete_events, _ = event_study(
        band,
        complete_flow_grid,
        {"fictional": {"AAA"}},
        draws=8,
        seed=19,
    )
    sparse_events, _ = event_study(
        band,
        sparse_flow_rows,
        {"fictional": {"AAA"}},
        draws=8,
        seed=19,
    )
    complete_premium = complete_events.loc[complete_events["side"].eq("premium")]
    sparse_premium = sparse_events.loc[sparse_events["side"].eq("premium")]
    affected = sparse_premium.loc[sparse_premium["k"].eq(1)].iloc[0]

    assert np.isnan(affected["uplift"])
    pd.testing.assert_frame_equal(
        sparse_premium.loc[sparse_premium["k"].ne(1)].reset_index(drop=True),
        complete_premium.loc[complete_premium["k"].ne(1)].reset_index(drop=True),
    )


def test_integer_share_changes_remain_exact_above_float_precision() -> None:
    sessions = xnys_sessions(
        pd.Timestamp("2024-01-02").date(),
        pd.Timestamp("2024-01-09").date(),
    )
    base_shares = 2**53
    counts = [
        base_shares,
        base_shares + 1,
        base_shares,
        base_shares,
        base_shares,
        base_shares + 1,
    ]
    flows = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": sessions,
            "shares_outstanding": counts,
            "nav": 100.0,
            "status": "ok",
        }
    )

    prepared = _prepare_flows(flows)
    fractional_change = daily_share_change(
        pd.DataFrame(
            {
                "ticker": "BBB",
                "date": sessions[:2],
                "shares_outstanding": [1.5, 1.75],
                "status": "ok",
            }
        )
    )

    expected_change = 1 / base_shares
    daily_changes = prepared["c_t"].to_numpy(dtype=np.float64)
    forward_changes = prepared["F_t"].to_numpy(dtype=np.float64)
    create_indicators = prepared["Y_create"].to_numpy(dtype=np.float64)
    assert daily_changes[1] == pytest.approx(expected_change)
    assert daily_changes[1] > 0
    assert forward_changes[0] == pytest.approx(expected_change)
    assert forward_changes[0] > 0
    assert create_indicators[0] == 1
    assert fractional_change.iloc[1] == pytest.approx(1.75 / 1.5 - 1)


@pytest.mark.parametrize(
    "rows",
    [
        [
            ("A", "2024-01-01", 2.0, 1.0),
            ("A", "2024-01-02", 4.0, 2.0),
            ("B", "2024-01-01", 3.0, 0.0),
            ("B", "2024-01-02", 7.0, 4.0),
        ],
        [
            ("A", "2024-01-01", 1.0, 5.0),
            ("A", "2024-01-02", 3.0, 1.0),
            ("A", "2024-01-03", 4.0, 2.0),
            ("B", "2024-01-01", 2.0, 2.0),
            ("B", "2024-01-03", 8.0, 5.0),
            ("C", "2024-01-02", 0.0, 3.0),
            ("C", "2024-01-03", 6.0, 1.0),
        ],
    ],
    ids=["balanced", "unbalanced"],
)
def test_two_way_demeaning_matches_brute_force_dummy_ols(
    rows: list[tuple[str, str, float, float]],
) -> None:
    frame = pd.DataFrame(rows, columns=["ticker", "date", "y", "x"])
    frame["date"] = pd.to_datetime(frame["date"])
    demeaned = two_way_demean(frame, ["y", "x"])
    fund_dummies = pd.get_dummies(frame["ticker"], drop_first=True, dtype=float)
    date_dummies = pd.get_dummies(frame["date"], drop_first=True, dtype=float)
    design = np.column_stack(
        [
            np.ones(len(frame)),
            fund_dummies.to_numpy(),
            date_dummies.to_numpy(),
        ]
    )

    for column in ("y", "x"):
        observed = frame[column].to_numpy(dtype=np.float64)
        coefficients = np.linalg.lstsq(design, observed, rcond=None)[0]
        expected_residuals = observed - design @ coefficients
        assert demeaned[column].to_numpy() == pytest.approx(expected_residuals)


def test_month_clustered_cr1_standard_error_matches_hand_calculation() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([1.0, 2.0, 2.0, 4.0, 3.0, 5.0])
    months = np.array(["a", "a", "b", "b", "c", "c"])

    result = ols_month_clustered_cr1(y, x, months)

    coefficient = float(x @ y / (x @ x))
    residuals = y - coefficient * x
    cluster_scores = np.array(
        [
            np.sum(x[:2] * residuals[:2]),
            np.sum(x[2:4] * residuals[2:4]),
            np.sum(x[4:] * residuals[4:]),
        ]
    )
    correction = 3 / 2 * (6 - 1) / (6 - 1)
    expected_se = np.sqrt(correction * np.sum(cluster_scores**2) / (x @ x) ** 2)

    assert result.coefficients[0] == pytest.approx(coefficient)
    assert result.standard_errors[0] == pytest.approx(expected_se)
    assert result.observations == 6
    assert result.clusters == 3


def test_local_linear_rd_recovers_known_tau_with_fund_fixed_effects() -> None:
    cutoff = 0.5
    running = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    records: list[dict[str, object]] = []
    for fund_index, ticker in enumerate(("AAA", "BBB", "CCC")):
        right = (running > cutoff).astype(float)
        distance = running - cutoff
        outcome = (
            2.0 * fund_index + 0.35 * right + 0.7 * distance - 0.2 * distance * right
        )
        records.extend(
            {"ticker": ticker, "s": value, "outcome": result}
            for value, result in zip(running, outcome, strict=True)
        )

    estimate = local_linear_rd_jump(
        pd.DataFrame(records),
        cutoff=cutoff,
    )

    assert estimate["tau"] == pytest.approx(0.35)
    assert estimate["n_left"] == 9
    assert estimate["n_right"] == 9


def test_rolling_beta_uses_only_rows_before_the_current_session() -> None:
    panel = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "premium_discount_bps": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0],
            "market_return_bps": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    changed = panel.copy()
    changed.loc[4, "premium_discount_bps"] = 1000.0

    original_beta = rolling_past_beta(panel, window=3, min_observations=3).loc[
        4, "beta"
    ]
    changed_beta = rolling_past_beta(changed, window=3, min_observations=3).loc[
        4, "beta"
    ]

    assert original_beta == pytest.approx(2.0)
    assert changed_beta == pytest.approx(original_beta)


def test_month_block_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    months = pd.Series(["2024-02", "2024-01", "2024-02", "2024-03"])

    first = month_block_bootstrap(months, draws=12, seed=17)
    second = month_block_bootstrap(months, draws=12, seed=17)

    assert first.months == ("2024-01", "2024-02", "2024-03")
    assert np.array_equal(first.counts, second.counts)
    assert np.all(first.counts.sum(axis=1) == len(first.months))


def test_dose_response_uses_left_closed_multiple_bins() -> None:
    multiples = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
    panel = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": pd.date_range("2024-01-01", periods=len(multiples), freq="MS"),
            "r_t": multiples,
            "Y_create": np.arange(len(multiples), dtype=float) % 2,
            "Y_redeem": 0.0,
        }
    )

    bins = dose_response_bins(
        panel,
        {"fictional": {"AAA"}},
        draws=20,
        seed=11,
    )

    assert bins.loc[bins["side"].eq("premium"), "n"].tolist() == [1] * 9
    assert bins.loc[bins["side"].eq("premium"), "bin"].tolist() == [
        "[0,0.25)",
        "[0.25,0.5)",
        "[0.5,0.75)",
        "[0.75,1)",
        "[1,1.25)",
        "[1.25,1.5)",
        "[1.5,2)",
        "[2,3)",
        "[3,inf)",
    ]


def test_rd_emits_insufficient_support_row() -> None:
    panel = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "r_t": [1.6, 1.8, 1.9, 2.0],
            "Y_create": [0.0, 1.0, 0.0, 1.0],
            "Y_redeem": np.nan,
        }
    )
    plan = month_block_bootstrap(panel["date"].dt.strftime("%Y-%m"), draws=8)

    rows = _rd_jump_table(panel, {"fictional": {"AAA"}}, plan)
    row = rows.loc[rows["side"].eq("premium") & rows["cutoff"].eq(2.0)].iloc[0]

    assert row["n_left"] == 4
    assert row["n_right"] == 0
    assert row["status"] == "insufficient_support"
    assert np.isnan(row["tau"])
    assert np.isnan(row["lo"])
    assert np.isnan(row["hi"])


def test_rd_records_count_of_rank_deficient_bootstrap_draws() -> None:
    left_values = np.linspace(1.55, 1.95, 10)
    right_values = np.linspace(2.05, 2.45, 10)
    dates_by_side = (
        pd.date_range("2024-01-01", periods=10, freq="D"),
        pd.date_range("2024-02-01", periods=10, freq="D"),
    )
    records: list[dict[str, object]] = []
    for fund_index, ticker in enumerate(("AAA", "BBB", "CCC")):
        for running, dates, right in (
            (left_values, dates_by_side[0], False),
            (right_values, dates_by_side[1], True),
        ):
            for session, value in zip(dates, running, strict=True):
                distance = value - 2.0
                outcome = (
                    0.5 * fund_index
                    + 0.2 * right
                    + 0.4 * distance
                    + 0.1 * distance * right
                )
                records.append(
                    {
                        "ticker": ticker,
                        "date": session,
                        "r_t": value,
                        "Y_create": outcome,
                        "Y_redeem": np.nan,
                    }
                )
    panel = pd.DataFrame(records)
    plan = month_block_bootstrap(
        panel["date"].dt.strftime("%Y-%m"), draws=500, seed=20261118
    )
    expected_valid = int(((plan.counts[:, 0] > 0) & (plan.counts[:, 1] > 0)).sum())

    rows = _rd_jump_table(panel, {"fictional": {"AAA", "BBB", "CCC"}}, plan)
    row = rows.loc[rows["side"].eq("premium") & rows["cutoff"].eq(2.0)].iloc[0]

    assert row["n_left"] == 30
    assert row["n_right"] == 30
    assert row["valid_draws"] == expected_valid
    assert 0 < row["valid_draws"] < 400
    assert row["status"] == "unstable_bootstrap"
    assert np.isfinite(row["tau"])
    assert np.isnan(row["lo"])
    assert np.isnan(row["hi"])


def test_stale_and_panel_regressions_mark_small_samples_insufficient() -> None:
    dates = pd.date_range("2022-01-01", periods=25, freq="MS")
    stale_rows = []
    panel_rows = []
    for fund_index, ticker in enumerate(("AAA", "BBB")):
        for session_index, session in enumerate(dates):
            month = session.strftime("%Y-%m")
            premium = float(fund_index + session_index)
            stale_rows.append(
                {
                    "ticker": ticker,
                    "date": session,
                    "month": month,
                    "premium_discount_bps": premium,
                    "mech_bps": premium * 0.5,
                    "resid_bps": premium * 0.5,
                    "beta": 0.5,
                    "F_t_bps": premium,
                    "Y_create": float(session_index % 2),
                }
            )
            panel_rows.append(
                {
                    "ticker": ticker,
                    "date": session,
                    "month": month,
                    "premium_discount_bps": premium,
                    "band_bps": 2.0,
                    "c_t_bps": premium,
                    "prior_5_c_sum_bps": premium + 1.0,
                    "nav_return_5_bps": premium + 2.0,
                    "F_t_bps": premium + 3.0,
                }
            )

    stale, _ = stale_placebo_regression(pd.DataFrame(stale_rows), {"AAA", "BBB"})
    panel, _ = panel_regression(pd.DataFrame(panel_rows), {"two_funds": {"AAA", "BBB"}})

    assert len(stale) == 4
    assert stale["status"].eq("insufficient_support").all()
    assert stale["coefficient"].isna().all()
    assert len(panel) == 6
    assert panel["status"].eq("insufficient_support").all()
    assert panel["coefficient_per_10_bps"].isna().all()


def test_band_v2_resolves_recorded_relative_flow_path_from_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    band_dir = tmp_path / "custom-output" / "arbitrage_band" / "band-v2"
    flows_dir = tmp_path / "data" / "premium_flows" / "flows-v2"
    band_dir.mkdir(parents=True)
    flows_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    (band_dir / "summary.json").write_text(
        json.dumps({"premium_flows": "data/premium_flows/flows-v2"})
    )

    _validate_premium_flow_source(band_dir, flows_dir)

    with pytest.raises(ValueError, match="does not match the v2 band source"):
        _validate_premium_flow_source(band_dir, tmp_path / "different-flows")
