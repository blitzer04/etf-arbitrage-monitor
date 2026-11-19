"""Fictional hand-calculated tests for the Rule 6c-11 DiD analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_monitor.arbitrage.identification import two_way_demean
from etf_monitor.arbitrage.rule_6c11 import (
    MAIN_REGRESSORS,
    PLACEBO_END,
    RULE_DATE,
    _add_forward_end_session,
    _dynamic_design,
    _exclude_forward_windows_crossing,
    _fit_model,
    _main_design,
    _prepare_fund_rows,
    _study_window,
    signed_band_multiple,
)
from etf_monitor.cli import _build_parser
from etf_monitor.xnys import xnys_sessions


def test_signed_band_multiple_clips_at_three_band_widths() -> None:
    premium = pd.Series([80.0, -80.0, 20.0])
    band = pd.Series([10.0, 10.0, 10.0])

    result = signed_band_multiple(premium, band)

    pd.testing.assert_series_equal(result, pd.Series([3.0, -3.0, 2.0]))


def test_ap_6c11_cli_accepts_the_four_input_directories_and_output() -> None:
    args = _build_parser().parse_args(
        [
            "ap-6c11-did",
            "--arbitrage-band",
            "arbitrage-band",
            "--core-band",
            "core-band",
            "--arbitrage-flows",
            "arbitrage-flows",
            "--core-flows",
            "core-flows",
            "--out",
            "output",
        ]
    )

    assert args.command == "ap-6c11-did"
    assert str(args.arbitrage_band) == "arbitrage-band"
    assert str(args.core_band) == "core-band"
    assert str(args.arbitrage_flows) == "arbitrage-flows"
    assert str(args.core_flows) == "core-flows"
    assert str(args.out) == "output"


def test_fund_post_fixed_effect_absorbs_a_fund_specific_level_shift() -> None:
    rows: list[dict[str, object]] = []
    dates = pd.to_datetime(["2018-01-31", "2018-02-28", "2020-01-31", "2020-02-28"])
    for fund_index, ticker in enumerate(("AAA", "BBB", "CCC")):
        for date_index, date in enumerate(dates):
            post = int(date >= pd.Timestamp("2019-12-23"))
            rows.append(
                {
                    "ticker_post": f"{ticker}|{post}",
                    "date": date,
                    "value": (
                        5.0 * date_index
                        + 17.0 * fund_index
                        + 31.0 * (fund_index + 1) * post
                    ),
                }
            )
    frame = pd.DataFrame(rows)

    residualized = two_way_demean(
        frame,
        ["value"],
        fund_column="ticker_post",
        date_column="date",
    )

    assert np.allclose(residualized["value"], 0.0, atol=1e-10)


def test_main_did_recovers_known_bond_post_slope() -> None:
    frame = _synthetic_panel()
    sample = _study_window(
        frame,
        pd.Timestamp("2018-01-02"),
        pd.Timestamp("2021-12-31"),
        pd.Timestamp("2019-12-23"),
    )
    design = _main_design(sample)
    terms = [term for term, _ in MAIN_REGRESSORS]

    estimates = _fit_model(
        sample,
        "outcome",
        design,
        terms,
        ci_terms=set(),
        bootstrap_plan=None,
        draws=0,
        seed=20261118,
    )

    assert estimates.loc[estimates["term"].eq("b3"), "coefficient"].iloc[0] == (
        pytest.approx(3.25)
    )
    assert estimates["status"].eq("ok").all()


def test_dynamic_design_uses_2018h1_as_the_base_period() -> None:
    dates = pd.to_datetime(
        [
            "2018-03-30",
            "2018-09-28",
            "2019-03-29",
            "2019-09-30",
            "2020-03-31",
            "2020-09-30",
            "2021-03-31",
            "2021-09-30",
        ]
    )
    frame = pd.DataFrame(
        {
            "date": dates,
            "s": np.arange(1.0, 9.0),
            "bond": 1,
        }
    )

    design, terms = _dynamic_design(frame)

    assert design.loc[0, "common_s_2018H1"] == 1.0
    assert design.loc[0, "bond_s_base_2018H1"] == 1.0
    assert "bond_s_2018H1" not in terms
    assert design.loc[1, "common_s_2018H2"] == 2.0
    assert design.loc[1, "bond_s_2018H2"] == 2.0
    assert len(terms) == 16


def test_placebo_window_ends_before_the_rule_effective_date() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 5,
            "date": pd.to_datetime(
                [
                    "2018-01-02",
                    "2018-12-21",
                    "2018-12-24",
                    "2019-12-13",
                    "2019-12-16",
                ]
            ),
            "s": [0.1, 0.2, 0.3, 0.4, 0.5],
            "bond": [1, 1, 1, 1, 1],
        }
    )

    sample = _study_window(
        frame,
        pd.Timestamp("2018-01-02"),
        PLACEBO_END,
        pd.Timestamp("2018-12-24"),
    )

    assert sample["date"].max() == PLACEBO_END
    assert sample["post"].tolist() == [0, 0, 1, 1]


def test_forward_outcomes_crossing_rule_date_are_excluded() -> None:
    dates = xnys_sessions(
        pd.Timestamp("2019-12-13").date(),
        pd.Timestamp("2019-12-31").date(),
    )
    grid = _add_forward_end_session(pd.DataFrame({"ticker": "AAA", "date": dates}))
    outcomes_available = grid.loc[grid["F_t_date"].notna()].copy()
    window = _study_window(
        outcomes_available,
        pd.Timestamp("2019-12-13"),
        pd.Timestamp("2019-12-31"),
        RULE_DATE,
    )

    filtered, excluded = _exclude_forward_windows_crossing(window, RULE_DATE)

    assert excluded == 5
    assert filtered.loc[filtered["date"] < RULE_DATE, "date"].tolist() == [
        pd.Timestamp("2019-12-13")
    ]
    assert filtered.loc[filtered["date"] == RULE_DATE, "date"].tolist() == [RULE_DATE]


def test_pre_rule_support_requires_non_null_forward_outcomes() -> None:
    dates = xnys_sessions(
        pd.Timestamp("2018-01-02").date(),
        pd.Timestamp("2020-01-10").date(),
    )
    shares = pd.Series(
        [pd.NA if date == pd.Timestamp("2018-01-09") else 100 for date in dates],
        dtype="Int64",
    )
    flows = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": dates,
            "shares_outstanding": shares,
            "nav": 100.0,
            "status": "ok",
        }
    )
    band = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": pd.to_datetime(["2018-01-02", "2020-01-02"]),
            "premium_discount_bps": [1.0, 1.0],
            "band_bps": [2.0, 2.0],
            "status": ["ok", "ok"],
        }
    )

    with pytest.raises(ValueError, match=r"non-null forward outcomes.*pre=\['AAA'\]"):
        _prepare_fund_rows(band, flows, {"AAA"}, "fictional", bond=1)


def test_support_rule_emits_nan_rows_for_fewer_than_three_funds() -> None:
    frame = _synthetic_panel().loc[
        lambda data: (
            data["ticker"].isin(["BOND0", "EQ0"])
            & data["date"].le(pd.Timestamp("2018-12-31"))
        )
    ]
    sample = _study_window(
        frame,
        pd.Timestamp("2018-01-02"),
        pd.Timestamp("2018-12-31"),
        pd.Timestamp("2019-12-23"),
    )
    design = _main_design(sample)
    terms = [term for term, _ in MAIN_REGRESSORS]

    result = _fit_model(
        sample,
        "outcome",
        design,
        terms,
        ci_terms=set(),
        bootstrap_plan=None,
        draws=0,
        seed=20261118,
    )

    assert result["status"].eq("insufficient_support").all()
    assert result["coefficient"].isna().all()
    assert result["standard_error"].isna().all()


def _synthetic_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2018-01-31", periods=48, freq="ME")
    for fund_index in range(6):
        ticker = f"BOND{fund_index}" if fund_index < 3 else f"EQ{fund_index - 3}"
        bond = int(fund_index < 3)
        for date_index, date in enumerate(dates):
            post = int(date >= pd.Timestamp("2019-12-23"))
            s = (
                (fund_index * 7 + date_index * 5 + fund_index * date_index % 9) % 19 - 9
            ) / 3
            level = 11.0 * fund_index + post * (2.0 * fund_index - 4.0)
            date_effect = 6.0 * np.sin(date_index / 4.0)
            outcome = (
                0.7 * s
                + 0.2 * s * post
                + 0.1 * s * bond
                + 3.25 * s * bond * post
                + level
                + date_effect
            )
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "month": date.strftime("%Y-%m"),
                    "s": s,
                    "bond": bond,
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)
