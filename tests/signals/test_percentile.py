"""Hand-calculated tests for per-ETF prior-session midrank percentiles."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from etf_monitor.signals.percentile import normalize_midrank_percentiles
from etf_monitor.xnys import xnys_sessions


def _sessions(count: int) -> pd.DatetimeIndex:
    sessions = xnys_sessions(date(2018, 1, 2), date(2025, 12, 31))
    return sessions[:count]


def _normalize(
    values: list[float],
    eligible: list[bool] | None = None,
    *,
    window: int = 756,
    minimum: int = 1,
) -> pd.DataFrame:
    return normalize_midrank_percentiles(
        values,
        _sessions(len(values)),
        [True] * len(values) if eligible is None else eligible,
        reference_window_sessions=window,
        minimum_reference_observations=minimum,
    )


def test_hand_calculated_midrank_and_ties() -> None:
    result = _normalize([1.0, 2.0, 3.0, 4.0, 3.0], window=5, minimum=4)

    # (2 lower references + half of 1 exact tie) / 4 * 100 = 62.5.
    assert result["percentile"].iloc[-1] == 62.5
    assert result["reference_count"].iloc[-1] == 4

    tied = _normalize([4.0, 4.0], window=5, minimum=1)
    above = _normalize([2.0, 3.0], window=5, minimum=1)
    below = _normalize([2.0, 1.0], window=5, minimum=1)
    assert tied["percentile"].iloc[-1] == 50.0
    assert above["percentile"].iloc[-1] == 100.0
    assert below["percentile"].iloc[-1] == 0.0


def test_current_observation_is_excluded_and_reference_dates_are_reported() -> None:
    sessions = _sessions(3)
    result = normalize_midrank_percentiles(
        [1.0, 2.0, 3.0],
        sessions,
        [True, True, True],
        reference_window_sessions=3,
        minimum_reference_observations=2,
    )

    assert result["reference_count"].iloc[2] == 2
    assert result["percentile"].iloc[2] == 100.0
    assert result["reference_first_session"].iloc[2] == sessions[0]
    assert result["reference_last_session"].iloc[2] == sessions[1]


def test_reference_minimum_is_251_vs_252_eligible_observations() -> None:
    values = [float(value) for value in range(253)]
    result = _normalize(values, window=756, minimum=252)

    assert result["reference_count"].iloc[251] == 251
    assert result["normalization_status"].iloc[251] == "insufficient_reference"
    assert np.isnan(result["percentile"].iloc[251])
    assert result["reference_count"].iloc[252] == 252
    assert result["normalization_status"].iloc[252] == "available"


def test_window_drops_t_minus_756_and_retains_t_minus_755() -> None:
    values = [1.0] * 757
    values[0] = -10.0
    values[1] = 0.0
    values[756] = 0.5
    sessions = _sessions(len(values))
    result = normalize_midrank_percentiles(
        values,
        sessions,
        [True] * len(values),
        reference_window_sessions=756,
        minimum_reference_observations=1,
    )

    assert result["reference_first_session"].iloc[756] == sessions[1]
    assert result["reference_last_session"].iloc[756] == sessions[755]
    assert result["reference_count"].iloc[756] == 755
    assert result["percentile"].iloc[756] == pytest.approx(100.0 / 755)


def test_raw_ineligible_and_empty_references_preserve_nan_and_nat() -> None:
    result = _normalize(
        [float("nan"), float("nan")],
        eligible=[False, False],
        window=3,
        minimum=1,
    )

    assert result["normalization_status"].tolist() == [
        "raw_ineligible",
        "raw_ineligible",
    ]
    assert result["reference_count"].tolist() == [0, 0]
    assert result["reference_first_session"].isna().all()
    assert result["reference_last_session"].isna().all()
    assert result["percentile"].isna().all()


def test_percentile_rejects_misaligned_or_invalid_inputs() -> None:
    sessions = _sessions(2)

    with pytest.raises(ValueError, match="equal lengths"):
        normalize_midrank_percentiles([1.0], sessions, [True, True])
    with pytest.raises(ValueError, match="positive integer"):
        normalize_midrank_percentiles(
            [1.0, 2.0],
            sessions,
            [True, True],
            reference_window_sessions=True,
        )
