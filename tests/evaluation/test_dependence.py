from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from etf_monitor.evaluation.dependence import calculate_dependence
from etf_monitor.universe import load_universe


def test_dependence_constant_input_insufficient_pairs_and_hand_correlations() -> None:
    tickers = [etf.ticker for etf in load_universe().etfs]
    sessions = pd.to_datetime(["2018-01-02", "2018-01-03", "2018-01-04"])
    momentum_rows: list[dict[str, object]] = []
    volatility_rows: list[dict[str, object]] = []
    for position, ticker in enumerate(tickers):
        momentum_rows.append(
            {
                "ticker": ticker,
                "signal_date": sessions[0],
                "percentile": float(position),
            }
        )
        volatility_rows.append(
            {
                "ticker": ticker,
                "signal_date": sessions[0],
                "percentile": 50.0,
            }
        )
    for ticker, value in zip(tickers[:2], (1.0, 2.0), strict=True):
        momentum_rows.append(
            {"ticker": ticker, "signal_date": sessions[1], "percentile": value}
        )
        volatility_rows.append(
            {"ticker": ticker, "signal_date": sessions[1], "percentile": value}
        )
    for ticker, first, second in zip(
        tickers[:4], (1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 2.0, 4.0), strict=True
    ):
        momentum_rows.append(
            {"ticker": ticker, "signal_date": sessions[2], "percentile": first}
        )
        volatility_rows.append(
            {"ticker": ticker, "signal_date": sessions[2], "percentile": second}
        )
    momentum = pd.DataFrame(momentum_rows)
    volatility = pd.DataFrame(volatility_rows)

    cross, _ = calculate_dependence(
        momentum, volatility, load_universe(), pd.DatetimeIndex(sessions)
    )

    constant = cross.iloc[0]
    assert constant["n_pairs"] == 24
    assert constant["status"] == "constant_input"
    assert math.isnan(constant["pearson"])
    assert math.isnan(constant["spearman"])
    assert cross.iloc[1]["n_pairs"] == 2
    assert cross.iloc[1]["status"] == "insufficient_pairs"
    assert math.isnan(cross.iloc[1]["pearson"])
    assert cross.iloc[2]["n_pairs"] == 4
    assert cross.iloc[2]["status"] == "available"
    expected_pearson = 4.5 / math.sqrt(5.0 * 4.75)
    expected_spearman = 4.5 / math.sqrt(5.0 * 4.5)
    assert cross.iloc[2]["pearson"] == pytest.approx(expected_pearson)
    assert cross.iloc[2]["spearman"] == pytest.approx(expected_spearman)


def test_dependence_uses_only_exact_ticker_and_session_pairs() -> None:
    momentum = pd.DataFrame(
        {
            "ticker": pd.Series(["SPY"], dtype="string"),
            "signal_date": pd.to_datetime(["2018-01-02"]),
            "percentile": np.array([10.0], dtype=np.float64),
        }
    )
    volatility = pd.DataFrame(
        {
            "ticker": pd.Series(["SPY"], dtype="string"),
            "signal_date": pd.to_datetime(["2018-01-03"]),
            "percentile": np.array([20.0], dtype=np.float64),
        }
    )

    cross, per_etf = calculate_dependence(
        momentum,
        volatility,
        load_universe(),
        pd.to_datetime(["2018-01-02", "2018-01-03"]),
    )

    assert cross["n_pairs"].tolist() == [0, 0]
    assert set(cross["status"]) == {"insufficient_pairs"}
    assert per_etf.set_index("ticker").loc["SPY", "n_pairs"] == 0
