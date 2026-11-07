"""Long-series differential, isolation, and no-look-ahead regressions."""

from __future__ import annotations

import random
from datetime import date

import numpy as np
import pandas as pd

from etf_monitor.signals.momentum import calculate_momentum
from etf_monitor.signals.volatility import calculate_volatility
from etf_monitor.xnys import xnys_sessions
from tests.independent_signals import calculate_independent_signals
from tests.signals.helpers import price_frame


def _sessions(count: int) -> pd.DatetimeIndex:
    return xnys_sessions(date(2018, 1, 2), date(2025, 12, 31))[:count]


def _long_fictional_prices() -> tuple[pd.DatetimeIndex, pd.DataFrame, list[str]]:
    sessions = _sessions(1100)
    tickers = ["SPY", "QQQ", "IWM"]
    randomizer = random.Random(20261019)
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        exponent = 0
        adjusted: list[float | None] = []
        missing_rows: set[int] = set()
        for position in range(len(sessions)):
            exponent = max(
                -4,
                min(4, exponent + randomizer.choice([-2, -1, 0, 1, 2])),
            )
            value: float | None = float(2.0**exponent)
            if position > 20 and randomizer.random() < 0.012:
                missing_rows.add(position)
            elif position > 20 and randomizer.random() < 0.012:
                value = None
            adjusted.append(value)
        frames.append(
            price_frame(
                ticker,
                sessions,
                adjusted,
                missing_rows=missing_rows,
            )
        )
    return (
        sessions,
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"], kind="mergesort")
        .reset_index(drop=True),
        tickers,
    )


def _package_results(
    prices: pd.DataFrame,
    tickers: list[str],
    target: pd.Timestamp,
    grid_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum = pd.concat(
        [
            calculate_momentum(prices, ticker, target, grid_start=grid_start)
            for ticker in tickers
        ],
        ignore_index=True,
    )
    volatility = pd.concat(
        [
            calculate_volatility(prices, ticker, target, grid_start=grid_start)
            for ticker in tickers
        ],
        ignore_index=True,
    )
    return momentum, volatility


def _package_raw_midrank_percentiles(
    output: pd.DataFrame,
    raw_column: str,
) -> dict[tuple[str, pd.Timestamp], float]:
    percentiles: dict[tuple[str, pd.Timestamp], float] = {}
    for ticker, ticker_output in output.groupby("ticker", sort=False):
        ticker_output = ticker_output.reset_index(drop=True)
        raw_values = ticker_output[raw_column].to_numpy(dtype=np.float64)
        eligible = ticker_output["status"].eq("eligible").to_numpy()
        for position, row in enumerate(ticker_output.to_dict(orient="records")):
            first_reference_position = max(0, position - 755)
            references = [
                raw_values[reference_position]
                for reference_position in range(first_reference_position, position)
                if eligible[reference_position]
            ]
            if not eligible[position] or len(references) < 252:
                value = float("nan")
            else:
                lower_count = sum(
                    reference < raw_values[position] for reference in references
                )
                equal_count = sum(
                    reference == raw_values[position] for reference in references
                )
                value = 100.0 * (lower_count + 0.5 * equal_count) / len(references)
            percentiles[(str(ticker), pd.Timestamp(row["signal_date"]))] = value
    return percentiles


def test_seeded_1100_session_three_etf_reference_matches() -> None:
    sessions, prices, tickers = _long_fictional_prices()
    expected_momentum, expected_volatility = calculate_independent_signals(
        prices,
        sessions,
        tickers,
    )
    momentum, volatility = _package_results(prices, tickers, sessions[-1], sessions[0])

    for output in momentum.to_dict(orient="records"):
        key = (str(output["ticker"]), pd.Timestamp(output["signal_date"]))
        expected = expected_momentum[key]
        raw_log_return = float(output["raw_log_return"])
        if expected["raw_log_return"] is None:
            assert np.isnan(raw_log_return)
        else:
            assert raw_log_return == expected["raw_log_return"]
        if expected["percentile"] is None:
            assert np.isnan(float(output["percentile"]))
        else:
            assert float(output["percentile"]) == expected["percentile"]
        assert output["status"] == expected["status"]
        assert output["reference_count"] == expected["reference_count"]

    package_raw_volatility_percentiles = _package_raw_midrank_percentiles(
        volatility, "raw_annualized_vol"
    )
    for output in volatility.to_dict(orient="records"):
        key = (str(output["ticker"]), pd.Timestamp(output["signal_date"]))
        expected = expected_volatility[key]
        if expected["raw_annualized_vol"] is None:
            assert np.isnan(float(output["raw_annualized_vol"]))
        else:
            np.testing.assert_allclose(
                float(output["raw_annualized_vol"]),
                expected["raw_annualized_vol"],
                rtol=1e-12,
                atol=0.0,
            )
        package_percentile = package_raw_volatility_percentiles[key]
        if np.isnan(package_percentile):
            assert np.isnan(float(output["percentile"]))
        else:
            assert float(output["percentile"]) == package_percentile
        assert output["status"] == expected["status"]
        assert output["reference_count"] == expected["reference_count"]
    assert not any(
        any(
            disallowed in column.lower()
            for disallowed in ("composite", "score", "label")
        )
        for column in [*momentum.columns, *volatility.columns]
    )


def test_signals_are_ticker_isolated_and_ignore_appended_future_prices() -> None:
    sessions = _sessions(40)
    tickers = ["SPY", "QQQ", "IWM"]
    frames = [
        price_frame(
            ticker,
            sessions,
            [float(2.0 ** (position + ticker_index)) for position in range(40)],
        )
        for ticker_index, ticker in enumerate(tickers)
    ]
    prices = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
    target_position = 30
    target = sessions[target_position]
    historical = prices.loc[prices["date"].le(target)].copy().reset_index(drop=True)
    future_changed = prices.copy()
    future_mask = future_changed["date"].gt(target)
    future_changed.loc[future_mask, "adjusted_close"] *= 4.0
    changed_other_ticker = prices.copy()
    qqq_mask = changed_other_ticker["ticker"].eq("QQQ")
    changed_other_ticker.loc[qqq_mask, "adjusted_close"] *= 3.0

    def signals(
        frame: pd.DataFrame, end: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (
            calculate_momentum(
                frame,
                "SPY",
                end,
                grid_start=sessions[0],
                horizon_sessions=3,
                reference_window_sessions=5,
                minimum_reference_observations=2,
            ),
            calculate_volatility(
                frame,
                "SPY",
                end,
                grid_start=sessions[0],
                return_count=3,
                reference_window_sessions=5,
                minimum_reference_observations=2,
            ),
        )

    baseline_momentum, baseline_volatility = signals(historical, target)
    isolated_momentum, isolated_volatility = signals(
        changed_other_ticker.loc[changed_other_ticker["date"].le(target)].reset_index(
            drop=True
        ),
        target,
    )
    pd.testing.assert_frame_equal(baseline_momentum, isolated_momentum)
    pd.testing.assert_frame_equal(baseline_volatility, isolated_volatility)

    extended_momentum, extended_volatility = signals(future_changed, sessions[-1])
    pd.testing.assert_frame_equal(
        baseline_momentum,
        extended_momentum.loc[extended_momentum["signal_date"].le(target)].reset_index(
            drop=True
        ),
    )
    pd.testing.assert_frame_equal(
        baseline_volatility,
        extended_volatility.loc[
            extended_volatility["signal_date"].le(target)
        ].reset_index(drop=True),
    )
