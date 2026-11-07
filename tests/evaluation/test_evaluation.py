from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from etf_monitor.demo import build_demo_prices
from etf_monitor.evaluation import evaluate
from etf_monitor.prices.schema import PriceValidationError
from etf_monitor.universe import load_universe
from etf_monitor.xnys import resolve_target_session
from tests.prices.helpers import canonical_frame


def _instant(day: str) -> datetime:
    return datetime.fromisoformat(f"{day}T22:00:00+00:00")


def _small_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    rows = [
        canonical_frame(
            ticker=ticker,
            day="2018-01-02",
            source="FICTIONAL_synthetic",
        )
        for ticker in tickers
    ]
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["ticker", "date"], kind="mergesort")
        .reset_index(drop=True)
    )


def test_zero_row_ticker_remains_in_coverage_and_does_not_affect_others() -> None:
    tickers = tuple(etf.ticker for etf in load_universe().etfs)
    prices = _small_prices(tickers[:-1])

    result = evaluate(prices, _instant("2018-01-02"))

    assert result.coverage["ticker"].tolist() == list(tickers)
    assert len(result.coverage) == 24
    missing = result.coverage.set_index("ticker").loc[tickers[-1]]
    assert missing["present_row_count"] == 0
    assert missing["missing_row_count"] == 1
    assert missing["missing_row_dates"] == [pd.Timestamp("2018-01-02").date()]
    assert missing["missing_adjusted_close_count"] == 0
    assert pd.isna(missing["first_row_date"])
    assert pd.isna(missing["last_row_date"])
    assert pd.isna(missing["last_eligible_adjusted_close_date"])
    assert pd.isna(missing["price_staleness_sessions"])
    assert not missing["target_row_present"]
    assert not missing["target_adjusted_close_present"]
    assert pd.isna(missing["retrieved_at_min"])
    assert pd.isna(missing["retrieved_at_max"])
    assert pd.isna(missing["momentum_target_raw"])
    present = result.coverage.set_index("ticker").loc[tickers[0]]
    assert present["present_row_count"] == 1
    assert present["target_row_present"]


def test_post_target_rows_are_counted_but_do_not_change_evaluation_results() -> None:
    prices = _small_prices(("SPY",))
    future = canonical_frame(
        ticker="SPY",
        day="2018-01-03",
        adjusted_close=110.0,
        source="FICTIONAL_synthetic",
    )
    with_future = (
        pd.concat([prices, future], ignore_index=True)
        .sort_values(["ticker", "date"], kind="mergesort")
        .reset_index(drop=True)
    )

    result_with_future = evaluate(with_future, _instant("2018-01-02"))
    result_truncated = evaluate(prices, _instant("2018-01-02"))

    spy = result_with_future.coverage.set_index("ticker").loc["SPY"]
    assert spy["rows_after_target"] == 1
    assert (
        result_truncated.coverage.set_index("ticker").loc["SPY", "rows_after_target"]
        == 0
    )
    pd.testing.assert_frame_equal(
        result_with_future.momentum, result_truncated.momentum
    )
    pd.testing.assert_frame_equal(
        result_with_future.volatility, result_truncated.volatility
    )
    pd.testing.assert_frame_equal(
        result_with_future.dependence_cross_section,
        result_truncated.dependence_cross_section,
    )
    pd.testing.assert_frame_equal(
        result_with_future.dependence_per_etf,
        result_truncated.dependence_per_etf,
    )


def test_incident_8_missing_target_close_never_rolls_back_values() -> None:
    instant = _instant("2020-01-15")
    prices = build_demo_prices(instant)
    target = resolve_target_session(instant)
    target_mask = prices["date"].eq(target)
    assert target_mask.sum() == 24
    prices.loc[target_mask, "adjusted_close"] = float("nan")

    result = evaluate(prices, instant)

    coverage = result.coverage.set_index("ticker")
    target_momentum = result.momentum.loc[result.momentum["signal_date"].eq(target)]
    target_volatility = result.volatility.loc[
        result.volatility["signal_date"].eq(target)
    ]
    assert len(target_momentum) == len(target_volatility) == 24
    assert target_momentum["raw_log_return"].isna().all()
    assert target_momentum["percentile"].isna().all()
    assert target_momentum["status"].eq("missing_end_adjusted_close").all()
    assert target_volatility["raw_annualized_vol"].isna().all()
    assert target_volatility["percentile"].isna().all()
    assert target_volatility["status"].eq("missing_adjusted_close_in_window").all()
    for etf in load_universe().etfs:
        row = coverage.loc[etf.ticker]
        assert row["target_row_present"]
        assert not row["target_adjusted_close_present"]
        assert row["missing_adjusted_close_count"] == 1
        assert row["price_staleness_sessions"] == 1
        assert (
            row["last_eligible_adjusted_close_date"]
            == result.momentum["signal_date"].iloc[-2]
        )
        assert row["momentum_target_raw"] != row["momentum_target_raw"]
        assert row["momentum_target_percentile"] != row["momentum_target_percentile"]
        assert row["volatility_target_raw"] != row["volatility_target_raw"]
        assert (
            row["volatility_target_percentile"] != row["volatility_target_percentile"]
        )
        assert row["momentum_raw_staleness_sessions"] == 1
        assert row["volatility_raw_staleness_sessions"] == 1
        assert row["momentum_normalized_staleness_sessions"] == 1
        assert row["volatility_normalized_staleness_sessions"] == 1
        assert row["momentum_target_status"] == "missing_end_adjusted_close"
        assert row["volatility_target_status"] == "missing_adjusted_close_in_window"
        assert row[
            "momentum_last_raw_eligible_date"
        ] == result.target_session - pd.offsets.BDay(1)
        assert row[
            "volatility_last_raw_eligible_date"
        ] == result.target_session - pd.offsets.BDay(1)


def test_mixed_sources_are_refused() -> None:
    prices = _small_prices(("SPY", "QQQ"))
    prices.loc[prices["ticker"].eq("QQQ"), "source"] = "yahoo_chart_v8"

    with pytest.raises(PriceValidationError, match="single source"):
        evaluate(prices, _instant("2018-01-02"))


def test_single_provider_source_is_classified_as_observed_provider_data() -> None:
    prices = _small_prices(("SPY",))
    prices["source"] = pd.Series(["yahoo_chart_v8"], dtype="string")

    result = evaluate(prices, _instant("2018-01-02"))

    assert result.data_classification == "OBSERVED_PROVIDER_DATA"
