"""Adversarial offline tests for raw Yahoo chart parsing."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time

import pytest

from etf_monitor.providers.yahoo_chart import (
    ChartPayloadError,
    parse_chart_payload,
)
from tests.fixtures.yahoo_chart import build_yahoo_chart_payload


def _raw(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _payload(**kwargs: object) -> dict[str, object]:
    return build_yahoo_chart_payload(**kwargs)  # type: ignore[arg-type,return-value]


def test_1600_local_timestamp_maps_to_the_same_session_as_0930() -> None:
    morning = parse_chart_payload(
        _raw(_payload(dates=("2024-07-03",))),
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 5),
    )
    afternoon = parse_chart_payload(
        _raw(_payload(dates=("2024-07-03",), timestamp_time=time(16))),
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 5),
    )

    assert morning.returned_dates == (date(2024, 7, 3),)
    assert afternoon.returned_dates == morning.returned_dates


@pytest.mark.parametrize("day", ["2024-07-04", "2024-07-06"])
def test_non_session_local_date_is_rejected(day: str) -> None:
    with pytest.raises(ChartPayloadError, match="non-session") as error:
        parse_chart_payload(
            _raw(_payload(dates=(day,))),
            "SPY",
            date(2024, 7, 1),
            date(2024, 7, 10),
        )
    assert error.value.reason_code == "non_session_date"


def test_early_close_session_is_accepted() -> None:
    parsed = parse_chart_payload(
        _raw(_payload(dates=("2024-07-03",))),
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 5),
    )
    assert parsed.returned_dates == (date(2024, 7, 3),)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("symbol_mismatch", "identity_mismatch"),
        ("symbol_missing", "identity_missing"),
        ("meta_missing", "identity_missing"),
        ("timezone", "unexpected_timezone"),
        ("wrong_length", "malformed_payload"),
        ("non_list", "malformed_payload"),
        ("nested", "non_numeric_value"),
        ("two_results", "malformed_payload"),
        ("empty_result", "malformed_payload"),
        ("provider_error", "provider_error"),
    ],
)
def test_identity_and_shape_violations_fail(mutation: str, reason: str) -> None:
    payload = _payload()
    result = payload["chart"]["result"][0]  # type: ignore[index]
    if mutation == "symbol_mismatch":
        result["meta"]["symbol"] = "QQQ"  # type: ignore[index]
    elif mutation == "symbol_missing":
        del result["meta"]["symbol"]  # type: ignore[index]
    elif mutation == "meta_missing":
        del result["meta"]  # type: ignore[arg-type]
    elif mutation == "timezone":
        result["meta"]["exchangeTimezoneName"] = "America/Chicago"  # type: ignore[index]
    elif mutation == "wrong_length":
        result["indicators"]["quote"][0]["open"].pop()  # type: ignore[index]
    elif mutation == "non_list":
        result["indicators"]["quote"][0]["open"] = None  # type: ignore[index]
    elif mutation == "nested":
        result["indicators"]["quote"][0]["open"][0] = [100]  # type: ignore[index]
    elif mutation == "two_results":
        payload["chart"]["result"].append(result)  # type: ignore[index]
    elif mutation == "empty_result":
        payload = _payload(empty_result=True)
    else:
        payload = _payload(provider_error={"code": "fictional"})

    with pytest.raises(ChartPayloadError) as error:
        parse_chart_payload(_raw(payload), "SPY", date(2024, 1, 1), date(2024, 2, 1))
    assert error.value.reason_code == reason


def test_missing_timestamps_are_genuinely_empty() -> None:
    payload = _payload()
    del payload["chart"]["result"][0]["timestamp"]  # type: ignore[index]
    parsed = parse_chart_payload(
        _raw(payload), "SPY", date(2024, 1, 1), date(2024, 2, 1)
    )
    assert parsed.empty


def test_end_exclusive_and_start_bounds_are_enforced() -> None:
    for day, start, end in (
        ("2024-07-05", date(2024, 7, 1), date(2024, 7, 5)),
        ("2024-06-28", date(2024, 7, 1), date(2024, 7, 5)),
    ):
        with pytest.raises(ChartPayloadError) as error:
            parse_chart_payload(_raw(_payload(dates=(day,))), "SPY", start, end)
        assert error.value.reason_code == "date_outside_request"


def test_late_utc_epoch_uses_new_york_local_date() -> None:
    parsed = parse_chart_payload(
        _raw(_payload(dates=("2024-07-02",), timestamp_time=time(23, 30))),
        "SPY",
        date(2024, 7, 1),
        date(2024, 7, 5),
    )
    assert parsed.returned_dates == (date(2024, 7, 2),)


@pytest.mark.parametrize("same_values", [True, False])
def test_duplicate_local_dates_fail(same_values: bool) -> None:
    payload = _payload(dates=("2024-07-02", "2024-07-02"))
    if not same_values:
        payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][1] = 95  # type: ignore[index]
    with pytest.raises(ChartPayloadError) as error:
        parse_chart_payload(_raw(payload), "SPY", date(2024, 7, 1), date(2024, 7, 5))
    assert error.value.reason_code == "duplicate_date"


def test_absent_adjclose_is_never_copied_from_close() -> None:
    parsed = parse_chart_payload(
        _raw(_payload(include_adjusted_close=False)),
        "SPY",
        date(2024, 1, 1),
        date(2024, 2, 1),
    )
    assert parsed.adjclose_block_absent
    assert parsed.observations[0].close == 101.0
    assert parsed.observations[0].adjusted_close is None


def test_all_null_row_is_recorded_but_not_returned() -> None:
    parsed = parse_chart_payload(
        _raw(
            _payload(
                dates=("2024-01-02",),
                null_quote_positions={
                    "open": (0,),
                    "high": (0,),
                    "low": (0,),
                    "close": (0,),
                    "volume": (0,),
                },
                null_adjusted_positions=(0,),
            )
        ),
        "SPY",
        date(2024, 1, 1),
        date(2024, 2, 1),
    )
    assert parsed.returned_dates == ()
    assert parsed.null_row_dates == (date(2024, 1, 2),)


@pytest.mark.parametrize(
    ("field", "values", "reason"),
    [
        ("close", (0,), "invalid_observation"),
        ("close", (True,), "non_numeric_value"),
        ("close", ("100",), "non_numeric_value"),
        ("close", (2**53 + 1,), "non_numeric_value"),
        ("volume", (-1,), "volume_out_of_range"),
        ("volume", (1.5,), "non_integral_volume"),
        ("volume", (float(2**53 + 2),), "ambiguous_float_volume"),
        ("volume", (2**63 + 1,), "volume_out_of_range"),
        ("volume", (True,), "non_numeric_value"),
        ("volume", ("100",), "non_numeric_value"),
    ],
)
def test_numeric_scalar_boundaries(
    field: str, values: tuple[object, ...], reason: str
) -> None:
    quote_values = {field: values}
    payload = _payload(dates=("2024-01-02",), quote_values=quote_values)
    if reason == "invalid_observation":
        from etf_monitor.prices.acquire import fetch_price_batch
        from etf_monitor.universe import load_universe

        class OneTicker:
            def fetch(self, symbol: str, period1: int, period2: int) -> bytes:
                if symbol == "SPY":
                    return _raw(payload)
                return _raw(_payload(ticker=symbol, dates=()))

        batch = fetch_price_batch(
            load_universe(),
            date(2024, 1, 1),
            date(2024, 2, 1),
            OneTicker(),
            lambda: datetime(2024, 1, 3, tzinfo=UTC),
        )
        assert batch.statuses[0].reason == reason
        return
    with pytest.raises(ChartPayloadError) as error:
        parse_chart_payload(_raw(payload), "SPY", date(2024, 1, 1), date(2024, 2, 1))
    assert error.value.reason_code == reason


@pytest.mark.parametrize("token", [b"NaN", b"Infinity", b"-Infinity"])
def test_nonstandard_json_numeric_constants_are_rejected(token: bytes) -> None:
    raw = _raw(_payload()).replace(b"101.0", token, 1)
    with pytest.raises(ChartPayloadError) as error:
        parse_chart_payload(raw, "SPY", date(2024, 1, 1), date(2024, 2, 1))
    assert error.value.reason_code == "malformed_payload"


def test_integer_price_and_volume_precision_limits() -> None:
    payload = _payload(
        dates=("2024-01-02",),
        quote_values={
            "open": (2**53,),
            "high": (2**53 + 2,),
            "low": (2**53 - 2,),
            "close": (2**53,),
            "volume": (2**53 + 1,),
        },
    )
    parsed = parse_chart_payload(
        _raw(payload), "SPY", date(2024, 1, 1), date(2024, 2, 1)
    )
    row = parsed.observations[0]
    assert row.open == float(2**53)
    assert row.volume == 2**53 + 1


def test_null_price_is_preserved_when_other_market_values_are_present() -> None:
    parsed = parse_chart_payload(
        _raw(_payload(dates=("2024-01-02",), quote_values={"open": (None,)})),
        "SPY",
        date(2024, 1, 1),
        date(2024, 2, 1),
    )

    assert parsed.observations[0].open is None
    assert parsed.observations[0].close == 101.0


def test_maximum_int64_volume_is_accepted() -> None:
    parsed = parse_chart_payload(
        _raw(
            _payload(
                dates=("2024-01-02",),
                quote_values={"volume": (2**63 - 1,)},
            )
        ),
        "SPY",
        date(2024, 1, 1),
        date(2024, 2, 1),
    )
    assert parsed.observations[0].volume == 2**63 - 1
