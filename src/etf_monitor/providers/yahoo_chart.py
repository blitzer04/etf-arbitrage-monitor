"""Pure offline parsing and identity validation for Yahoo v8 chart payloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import NoReturn, cast
from zoneinfo import ZoneInfo

from etf_monitor.numeric import NumericConversionError, price_scalar, volume_scalar
from etf_monitor.xnys import is_session

_NEW_YORK = ZoneInfo("America/New_York")
_QUOTE_FIELDS = ("open", "high", "low", "close", "volume")


class ChartPayloadError(ValueError):
    """Indicate that a provider response violates the Yahoo chart contract."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class ParsedObservation:
    """One non-empty provider observation, before retrieval provenance is added."""

    date: date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    adjusted_close: float | None
    volume: int | None


@dataclass(frozen=True, slots=True)
class ParsedChart:
    """Validated chart observations and provider-missingness metadata."""

    observations: tuple[ParsedObservation, ...]
    returned_dates: tuple[date, ...]
    null_row_dates: tuple[date, ...]
    adjclose_block_absent: bool
    empty: bool


def _reject_json_constant(token: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant {token!r}")


def _fail(reason_code: str, message: str) -> NoReturn:
    raise ChartPayloadError(reason_code, message)


def _read_price(value: object, field: str) -> float | None:
    try:
        return price_scalar(value)
    except NumericConversionError as error:
        raise ChartPayloadError(
            error.reason_code, f"Invalid {field} value: {error}."
        ) from error


def _read_volume(value: object) -> int | None:
    try:
        return volume_scalar(value)
    except NumericConversionError as error:
        raise ChartPayloadError(
            error.reason_code, f"Invalid volume value: {error}."
        ) from error


def parse_chart_payload(
    raw: bytes, requested: str, start: date, end_exclusive: date
) -> ParsedChart:
    """Parse one raw Yahoo chart response without network or provider state.

    Yahoo epoch seconds are converted to the America/New_York local calendar
    date. A response is accepted only when every observation is inside the
    requested half-open window and on an XNYS session.

    Args:
        raw: Exact provider response bytes.
        requested: Exact requested ETF ticker.
        start: Inclusive local-market request date.
        end_exclusive: Exclusive local-market request date.

    Returns:
        Parsed rows, null-only row dates, and adjusted-close block metadata.

    Raises:
        ChartPayloadError: If the response identity, shape, scalars, or dates
            violate the contract.
    """
    if start >= end_exclusive:
        raise ValueError("start must be earlier than end_exclusive.")
    try:
        payload = json.loads(raw, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ChartPayloadError(
            "malformed_payload", f"Yahoo response is not strict JSON: {error}."
        ) from error

    if not isinstance(payload, dict) or not isinstance(payload.get("chart"), dict):
        _fail("malformed_payload", "Yahoo response is missing the chart object.")
    chart = payload["chart"]
    if chart.get("error") is not None:
        _fail("provider_error", "Yahoo chart response contains a provider error.")
    results = chart.get("result")
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
    ):
        _fail(
            "malformed_payload",
            "Yahoo chart result must contain exactly one result object.",
        )

    result = results[0]
    meta = result.get("meta")
    if not isinstance(meta, dict) or not isinstance(meta.get("symbol"), str):
        _fail("identity_missing", "Yahoo chart result is missing meta.symbol.")
    if meta["symbol"] != requested:
        _fail(
            "identity_mismatch",
            f"Yahoo returned {meta['symbol']!r}, not requested {requested!r}.",
        )
    if meta.get("exchangeTimezoneName") != "America/New_York":
        _fail(
            "unexpected_timezone",
            "Yahoo chart exchangeTimezoneName must be America/New_York.",
        )

    if "timestamp" not in result:
        return ParsedChart((), (), (), False, True)
    timestamps = result["timestamp"]
    if isinstance(timestamps, list) and not timestamps:
        return ParsedChart((), (), (), False, True)
    if not isinstance(timestamps, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in timestamps
    ):
        _fail("malformed_payload", "Yahoo timestamps must be a list of integer epochs.")

    indicators = result.get("indicators")
    if not isinstance(indicators, dict):
        _fail("malformed_payload", "Yahoo chart result is missing indicators.")
    quote_blocks = indicators.get("quote")
    if (
        not isinstance(quote_blocks, list)
        or len(quote_blocks) != 1
        or not isinstance(quote_blocks[0], dict)
    ):
        _fail("malformed_payload", "Yahoo chart must contain one quote object.")
    quote = cast(dict[str, object], quote_blocks[0])
    quote_values: dict[str, list[object]] = {}
    for field in _QUOTE_FIELDS:
        field_values = quote.get(field)
        if not isinstance(field_values, list) or len(field_values) != len(timestamps):
            _fail(
                "malformed_payload",
                f"Yahoo quote field {field!r} must match timestamp length.",
            )
        quote_values[field] = field_values

    adjclose_block_absent = "adjclose" not in indicators
    adjusted_values: list[object] = [None] * len(timestamps)
    if not adjclose_block_absent:
        adjusted_blocks = indicators["adjclose"]
        if (
            not isinstance(adjusted_blocks, list)
            or len(adjusted_blocks) != 1
            or not isinstance(adjusted_blocks[0], dict)
        ):
            _fail(
                "malformed_payload",
                "Yahoo adjusted-close block must contain one object.",
            )
        adjusted_block = cast(dict[str, object], adjusted_blocks[0])
        adjusted_values_candidate = adjusted_block.get("adjclose")
        if not isinstance(adjusted_values_candidate, list) or len(
            adjusted_values_candidate
        ) != len(timestamps):
            _fail(
                "malformed_payload",
                "Yahoo adjusted-close values must match timestamp length.",
            )
        adjusted_values = adjusted_values_candidate

    observations: list[ParsedObservation] = []
    null_row_dates: list[date] = []
    seen_dates: set[date] = set()
    for position, epoch in enumerate(timestamps):
        try:
            local_date = datetime.fromtimestamp(epoch, UTC).astimezone(_NEW_YORK).date()
        except (OverflowError, OSError, ValueError) as error:
            raise ChartPayloadError(
                "malformed_payload", "Yahoo timestamp is outside supported range."
            ) from error
        if local_date in seen_dates:
            _fail(
                "duplicate_date",
                f"Yahoo response contains duplicate local date {local_date}.",
            )
        seen_dates.add(local_date)
        if local_date < start or local_date >= end_exclusive:
            _fail(
                "date_outside_request",
                f"Yahoo returned {local_date} outside [{start}, {end_exclusive}).",
            )
        if not is_session(local_date):
            _fail(
                "non_session_date",
                f"Yahoo returned non-session date {local_date}.",
            )

        normalized_values: dict[str, float | None] = {
            field: _read_price(quote_values[field][position], field)
            for field in ("open", "high", "low", "close")
        }
        volume = _read_volume(quote_values["volume"][position])
        adjusted_close = _read_price(adjusted_values[position], "adjusted_close")
        row_values = (*normalized_values.values(), volume, adjusted_close)
        if all(value is None for value in row_values):
            null_row_dates.append(local_date)
            continue
        observations.append(
            ParsedObservation(
                date=local_date,
                open=normalized_values["open"],
                high=normalized_values["high"],
                low=normalized_values["low"],
                close=normalized_values["close"],
                adjusted_close=adjusted_close,
                volume=volume,
            )
        )

    observations.sort(key=lambda observation: observation.date)
    null_row_dates.sort()
    returned_dates = tuple(observation.date for observation in observations)
    return ParsedChart(
        observations=tuple(observations),
        returned_dates=returned_dates,
        null_row_dates=tuple(null_row_dates),
        adjclose_block_absent=adjclose_block_absent,
        empty=False,
    )
