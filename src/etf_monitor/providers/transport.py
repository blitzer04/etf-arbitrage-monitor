"""Minimal urllib transports for Yahoo charts and issuer NAV-history files."""

from __future__ import annotations

from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ChartTransport(Protocol):
    """Fetch raw chart bytes for one provider symbol and date interval."""

    def fetch(self, symbol: str, period1: int, period2: int) -> bytes:
        """Return the exact response body for one chart request."""
        ...


class ProviderTransportError(RuntimeError):
    """Indicate that an HTTP chart request did not return a valid response."""


class UrllibChartTransport:
    """Fetch Yahoo chart responses with stdlib urllib and no retry policy."""

    _BASE_URL = "https://query2.finance.yahoo.com/v8/finance/chart/"
    # Observed 2026-10-19: the full Chrome user-agent string receives HTTP 429
    # while the minimal token receives 200 for the same request.
    _USER_AGENT = "Mozilla/5.0"
    _TIMEOUT_SECONDS = 30

    def __init__(self) -> None:
        """Create a stateless HTTP transport without opening a connection."""

    def fetch(self, symbol: str, period1: int, period2: int) -> bytes:
        """Make one GET request and return its response body.

        Args:
            symbol: Exact ETF ticker requested.
            period1: Inclusive epoch-second request start.
            period2: Exclusive epoch-second request end.

        Returns:
            Raw response bytes without decoding or transformation.

        Raises:
            ProviderTransportError: If the request fails or returns non-200.
        """
        parameters = urlencode(
            {
                "interval": "1d",
                "includeAdjustedClose": "true",
                "events": "div|split",
                "period1": period1,
                "period2": period2,
            }
        )
        request = Request(
            f"{self._BASE_URL}{symbol}?{parameters}",
            headers={"User-Agent": self._USER_AGENT},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self._TIMEOUT_SECONDS) as response:
                status = response.status
                body = response.read()
        except HTTPError as error:
            raise ProviderTransportError(
                f"Yahoo chart request for {symbol} returned HTTP {error.code}."
            ) from error
        except (URLError, OSError, TimeoutError) as error:
            raise ProviderTransportError(
                f"Yahoo chart request for {symbol} failed: {error}"
            ) from error
        if status != 200:
            raise ProviderTransportError(
                f"Yahoo chart request for {symbol} returned HTTP {status}."
            )
        return cast(bytes, body)


class IssuerFileTransport(Protocol):
    """Fetch one issuer-published file for one ETF ticker."""

    def fetch(self, ticker: str) -> bytes:
        """Return the exact response body for one issuer file request."""
        ...


class UrllibSsgaNavHistoryTransport:
    """Fetch SSGA daily NAV-history workbooks with stdlib urllib, no retries."""

    _URL_TEMPLATE = (
        "https://www.ssga.com/us/en/intermediary/library-content/products/"
        "fund-data/etfs/us/navhist-us-en-{ticker}.xlsx"
    )
    # Observed 2026-11-03: the URL answers HTTP 301 to a canonical
    # library-content path; urllib follows the redirect. The same minimal
    # user-agent token used for Yahoo received HTTP 200.
    _USER_AGENT = "Mozilla/5.0"
    _TIMEOUT_SECONDS = 60
    _LABEL = "NAV-history"

    def __init__(self) -> None:
        """Create a stateless HTTP transport without opening a connection."""

    def fetch(self, ticker: str) -> bytes:
        """Make one GET request and return the workbook bytes unchanged.

        Args:
            ticker: Exact ETF ticker requested; SSGA paths use lower case.

        Returns:
            Raw response bytes without decoding or transformation.

        Raises:
            ProviderTransportError: If the request fails or returns non-200.
        """
        request = Request(
            self._URL_TEMPLATE.format(ticker=ticker.lower()),
            headers={"User-Agent": self._USER_AGENT},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self._TIMEOUT_SECONDS) as response:
                status = response.status
                body = response.read()
        except HTTPError as error:
            raise ProviderTransportError(
                f"SSGA {self._LABEL} request for {ticker} returned HTTP {error.code}."
            ) from error
        except (URLError, OSError, TimeoutError) as error:
            raise ProviderTransportError(
                f"SSGA {self._LABEL} request for {ticker} failed: {error}"
            ) from error
        if status != 200:
            raise ProviderTransportError(
                f"SSGA {self._LABEL} request for {ticker} returned HTTP {status}."
            )
        return cast(bytes, body)


class UrllibSsgaHoldingsTransport(UrllibSsgaNavHistoryTransport):
    """Fetch SSGA daily holdings workbooks (latest snapshot only), no retries."""

    # Observed 2026-11-03: same redirect and user-agent behavior as the
    # NAV-history files; the response is the latest published snapshot.
    _URL_TEMPLATE = (
        "https://www.ssga.com/us/en/intermediary/library-content/products/"
        "fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx"
    )
    _LABEL = "holdings"
