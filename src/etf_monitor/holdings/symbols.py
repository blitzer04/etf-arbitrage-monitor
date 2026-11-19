"""Ticker normalization for issuer holdings and Yahoo chart requests."""

from __future__ import annotations

import re

_ISSUER_TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


def yahoo_symbol(security_ticker: str | None) -> str | None:
    """Map an issuer security ticker to a Yahoo symbol, or None if unmappable."""
    if security_ticker is None or _ISSUER_TICKER.fullmatch(security_ticker) is None:
        return None
    return security_ticker.replace(".", "-")
