from __future__ import annotations

from datetime import datetime
from pathlib import Path

from etf_monitor.bundles.bundle import write_bundle
from etf_monitor.evaluation import evaluate
from tests.prices.helpers import canonical_frame


def write_fictional_bundle(root: Path, day: str) -> Path:
    """Write a small fictional bundle ending on one XNYS session."""
    instant = datetime.fromisoformat(f"{day}T22:00:00+00:00")
    prices = canonical_frame(
        ticker="SPY",
        day=day,
        source="FICTIONAL_synthetic",
    )
    return write_bundle(evaluate(prices, instant), root)
