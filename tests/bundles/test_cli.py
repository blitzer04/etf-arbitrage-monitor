from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, tzinfo
from datetime import datetime as RealDatetime
from pathlib import Path

import pytest

from etf_monitor import cli
from etf_monitor.prices.store import _atomic_parquet_write
from etf_monitor.universe import load_universe
from tests.prices.helpers import canonical_frame

INSTANT = RealDatetime.fromisoformat("2018-01-02T22:00:00+00:00")


def _write_input(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet_write(
        canonical_frame(
            ticker="SPY",
            day="2018-01-02",
            source="FICTIONAL_synthetic",
        ),
        path,
        load_universe(),
    )


def _run_cli(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "etf_monitor", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_evaluate_verify_and_compare_from_temporary_cwd(tmp_path: Path) -> None:
    input_path = tmp_path / "inputs" / "prices.parquet"
    output_root = tmp_path / "bundles"
    _write_input(input_path)
    evaluation = _run_cli(
        tmp_path,
        "evaluate",
        "--prices",
        str(input_path),
        "--instant",
        "2018-01-02T22:00:00Z",
        "--out",
        str(output_root),
    )
    assert evaluation.returncode == 0, evaluation.stderr
    bundle_path = Path(evaluation.stdout.strip().splitlines()[-1])
    verified = _run_cli(tmp_path, "verify", str(bundle_path))
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout.strip().startswith("verified ")
    compared = _run_cli(
        tmp_path,
        "compare",
        str(bundle_path),
        str(bundle_path),
    )
    assert compared.returncode == 0, compared.stderr
    comparison = json.loads(compared.stdout)
    assert comparison["input_row_counts"][0]["rows_only_in_a"] == 0


def test_cli_now_captures_clock_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "prices.parquet"
    _write_input(input_path)
    calls = 0

    class FixedDatetime(RealDatetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> FixedDatetime:
            nonlocal calls
            calls += 1
            assert tz is UTC
            return cls(
                INSTANT.year,
                INSTANT.month,
                INSTANT.day,
                INSTANT.hour,
                INSTANT.minute,
                INSTANT.second,
                INSTANT.microsecond,
                tzinfo=tz,
            )

    monkeypatch.setattr(cli, "datetime", FixedDatetime)
    result = cli.main(
        [
            "evaluate",
            "--prices",
            str(input_path),
            "--instant",
            "now",
            "--out",
            str(tmp_path / "bundles"),
        ]
    )

    assert result == 0
    assert calls == 1


def test_full_history_demo_bundle_round_trips_through_cli_verification(
    tmp_path: Path,
) -> None:
    instant = "2021-01-04T22:00:00Z"
    generated = _run_cli(
        tmp_path,
        "demo-bundle",
        "--instant",
        instant,
        "--out",
        str(tmp_path / "demo"),
    )
    assert generated.returncode == 0, generated.stderr
    bundle_path = Path(generated.stdout.strip().splitlines()[-1])
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_classification"] == "FICTIONAL"
    assert manifest["sources"] == ["FICTIONAL_synthetic"]
    verified = _run_cli(tmp_path, "verify", str(bundle_path))
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout.strip().startswith("verified ")
