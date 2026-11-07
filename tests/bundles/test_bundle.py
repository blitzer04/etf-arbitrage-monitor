from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow.parquet as pq
import pytest

from etf_monitor.bundles.bundle import (
    MANIFEST_FILENAME,
    BundleContractError,
    BundleIntegrityError,
    _frame_to_table,
    _table_to_frame,
    list_candidate_runs,
    verify_bundle,
    write_bundle,
)
from etf_monitor.bundles.content_hash import content_sha256, file_sha256
from etf_monitor.bundles.schemas import ARTIFACT_FILENAMES, ARTIFACT_SCHEMAS
from etf_monitor.evaluation import EvaluationResult, evaluate
from tests.prices.helpers import canonical_frame

INSTANT = datetime.fromisoformat("2018-01-02T22:00:00+00:00")


def _result() -> EvaluationResult:
    return evaluate(
        canonical_frame(
            ticker="SPY",
            day="2018-01-02",
            source="FICTIONAL_synthetic",
        ),
        INSTANT,
    )


def _copy_bundle(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    return Path(shutil.copytree(source, destination / source.name))


def _make_writable(path: Path) -> Path:
    """Clear the bundle's read-only bit; Windows refuses to unlink read-only files."""
    path.chmod(0o644)
    return path


def test_write_verify_and_deterministic_content_hashes(tmp_path: Path) -> None:
    result = _result()
    first_path = write_bundle(result, tmp_path / "first")
    second_path = write_bundle(result, tmp_path / "second")

    first = verify_bundle(first_path)
    second = verify_bundle(second_path)

    assert first.run_id == second.run_id
    assert {path.name for path in first_path.iterdir()} == {
        *ARTIFACT_FILENAMES.values(),
        MANIFEST_FILENAME,
    }
    assert set(first.artifacts) == set(ARTIFACT_SCHEMAS)
    first_artifacts = cast(
        Mapping[str, Mapping[str, object]], first.manifest["artifacts"]
    )
    second_artifacts = cast(
        Mapping[str, Mapping[str, object]], second.manifest["artifacts"]
    )
    for name in ARTIFACT_SCHEMAS:
        assert (
            first_artifacts[name]["content_sha256"]
            == second_artifacts[name]["content_sha256"]
        )
    assert first.manifest["data_classification"] == "FICTIONAL"
    assert first.manifest["history_status"] == "current_vintage_reconstruction"
    assert all(path.stat().st_mode & 0o222 == 0 for path in first_path.iterdir())
    with pytest.raises(FileExistsError):
        write_bundle(result, tmp_path / "first")


@pytest.mark.parametrize("artifact_name", tuple(ARTIFACT_FILENAMES))
def test_one_byte_artifact_tampering_is_rejected(
    tmp_path: Path, artifact_name: str
) -> None:
    bundle = write_bundle(_result(), tmp_path / "bundles")
    target = bundle / ARTIFACT_FILENAMES[artifact_name]
    target.chmod(0o644)
    payload = bytearray(target.read_bytes())
    payload[len(payload) // 2] ^= 1
    target.write_bytes(payload)

    with pytest.raises(BundleIntegrityError):
        verify_bundle(bundle)


@pytest.mark.parametrize("mutation", ["missing", "extra", "renamed", "directory"])
def test_bundle_file_set_and_directory_name_are_enforced(
    tmp_path: Path, mutation: str
) -> None:
    original = write_bundle(_result(), tmp_path / "source")
    candidate = _copy_bundle(original, tmp_path / "candidate")
    if mutation == "missing":
        _make_writable(candidate / ARTIFACT_FILENAMES["coverage"]).unlink()
    elif mutation == "extra":
        (candidate / "extra.txt").write_text("unexpected")
    elif mutation == "renamed":
        candidate = candidate.rename(tmp_path / "renamed")
    else:
        artifact = _make_writable(candidate / ARTIFACT_FILENAMES["coverage"])
        artifact.unlink()
        artifact.mkdir()

    with pytest.raises(BundleContractError):
        verify_bundle(candidate)


def test_symlink_in_bundle_is_rejected(tmp_path: Path) -> None:
    original = write_bundle(_result(), tmp_path / "source")
    candidate = _copy_bundle(original, tmp_path / "candidate")
    artifact = _make_writable(candidate / ARTIFACT_FILENAMES["coverage"])
    artifact.unlink()
    try:
        artifact.symlink_to(original / ARTIFACT_FILENAMES["coverage"])
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")

    with pytest.raises(BundleContractError):
        verify_bundle(candidate)


def test_candidate_runs_are_sorted_unverified_direct_child_directories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    (root / "z-candidate").mkdir()
    (root / "a-candidate").mkdir()
    (root / "not-a-directory").write_text("ignored")

    assert list_candidate_runs(root) == ["a-candidate", "z-candidate"]


def test_manifest_row_count_schema_and_duplicate_keys_are_rejected(
    tmp_path: Path,
) -> None:
    original = write_bundle(_result(), tmp_path / "source")
    for mutation in (
        "row_count",
        "schema",
        "duplicate",
        "format_version",
        "unknown_key",
        "missing_key",
    ):
        candidate = _copy_bundle(original, tmp_path / mutation)
        manifest_path = candidate / MANIFEST_FILENAME
        manifest_path.chmod(0o644)
        payload = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(payload)
        if mutation == "row_count":
            manifest["artifacts"]["coverage"]["row_count"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            error_type: type[ValueError] = BundleIntegrityError
        elif mutation == "schema":
            manifest["artifacts"]["coverage"]["schema"][0][1] = "int64"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            error_type = BundleContractError
        elif mutation == "duplicate":
            manifest_path.write_text(
                '{"run_id":"duplicate",' + payload[1:], encoding="utf-8"
            )
            error_type = BundleContractError
        elif mutation == "format_version":
            manifest["format_version"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            error_type = BundleContractError
        elif mutation == "unknown_key":
            manifest["unexpected"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            error_type = BundleContractError
        else:
            del manifest["run_id"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            error_type = BundleContractError
        with pytest.raises(error_type):
            verify_bundle(candidate)


def test_self_consistent_but_wrong_signal_artifact_fails_recomputation(
    tmp_path: Path,
) -> None:
    prices = pd.concat(
        [
            canonical_frame(
                ticker="SPY",
                day="2018-01-02",
                source="FICTIONAL_synthetic",
            ),
            canonical_frame(
                ticker="SPY",
                day="2018-01-03",
                source="FICTIONAL_synthetic",
            ),
        ],
        ignore_index=True,
    )
    result = evaluate(
        prices,
        datetime.fromisoformat("2018-01-03T22:00:00+00:00"),
    )
    bundle = write_bundle(result, tmp_path / "bundles")
    momentum_path = bundle / ARTIFACT_FILENAMES["momentum"]
    momentum_path.chmod(0o644)
    momentum = _table_to_frame(
        pq.read_table(momentum_path), ARTIFACT_SCHEMAS["momentum"]
    )
    earlier_session = pd.Timestamp("2018-01-02")
    non_target_momentum = momentum["ticker"].eq("SPY") & momentum["signal_date"].eq(
        earlier_session
    )
    momentum.loc[non_target_momentum, "status"] = "tampered"
    momentum_table = _frame_to_table(momentum, ARTIFACT_SCHEMAS["momentum"])
    pq.write_table(momentum_table, momentum_path)
    manifest_path = bundle / MANIFEST_FILENAME
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["momentum"]["file_sha256"] = file_sha256(
        momentum_path.read_bytes()
    )
    manifest["artifacts"]["momentum"]["content_sha256"] = content_sha256(
        momentum, ARTIFACT_SCHEMAS["momentum"]
    )
    manifest["artifacts"]["momentum"]["row_count"] = momentum_table.num_rows
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleIntegrityError, match="Recomputed momentum"):
        verify_bundle(bundle)
