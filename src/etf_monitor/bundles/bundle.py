"""Atomic bundle publication and one-read integrity verification."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from etf_monitor import __version__
from etf_monitor.bundles.content_hash import content_sha256, file_sha256
from etf_monitor.bundles.schemas import (
    ARTIFACT_FILENAMES,
    ARTIFACT_SCHEMAS,
    ArtifactSchema,
)
from etf_monitor.evaluation import EvaluationResult, evaluate
from etf_monitor.methodology import load_methodology
from etf_monitor.prices.schema import ALLOWED_SOURCES, canonicalize_frame
from etf_monitor.universe import load_universe
from etf_monitor.xnys import xnys_sessions

FORMAT_VERSION = 1
HISTORY_STATUS = "current_vintage_reconstruction"
LIMITATIONS = (
    "Unsigned manifest: integrity is not authenticity.",
    "Current-vintage reconstruction; historical provider vintages are not claimed.",
    "Flow and Concentration diagnostics are deferred.",
    "No composite score or label is produced.",
)
MANIFEST_FILENAME = "manifest.json"
_ARTIFACT_NAMES = tuple(ARTIFACT_SCHEMAS)
_EXPECTED_FILENAMES = set(ARTIFACT_FILENAMES.values()) | {MANIFEST_FILENAME}


class BundleIntegrityError(ValueError):
    """Indicate byte, row, or recomputation integrity failure."""


class BundleContractError(ValueError):
    """Indicate that a bundle violates the required file or schema contract."""


class BundleIncompatibleError(ValueError):
    """Indicate that a bundle uses a different current universe or methodology."""


@dataclass(frozen=True, slots=True)
class VerifiedBundle:
    """Hold a verified bundle and immutable Arrow artifact tables."""

    path: Path
    run_id: str
    manifest: Mapping[str, object]
    artifacts: Mapping[str, pa.Table]

    def frame(self, name: str) -> pd.DataFrame:
        """Return a fresh pandas copy of one verified artifact.

        Args:
            name: Artifact name, such as ``"coverage"`` or ``"momentum"``.

        Returns:
            A new DataFrame with the artifact's contract dtypes; mutating it
            does not affect the verified Arrow table.
        """
        if name not in ARTIFACT_SCHEMAS:
            raise KeyError(f"Unknown bundle artifact {name!r}.")
        return _table_to_frame(self.artifacts[name], ARTIFACT_SCHEMAS[name])


def _artifact_frames(result: EvaluationResult) -> dict[str, pd.DataFrame]:
    return {
        "input_prices": result.input_prices,
        "coverage": result.coverage,
        "momentum": result.momentum,
        "volatility": result.volatility,
        "dependence_cross_section": result.dependence_cross_section,
        "dependence_per_etf": result.dependence_per_etf,
    }


def _arrow_array(series: pd.Series, dtype: str, arrow_type: pa.DataType) -> pa.Array:
    if dtype == "float64":
        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        return pa.array(values, type=arrow_type, from_pandas=False)
    if dtype == "date":
        dates = pd.to_datetime(series).dt.date
        return pa.array(dates, type=arrow_type, from_pandas=True)
    if dtype == "list<date>":
        return pa.array(series.tolist(), type=arrow_type, from_pandas=True)
    return pa.array(series, type=arrow_type, from_pandas=True)


def _frame_to_table(frame: pd.DataFrame, schema: ArtifactSchema) -> pa.Table:
    expected = [name for name, _ in schema.fields]
    if list(frame.columns) != expected:
        raise BundleContractError(
            f"{schema.name} DataFrame columns do not match the bundle contract."
        )
    arrow_schema = schema.arrow_schema()
    arrays = [
        _arrow_array(frame[name], dtype, arrow_schema.field(name).type)
        for name, dtype in schema.fields
    ]
    return pa.Table.from_arrays(arrays, schema=arrow_schema)


def _pandas_type_mapper(arrow_type: pa.DataType) -> Any:
    if pa.types.is_int64(arrow_type):
        return pd.Int64Dtype()
    if pa.types.is_float64(arrow_type):
        return pd.Float64Dtype()
    if pa.types.is_string(arrow_type):
        return pd.StringDtype()
    return None


def _table_to_frame(table: pa.Table, schema: ArtifactSchema) -> pd.DataFrame:
    frame = table.to_pandas(types_mapper=_pandas_type_mapper)
    for name, dtype in schema.fields:
        if dtype == "date":
            frame[name] = pd.to_datetime(frame[name])
        elif dtype == "list<date>":
            frame[name] = frame[name].map(
                lambda values: [] if values is None else list(values)
            )
        elif dtype == "string":
            frame[name] = frame[name].astype("string")
        elif dtype == "Int64":
            frame[name] = frame[name].astype("Int64")
        elif dtype == "int64":
            frame[name] = frame[name].astype("int64")
        elif dtype == "bool":
            frame[name] = frame[name].astype("bool")
    return cast(pd.DataFrame, frame.loc[:, [name for name, _ in schema.fields]])


def _utc_iso(value: datetime | pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        raise BundleContractError("retrieved_at values must be timezone-aware.")
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _retrieved_at(result: EvaluationResult) -> dict[str, object]:
    prices = result.input_prices
    by_ticker: list[dict[str, object]] = []
    for ticker in (etf.ticker for etf in result.universe.etfs):
        values = prices.loc[prices["ticker"].eq(ticker), "retrieved_at"]
        by_ticker.append(
            {
                "ticker": ticker,
                "min": None if values.empty else _utc_iso(values.min()),
                "max": None if values.empty else _utc_iso(values.max()),
            }
        )
    retrieved = prices["retrieved_at"]
    return {
        "min": None if retrieved.empty else _utc_iso(retrieved.min()),
        "max": None if retrieved.empty else _utc_iso(retrieved.max()),
        "by_ticker": by_ticker,
    }


def _git_state() -> tuple[str | None, bool | None]:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repository_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _package_versions() -> dict[str, str]:
    packages = {
        "etf-monitor": __version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "exchange-calendars": importlib.metadata.version("exchange-calendars"),
    }
    return {"python": sys.version.split()[0], **packages}


def _manifest(
    result: EvaluationResult,
    run_id: str,
    artifact_metadata: dict[str, dict[str, object]],
    input_content_sha256: str,
) -> dict[str, object]:
    commit, dirty = _git_state()
    methodology = result.methodology
    return {
        "format_version": FORMAT_VERSION,
        "run_id": run_id,
        "data_classification": result.data_classification,
        "history_status": HISTORY_STATUS,
        "evaluation_instant_utc": _utc_iso(result.evaluation_instant_utc),
        "target_session": result.target_session.date().isoformat(),
        "requested_interval": {
            "start": result.requested_interval[0].isoformat(),
            "end_exclusive": result.requested_interval[1].isoformat(),
        },
        "universe": {
            "version": result.universe.version,
            "sha256": result.universe.sha256,
            "tickers": [etf.ticker for etf in result.universe.etfs],
        },
        "methodology": {
            "version": methodology.methodology_version,
            "sha256": methodology.sha256,
        },
        "calendar": {
            "name": methodology.calendar,
            "version": methodology.exchange_calendars_version,
        },
        "sources": sorted(result.input_prices["source"].unique().tolist()),
        "retrieved_at": _retrieved_at(result),
        "input_content_sha256": input_content_sha256,
        "git": {"commit": commit, "dirty": dirty},
        "versions": _package_versions(),
        "artifacts": artifact_metadata,
        "limitations": list(LIMITATIONS),
    }


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bundle(result: EvaluationResult, root: Path) -> Path:
    """Publish one immutable seven-file bundle under ``root``.

    Args:
        result: Validated Gate 4 evaluation to serialize.
        root: Parent directory for uniquely named bundles.

    Returns:
        The final run directory.

    Raises:
        BundleContractError: If an artifact cannot satisfy its declared schema.
        FileExistsError: If the deterministic run ID already exists.
    """
    if result.data_classification not in {"FICTIONAL", "OBSERVED_PROVIDER_DATA"}:
        raise BundleContractError("Evaluation has an invalid data classification.")
    input_hash = content_sha256(result.input_prices, ARTIFACT_SCHEMAS["input_prices"])
    instant = result.evaluation_instant_utc.astimezone(UTC)
    run_id = (
        f"{result.target_session:%Y%m%d}T{instant:%Y%m%dT%H%M%SZ}-{input_hash[:12]}"
    )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Bundle destination already exists: {destination}.")

    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=root))
    try:
        artifact_metadata: dict[str, dict[str, object]] = {}
        frames = _artifact_frames(result)
        for name in _ARTIFACT_NAMES:
            schema = ARTIFACT_SCHEMAS[name]
            table = _frame_to_table(frames[name], schema)
            path = temporary / ARTIFACT_FILENAMES[name]
            pq.write_table(table, path)
            _fsync_file(path)
            payload = path.read_bytes()
            path.chmod(0o444)
            artifact_metadata[name] = {
                "filename": ARTIFACT_FILENAMES[name],
                "file_sha256": file_sha256(payload),
                "content_sha256": content_sha256(frames[name], schema),
                "row_count": table.num_rows,
                "schema": schema.manifest_schema(),
            }
        manifest = _manifest(result, run_id, artifact_metadata, input_hash)
        manifest_path = temporary / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _fsync_file(manifest_path)
        manifest_path.chmod(0o444)
        _fsync_directory(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Bundle destination already exists: {destination}.")
        os.replace(temporary, destination)
        _fsync_directory(root)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BundleContractError(f"Manifest has duplicate key {key!r}.")
        result[key] = value
    return result


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise BundleContractError(f"Manifest field {name!r} must be an object.")
    return dict(cast(Mapping[str, object], value))


def _require_keys(value: dict[str, object], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise BundleContractError(
            f"Manifest {name} keys mismatch; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


def _deep_freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _read_regular_file(path: Path) -> bytes:
    if path.is_symlink():
        raise BundleContractError(f"Bundle contains a symlink: {path.name}.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BundleContractError(f"Cannot open bundle file {path.name!r}.") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise BundleContractError(
                f"Bundle entry {path.name!r} is not a regular file."
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _parse_manifest(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleContractError("Manifest is not valid strict UTF-8 JSON.") from error
    return _mapping(value, "manifest")


def _validate_manifest(manifest: dict[str, object], path: Path) -> None:
    expected = {
        "format_version",
        "run_id",
        "data_classification",
        "history_status",
        "evaluation_instant_utc",
        "target_session",
        "requested_interval",
        "universe",
        "methodology",
        "calendar",
        "sources",
        "retrieved_at",
        "input_content_sha256",
        "git",
        "versions",
        "artifacts",
        "limitations",
    }
    _require_keys(manifest, expected, "top-level")
    if (
        type(manifest["format_version"]) is not int
        or manifest["format_version"] != FORMAT_VERSION
    ):
        raise BundleContractError("Unknown bundle format version.")
    if not isinstance(manifest["history_status"], str) or (
        manifest["history_status"] != HISTORY_STATUS
    ):
        raise BundleContractError("Manifest history_status is not current-vintage.")
    if not isinstance(manifest["data_classification"], str) or (
        manifest["data_classification"]
        not in {
            "FICTIONAL",
            "OBSERVED_PROVIDER_DATA",
        }
    ):
        raise BundleContractError("Manifest has an unknown data classification.")
    if not isinstance(manifest["target_session"], str):
        raise BundleContractError("Manifest target_session must be an ISO date.")
    try:
        target_date = date.fromisoformat(manifest["target_session"])
    except ValueError as error:
        raise BundleContractError("Manifest target_session is invalid.") from error
    if target_date.isoformat() != manifest["target_session"]:
        raise BundleContractError("Manifest target_session must use YYYY-MM-DD.")
    sources = manifest["sources"]
    if (
        not isinstance(sources, list)
        or len(sources) != 1
        or not all(isinstance(source, str) for source in sources)
        or sources != sorted(set(sources))
        or sources[0] not in ALLOWED_SOURCES
    ):
        raise BundleContractError("Manifest sources must contain one distinct source.")
    expected_classification = (
        "FICTIONAL" if sources[0] == "FICTIONAL_synthetic" else "OBSERVED_PROVIDER_DATA"
    )
    if manifest["data_classification"] != expected_classification:
        raise BundleIntegrityError("Manifest classification conflicts with its source.")
    if not isinstance(manifest["run_id"], str) or manifest["run_id"] != path.name:
        raise BundleContractError(
            "Bundle directory name does not match manifest run_id."
        )
    interval = _mapping(manifest["requested_interval"], "requested_interval")
    _require_keys(interval, {"start", "end_exclusive"}, "requested_interval")
    if not all(isinstance(interval[key], str) for key in ("start", "end_exclusive")):
        raise BundleContractError("Requested interval dates must be ISO strings.")
    try:
        date.fromisoformat(cast(str, interval["start"]))
        date.fromisoformat(cast(str, interval["end_exclusive"]))
    except ValueError as error:
        raise BundleContractError(
            "Requested interval contains an invalid date."
        ) from error
    universe = _mapping(manifest["universe"], "universe")
    _require_keys(universe, {"version", "sha256", "tickers"}, "universe")
    methodology = _mapping(manifest["methodology"], "methodology")
    _require_keys(methodology, {"version", "sha256"}, "methodology")
    calendar = _mapping(manifest["calendar"], "calendar")
    _require_keys(calendar, {"name", "version"}, "calendar")
    retrieved = _mapping(manifest["retrieved_at"], "retrieved_at")
    _require_keys(retrieved, {"min", "max", "by_ticker"}, "retrieved_at")
    if not isinstance(retrieved["by_ticker"], list):
        raise BundleContractError("retrieved_at.by_ticker must be a list.")
    if len(retrieved["by_ticker"]) != 24:
        raise BundleContractError("retrieved_at.by_ticker must contain 24 entries.")
    for item in retrieved["by_ticker"]:
        entry = _mapping(item, "retrieved_at.by_ticker")
        _require_keys(entry, {"ticker", "min", "max"}, "retrieved_at.by_ticker")
        if not isinstance(entry["ticker"], str):
            raise BundleContractError("Retrieved-at ticker entries must be strings.")
    git = _mapping(manifest["git"], "git")
    _require_keys(git, {"commit", "dirty"}, "git")
    if git["commit"] is not None and not isinstance(git["commit"], str):
        raise BundleContractError("Git commit metadata must be a string or null.")
    if git["dirty"] is not None and not isinstance(git["dirty"], bool):
        raise BundleContractError("Git dirty metadata must be a boolean or null.")
    versions = _mapping(manifest["versions"], "versions")
    _require_keys(
        versions,
        {"python", "etf-monitor", "numpy", "pandas", "pyarrow", "exchange-calendars"},
        "versions",
    )
    if not all(isinstance(version, str) for version in versions.values()):
        raise BundleContractError("All dependency version metadata must be strings.")
    artifacts = _mapping(manifest["artifacts"], "artifacts")
    _require_keys(artifacts, set(_ARTIFACT_NAMES), "artifacts")
    for name in _ARTIFACT_NAMES:
        item = _mapping(artifacts[name], f"artifacts.{name}")
        _require_keys(
            item,
            {"filename", "file_sha256", "content_sha256", "row_count", "schema"},
            f"artifacts.{name}",
        )
        if item["filename"] != ARTIFACT_FILENAMES[name]:
            raise BundleContractError(f"Unexpected filename for artifact {name!r}.")
        for hash_name in ("file_sha256", "content_sha256"):
            digest = item[hash_name]
            if not isinstance(digest, str) or len(digest) != 64:
                raise BundleContractError(
                    f"Artifact {name!r} has an invalid {hash_name}."
                )
            try:
                int(digest, 16)
            except ValueError as error:
                raise BundleContractError(
                    f"Artifact {name!r} has an invalid {hash_name}."
                ) from error
        if item["schema"] != ARTIFACT_SCHEMAS[name].manifest_schema():
            raise BundleContractError(f"Manifest schema mismatch for {name!r}.")
        if type(item["row_count"]) is not int or item["row_count"] < 0:
            raise BundleContractError(f"Invalid row count for artifact {name!r}.")
    if manifest["limitations"] != list(LIMITATIONS):
        raise BundleContractError(
            "Manifest limitations do not match the fixed contract."
        )
    input_hash = manifest["input_content_sha256"]
    if not isinstance(input_hash, str) or len(input_hash) != 64:
        raise BundleContractError("Manifest input content hash is invalid.")
    try:
        int(input_hash, 16)
    except ValueError as error:
        raise BundleContractError("Manifest input content hash is invalid.") from error


def _check_compatibility(manifest: dict[str, object]) -> None:
    universe = load_universe()
    methodology = load_methodology()
    serialized_universe = _mapping(manifest["universe"], "universe")
    serialized_methodology = _mapping(manifest["methodology"], "methodology")
    if (
        serialized_universe["version"] != universe.version
        or serialized_universe["sha256"] != universe.sha256
        or serialized_universe["tickers"] != [etf.ticker for etf in universe.etfs]
    ):
        raise BundleIncompatibleError(
            "Bundle universe differs from current configuration."
        )
    if (
        serialized_methodology["version"] != methodology.methodology_version
        or serialized_methodology["sha256"] != methodology.sha256
    ):
        raise BundleIncompatibleError(
            "Bundle methodology differs from current configuration."
        )


def _check_target_coverage(
    coverage: pd.DataFrame,
    momentum: pd.DataFrame,
    volatility: pd.DataFrame,
    target: pd.Timestamp,
) -> None:
    target_momentum = momentum.loc[momentum["signal_date"].eq(target)].set_index(
        "ticker"
    )
    target_volatility = volatility.loc[volatility["signal_date"].eq(target)].set_index(
        "ticker"
    )
    if len(target_momentum) != 24 or len(target_volatility) != 24:
        raise BundleIntegrityError("Signal artifacts do not have 24 target rows.")
    for row in coverage.itertuples(index=False):
        ticker = str(row.ticker)
        for coverage_name, signal_name in (
            ("momentum_target_raw", "raw_log_return"),
            ("momentum_target_display", "simple_return_pct"),
            ("momentum_target_percentile", "percentile"),
            ("momentum_target_reference_count", "reference_count"),
            ("momentum_target_status", "status"),
            (
                "momentum_target_normalization_status",
                "normalization_status",
            ),
        ):
            if not _same_value(
                getattr(row, coverage_name),
                target_momentum.loc[ticker, signal_name],
            ):
                raise BundleIntegrityError("Coverage Momentum target mismatch.")
        for coverage_name, signal_name in (
            ("volatility_target_raw", "raw_annualized_vol"),
            ("volatility_target_display", "display_vol_pct"),
            ("volatility_target_percentile", "percentile"),
            ("volatility_target_reference_count", "reference_count"),
            ("volatility_target_status", "status"),
            (
                "volatility_target_normalization_status",
                "normalization_status",
            ),
        ):
            if not _same_value(
                getattr(row, coverage_name),
                target_volatility.loc[ticker, signal_name],
            ):
                raise BundleIntegrityError("Coverage Volatility target mismatch.")


def _same_value(first: object, second: object) -> bool:
    if pd.isna(cast(Any, first)) and pd.isna(cast(Any, second)):
        return True
    return bool(first == second)


def verify_bundle(path: Path) -> VerifiedBundle:
    """Verify a bundle from one in-memory read and recompute all artifacts.

    Args:
        path: Candidate bundle directory.

    Returns:
        A frozen record whose artifact values are immutable Arrow tables.

    Raises:
        BundleContractError: If paths, manifest, or schemas violate contract.
        BundleIntegrityError: If any hash, row count, or recomputation differs.
        BundleIncompatibleError: If current universe/methodology has changed.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise BundleContractError("Bundle path must be a real directory.")
    entries = list(path.iterdir())
    names = {entry.name for entry in entries}
    if names != _EXPECTED_FILENAMES:
        raise BundleContractError(
            f"Bundle file set mismatch; missing={sorted(_EXPECTED_FILENAMES - names)}, "
            f"extra={sorted(names - _EXPECTED_FILENAMES)}."
        )
    payloads = {entry.name: _read_regular_file(entry) for entry in entries}
    manifest = _parse_manifest(payloads[MANIFEST_FILENAME])
    _validate_manifest(manifest, path)
    _check_compatibility(manifest)

    artifact_metadata = _mapping(manifest["artifacts"], "artifacts")
    tables: dict[str, pa.Table] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name in _ARTIFACT_NAMES:
        metadata = _mapping(artifact_metadata[name], f"artifacts.{name}")
        filename = ARTIFACT_FILENAMES[name]
        payload = payloads[filename]
        if file_sha256(payload) != metadata["file_sha256"]:
            raise BundleIntegrityError(f"Byte hash mismatch for {filename}.")
        try:
            table = pq.read_table(pa.BufferReader(payload))
        except (pa.ArrowException, OSError, ValueError) as error:
            raise BundleIntegrityError(f"Cannot read artifact {filename}.") from error
        schema = ARTIFACT_SCHEMAS[name]
        if not table.schema.equals(schema.arrow_schema(), check_metadata=False):
            raise BundleContractError(f"Arrow schema mismatch for {filename}.")
        if table.num_rows != metadata["row_count"]:
            raise BundleIntegrityError(f"Row count mismatch for {filename}.")
        frame = _table_to_frame(table, schema)
        if content_sha256(frame, schema) != metadata["content_sha256"]:
            raise BundleIntegrityError(f"Content hash mismatch for {filename}.")
        tables[name] = table
        frames[name] = frame

    input_hash = content_sha256(
        frames["input_prices"], ARTIFACT_SCHEMAS["input_prices"]
    )
    if input_hash != manifest["input_content_sha256"]:
        raise BundleIntegrityError("Input content hash does not match the manifest.")
    if not isinstance(manifest["input_content_sha256"], str):
        raise BundleContractError("Manifest input content hash must be a string.")
    input_sources = sorted(frames["input_prices"]["source"].unique().tolist())
    if input_sources != manifest["sources"]:
        raise BundleIntegrityError("Manifest sources do not match input rows.")
    if len(frames["coverage"]) != 24:
        raise BundleIntegrityError("Coverage artifact must contain exactly 24 rows.")
    target_value = manifest["target_session"]
    if not isinstance(target_value, str):
        raise BundleContractError("Manifest target_session must be an ISO date.")
    target = pd.Timestamp(date.fromisoformat(target_value))
    methodology = load_methodology()
    expected_sessions = xnys_sessions(methodology.price_history_start, target.date())
    expected_signal_rows = 24 * len(expected_sessions)
    if (
        len(frames["momentum"]) != expected_signal_rows
        or len(frames["volatility"]) != expected_signal_rows
    ):
        raise BundleIntegrityError("Signal artifact row count is not 24 x grid length.")
    _check_target_coverage(
        frames["coverage"], frames["momentum"], frames["volatility"], target
    )

    instant_value = manifest["evaluation_instant_utc"]
    if not isinstance(instant_value, str):
        raise BundleContractError("Manifest evaluation instant must be an ISO string.")
    try:
        instant = datetime.fromisoformat(instant_value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BundleContractError("Manifest evaluation instant is invalid.") from error
    if (
        not instant_value.endswith("Z")
        or instant.tzinfo is None
        or instant.utcoffset() != UTC.utcoffset(instant)
    ):
        raise BundleContractError("Manifest evaluation instant must be UTC.")
    expected_run_id = (
        f"{target:%Y%m%d}T{instant.astimezone(UTC):%Y%m%dT%H%M%SZ}-{input_hash[:12]}"
    )
    if manifest["run_id"] != expected_run_id:
        raise BundleIntegrityError("Manifest run ID does not match its inputs.")
    try:
        recomputed = evaluate(
            canonicalize_frame(frames["input_prices"]),
            instant,
        )
    except Exception as error:
        raise BundleIntegrityError(
            "Bundled inputs cannot reproduce the evaluation."
        ) from error
    if (
        recomputed.target_session.date().isoformat() != manifest["target_session"]
        or recomputed.data_classification != manifest["data_classification"]
        or recomputed.requested_interval[0].isoformat()
        != _mapping(manifest["requested_interval"], "requested_interval")["start"]
        or recomputed.requested_interval[1].isoformat()
        != _mapping(manifest["requested_interval"], "requested_interval")[
            "end_exclusive"
        ]
    ):
        raise BundleIntegrityError(
            "Manifest metadata does not match recomputed inputs."
        )
    for name, recomputed_frame in _artifact_frames(recomputed).items():
        if content_sha256(recomputed_frame, ARTIFACT_SCHEMAS[name]) != content_sha256(
            frames[name], ARTIFACT_SCHEMAS[name]
        ):
            raise BundleIntegrityError(f"Recomputed {name} artifact does not match.")
    if _retrieved_at(recomputed) != _mapping(manifest["retrieved_at"], "retrieved_at"):
        raise BundleIntegrityError(
            "Manifest retrieved-at metadata does not match input."
        )
    frozen_manifest = cast(Mapping[str, object], _deep_freeze(manifest))
    return VerifiedBundle(
        path=path,
        run_id=str(manifest["run_id"]),
        manifest=frozen_manifest,
        artifacts=MappingProxyType(tables),
    )


def list_candidate_runs(root: Path) -> list[str]:
    """List direct child directories as unverified bundle candidates."""
    root = Path(root)
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise BundleContractError("Candidate root must be a real directory.")
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and not entry.is_symlink()
    )
