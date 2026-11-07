"""Write-once vintage directories: exact raw files plus a SHA-256 manifest.

A vintage is staged in a hidden sibling directory, every file is fsynced and
made read-only, and the directory is renamed into place, so a vintage is
either complete or absent. Vintages are never merged or overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path

MANIFEST_NAME = "manifest.json"


class VintageError(ValueError):
    """Indicate that a vintage cannot be written or fails verification."""


def sha256_hex(payload: bytes) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_vintage_dir(
    final: Path, files: Mapping[str, bytes], manifest: dict[str, object]
) -> Path:
    """Atomically create ``final`` holding ``files`` and a manifest with digests.

    Raises:
        VintageError: If ``final`` already exists or a path is unsafe.
    """
    if final.exists():
        raise VintageError(f"Vintage {final} already exists.")
    if any(
        relative == MANIFEST_NAME
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        for relative in files
    ):
        raise VintageError("Vintage file names must be relative and not reserved.")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".staging-{final.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for relative, payload in sorted(files.items()):
            _write_bytes(staging / relative, payload)
        record = dict(manifest)
        record["files"] = {
            relative: sha256_hex(payload) for relative, payload in sorted(files.items())
        }
        _write_bytes(
            staging / MANIFEST_NAME,
            (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
        )
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def read_vintage_dir(
    path: Path, format_version: int
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Read a vintage after checking its file set and every digest.

    Returns:
        The manifest and the exact bytes of every listed file.

    Raises:
        VintageError: If the manifest is unreadable, the format differs, or any
            file is missing, extra, or altered.
    """
    try:
        manifest = json.loads((path / MANIFEST_NAME).read_text())
    except (OSError, ValueError) as error:
        raise VintageError(f"Cannot read manifest in {path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("format_version") != (
        format_version
    ):
        raise VintageError("Unsupported vintage format version.")
    listed = manifest.get("files")
    if not isinstance(listed, dict):
        raise VintageError("Manifest must list files.")
    on_disk = {
        file.relative_to(path).as_posix()
        for file in path.rglob("*")
        if file.is_file() and file.name != MANIFEST_NAME
    }
    if on_disk != set(listed):
        raise VintageError("Vintage files differ from the manifest.")
    payloads: dict[str, bytes] = {}
    for relative, digest in listed.items():
        payload = (path / relative).read_bytes()
        if sha256_hex(payload) != digest:
            raise VintageError(f"Digest mismatch for {relative}.")
        payloads[relative] = payload
    return manifest, payloads
