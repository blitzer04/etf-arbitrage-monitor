"""Strict YAML configuration loading shared by package configuration modules."""

from __future__ import annotations

from collections.abc import Hashable
from importlib.resources import files
from pathlib import Path
from typing import cast

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects repeated keys in mappings."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Hashable, object]:
    """Build a mapping while rejecting duplicate or unhashable keys."""
    loader.flatten_mapping(node)
    mapping: dict[Hashable, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        hashable_key = cast(Hashable, key)
        if hashable_key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {hashable_key!r}",
                key_node.start_mark,
            )
        mapping[hashable_key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_yaml_bytes(payload: bytes, *, source: str) -> object:
    """Parse YAML safely and reject duplicate mapping keys.

    Args:
        payload: YAML document bytes.
        source: Human-readable configuration source for error messages.

    Returns:
        The parsed YAML value.

    Raises:
        ValueError: If the YAML document is invalid or contains duplicate keys.
    """
    try:
        return yaml.load(payload, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {source}: {error}") from error


def read_config_bytes(path: Path | None, resource_name: str) -> tuple[bytes, str]:
    """Read a package-data resource or an explicit configuration file.

    Args:
        path: Optional filesystem path used by tests and tooling.
        resource_name: YAML filename inside ``etf_monitor.config``.

    Returns:
        Configuration bytes and a display name for diagnostics.

    Raises:
        OSError: If an explicit path or package resource cannot be read.
    """
    if path is not None:
        return path.read_bytes(), str(path)

    resource = files("etf_monitor.config").joinpath(resource_name)
    return resource.read_bytes(), f"etf_monitor.config/{resource_name}"
