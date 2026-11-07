"""Global test guards for live transport construction and socket access."""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from ipaddress import ip_address
from typing import Any

import pytest

from etf_monitor.providers.transport import (
    UrllibChartTransport,
    UrllibSsgaNavHistoryTransport,
)

_SOCKET_CONNECT = socket.socket.connect
_SOCKET_CONNECT_EX = socket.socket.connect_ex
_SOCKETPAIR = socket.socketpair
_SOCKETPAIR_STATE = threading.local()


def _is_loopback_address(address: object) -> bool:
    if not isinstance(address, tuple) or not address or not isinstance(address[0], str):
        return False
    try:
        return ip_address(address[0]).is_loopback
    except ValueError:
        return False


def _socketpair(*args: Any, **kwargs: Any) -> tuple[socket.socket, socket.socket]:
    """Run ``socket.socketpair`` with its internal loopback connect permitted.

    On Windows there is no native socketpair, so the stdlib fallback (used by
    asyncio's self-pipe) connects two sockets over 127.0.0.1.
    """
    _SOCKETPAIR_STATE.active = True
    try:
        return _SOCKETPAIR(*args, **kwargs)
    finally:
        _SOCKETPAIR_STATE.active = False


def _inside_socketpair(address: object) -> bool:
    return getattr(_SOCKETPAIR_STATE, "active", False) and _is_loopback_address(address)


@pytest.fixture(autouse=True)
def block_live_transport_and_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Fail immediately on live transport construction or non-loopback sockets."""

    def forbidden_transport_init(self: UrllibChartTransport) -> None:
        raise AssertionError("UrllibChartTransport must not be constructed in tests.")

    def forbidden_socket_connect(self: socket.socket, address: Any) -> None:
        if _inside_socketpair(address):
            _SOCKET_CONNECT(self, address)
            return
        raise AssertionError("Non-loopback socket connections are forbidden in tests.")

    def forbidden_socket_connect_ex(self: socket.socket, address: Any) -> int:
        if _inside_socketpair(address):
            return _SOCKET_CONNECT_EX(self, address)
        raise AssertionError("Non-loopback socket connections are forbidden in tests.")

    def forbidden_nav_transport_init(self: UrllibSsgaNavHistoryTransport) -> None:
        raise AssertionError(
            "UrllibSsgaNavHistoryTransport must not be constructed in tests."
        )

    monkeypatch.setattr(UrllibChartTransport, "__init__", forbidden_transport_init)
    monkeypatch.setattr(
        UrllibSsgaNavHistoryTransport, "__init__", forbidden_nav_transport_init
    )
    monkeypatch.setattr(socket.socket, "connect", forbidden_socket_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden_socket_connect_ex)
    monkeypatch.setattr(socket, "socketpair", _socketpair)
    yield


@pytest.fixture
def allow_loopback_socket_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow a test to connect only to an IP loopback address."""

    def connect(self: socket.socket, address: Any) -> None:
        if not _is_loopback_address(address):
            raise AssertionError(
                "Only loopback socket connections are allowed in this test."
            )
        _SOCKET_CONNECT(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if not _is_loopback_address(address):
            raise AssertionError(
                "Only loopback socket connections are allowed in this test."
            )
        return _SOCKET_CONNECT_EX(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
