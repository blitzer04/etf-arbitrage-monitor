from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def test_streamlit_launches_from_temporary_working_directory(
    tmp_path: Path, allow_loopback_socket_connections: None
) -> None:
    bundles_root = tmp_path / "bundles"
    bundles_root.mkdir()
    app_path = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    executable_name = "streamlit.exe" if os.name == "nt" else "streamlit"
    streamlit = Path(sys.executable).parent / executable_name
    port = _free_port()
    process = subprocess.Popen(
        [
            str(streamlit),
            "run",
            str(app_path),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
            "--",
            "--bundles-root",
            str(bundles_root),
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr = ""
    try:
        deadline = time.monotonic() + 60
        health_ok = False
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as (
                    response
                ):
                    health_ok = (
                        response.status == 200
                        and response.read().decode("utf-8") == "ok"
                    )
                if health_ok:
                    break
            except OSError as error:
                last_error = error
                time.sleep(0.2)
        assert health_ok, f"Streamlit health check did not return ok: {last_error}"

        with urlopen(f"http://127.0.0.1:{port}/", timeout=10) as response:
            assert response.status == 200
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            _, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate(timeout=10)

    assert "ModuleNotFoundError" not in stderr
    assert "Traceback" not in stderr
