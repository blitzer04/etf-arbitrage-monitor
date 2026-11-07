"""Process-level proof that installed package imports are cwd-independent."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_installed_package_imports_from_an_unrelated_working_directory(
    tmp_path: Path,
) -> None:
    code = (
        "import etf_monitor; "
        "from etf_monitor.universe import load_universe; "
        "assert etf_monitor.__version__ == '0.1.0'; "
        "assert len(load_universe().etfs) == 24"
    )

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
