from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.app.helpers import write_fictional_bundle

_BUNDLES_ENV = "ETF_MONITOR_BUNDLES_ROOT"
_APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
_APP_TIMEOUT = 120


def _text_values(app: AppTest, *element_names: str) -> list[str]:
    return [
        str(element.value) for name in element_names for element in getattr(app, name)
    ]


def _app_text(app: AppTest) -> str:
    return "\n".join(
        _text_values(
            app,
            "title",
            "header",
            "subheader",
            "caption",
            "markdown",
            "error",
            "info",
            "warning",
            "success",
            "json",
        )
    )


def test_run_selection_is_explicit_and_scoped_to_selected_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundles"
    run_a = write_fictional_bundle(root, "2018-01-02")
    run_b = write_fictional_bundle(root, "2018-01-03")
    monkeypatch.setenv(_BUNDLES_ENV, str(root))

    app = AppTest.from_file(_APP_PATH, default_timeout=_APP_TIMEOUT).run(
        timeout=_APP_TIMEOUT
    )
    assert app.sidebar.selectbox[0].value is None
    assert not app.tabs
    assert not app.dataframe
    assert "Coverage" not in [element.value for element in app.subheader]

    app.sidebar.selectbox[0].set_value(run_a.name)
    app.run(timeout=_APP_TIMEOUT)
    assert run_a.name in _app_text(app)
    assert any(
        "FICTIONAL synthetic data" in str(element.value) for element in app.error
    )

    app.sidebar.selectbox[0].set_value(run_b.name)
    app.run(timeout=_APP_TIMEOUT)
    assert run_b.name in _app_text(app)
    markdown_and_captions = _text_values(app, "markdown", "caption")
    assert run_a.name not in "\n".join(markdown_and_captions)
    assert "Flow: deferred, not calculated" in "\n".join(markdown_and_captions)
    assert "Concentration: deferred, not calculated" in "\n".join(markdown_and_captions)

    all_text = _app_text(app).casefold()
    for forbidden in (
        "crowding score",
        "composite score:",
        "high risk",
        "moderate risk",
        "low risk",
    ):
        assert forbidden not in all_text


def test_corrupt_bundle_is_rejected_without_rendering_tabs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = write_fictional_bundle(tmp_path / "source", "2018-01-02")
    root = tmp_path / "corrupted-runs"
    root.mkdir()
    corrupted = Path(shutil.copytree(original, root / original.name))
    momentum = corrupted / "momentum.parquet"
    momentum.chmod(0o644)
    payload = bytearray(momentum.read_bytes())
    payload[len(payload) // 2] ^= 1
    momentum.write_bytes(payload)
    monkeypatch.setenv(_BUNDLES_ENV, str(root))

    app = AppTest.from_file(_APP_PATH, default_timeout=_APP_TIMEOUT).run(
        timeout=_APP_TIMEOUT
    )
    app.sidebar.selectbox[0].set_value(corrupted.name)
    app.run(timeout=_APP_TIMEOUT)

    assert any("failed verification" in str(element.value) for element in app.error)
    assert not app.tabs
    assert not app.dataframe


def test_empty_root_shows_information_without_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    monkeypatch.setenv(_BUNDLES_ENV, str(root))

    app = AppTest.from_file(_APP_PATH, default_timeout=_APP_TIMEOUT).run(
        timeout=_APP_TIMEOUT
    )

    assert any(
        "Select a run in the sidebar" in str(element.value) for element in app.info
    )
    assert not app.error
    assert not app.tabs
