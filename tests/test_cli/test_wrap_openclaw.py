"""Tests for `headroom wrap openclaw` command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    """Create a minimal OpenClaw plugin directory fixture."""
    plugin = tmp_path / "plugins" / "openclaw"
    plugin.mkdir(parents=True)
    (plugin / "package.json").write_text('{"name":"headroom-openclaw"}\n')
    (plugin / "openclaw.plugin.json").write_text('{"id":"headroom"}\n')
    return plugin


def _make_successful_run(calls: list[dict]) -> object:
    def run(cmd, **kwargs):  # noqa: ANN001
        calls.append({"cmd": list(cmd), **kwargs})
        return MagicMock(returncode=0, stdout="", stderr="")

    return run


def test_wrap_openclaw_happy_path_installs_builds_configures_and_restarts(
    runner: CliRunner, plugin_dir: Path
) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap._default_openclaw_plugin_dir", return_value=plugin_dir):
            with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
                result = runner.invoke(main, ["wrap", "openclaw"])

    assert result.exit_code == 0, result.output

    cmds = [c["cmd"] for c in calls]
    assert ["npm", "install"] in cmds
    assert ["npm", "run", "build"] in cmds
    assert [
        "openclaw",
        "plugins",
        "install",
        "--dangerously-force-unsafe-install",
        "--link",
        ".",
    ] in cmds
    assert ["openclaw", "config", "validate"] in cmds
    assert ["openclaw", "gateway", "restart"] in cmds
    assert ["openclaw", "plugins", "inspect", "headroom"] in cmds

    # Verify plugin install uses plugin cwd when linking
    install_call = next(
        c
        for c in calls
        if c["cmd"][:4]
        == ["openclaw", "plugins", "install", "--dangerously-force-unsafe-install"]
    )
    assert install_call["cwd"] == str(plugin_dir)

    # Verify config payload includes enabled + expected defaults
    set_entry = next(c for c in calls if c["cmd"][:4] == ["openclaw", "config", "set", "plugins.entries.headroom"])
    payload = json.loads(set_entry["cmd"][4])
    assert payload["enabled"] is True
    assert payload["config"]["proxyPort"] == 8787
    assert payload["config"]["autoStart"] is True
    assert payload["config"]["startupTimeoutMs"] == 20000


def test_wrap_openclaw_skip_build_and_no_restart(runner: CliRunner, plugin_dir: Path) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "openclaw",
                    "--plugin-path",
                    str(plugin_dir),
                    "--skip-build",
                    "--no-restart",
                ],
            )

    assert result.exit_code == 0, result.output
    cmds = [c["cmd"] for c in calls]
    assert ["npm", "install"] not in cmds
    assert ["npm", "run", "build"] not in cmds
    assert ["openclaw", "gateway", "restart"] not in cmds


def test_wrap_openclaw_fails_when_openclaw_missing(runner: CliRunner, plugin_dir: Path) -> None:
    def which(name: str) -> str | None:
        return None if name == "openclaw" else "npm"

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(plugin_dir)])

    assert result.exit_code != 0
    assert "'openclaw' not found in PATH" in result.output


def test_wrap_openclaw_fails_when_plugin_path_invalid(runner: CliRunner, tmp_path: Path) -> None:
    invalid = tmp_path / "missing-plugin"

    with patch("headroom.cli.wrap.shutil.which", return_value="openclaw"):
        result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(invalid)])

    assert result.exit_code != 0
    assert "Plugin path not found" in result.output


def test_wrap_openclaw_uses_extension_fallback_on_linked_install_bug(
    runner: CliRunner, plugin_dir: Path
) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        calls.append({"cmd": list(cmd), **kwargs})
        if cmd[:3] == ["openclaw", "plugins", "install"]:
            return MagicMock(
                returncode=1,
                stdout="Also not a valid hook pack",
                stderr='Plugin installation blocked despite "--dangerously-force-unsafe-install"',
            )
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            with patch(
                "headroom.cli.wrap._copy_openclaw_plugin_into_extensions",
                return_value=Path("C:/Users/test/.openclaw/extensions/headroom"),
            ) as copy_fallback:
                result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(plugin_dir)])

    assert result.exit_code == 0, result.output
    copy_fallback.assert_called_once()
