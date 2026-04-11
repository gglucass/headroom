from __future__ import annotations

from headroom.install.models import InstallPreset, ProviderSelectionMode, ToolTarget
from headroom.install.planner import build_manifest, resolve_targets


def test_resolve_targets_auto_falls_back_when_detection_empty(monkeypatch) -> None:
    monkeypatch.setattr("headroom.install.planner.detect_targets", lambda: [])

    targets = resolve_targets(ProviderSelectionMode.AUTO.value, [])

    assert targets == [
        ToolTarget.CLAUDE.value,
        ToolTarget.CODEX.value,
        ToolTarget.COPILOT.value,
    ]


def test_build_manifest_for_persistent_docker_sets_expected_defaults() -> None:
    manifest = build_manifest(
        profile="default",
        preset=InstallPreset.PERSISTENT_DOCKER.value,
        runtime_kind="docker",
        scope="user",
        provider_mode="manual",
        targets=["claude", "copilot"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode="token",
        memory_enabled=True,
        telemetry_enabled=False,
        image="ghcr.io/chopratejas/headroom:latest",
    )

    assert manifest.supervisor_kind == "none"
    assert manifest.runtime_kind == "docker"
    assert manifest.health_url == "http://127.0.0.1:8787/readyz"
    assert manifest.base_env["HEADROOM_PORT"] == "8787"
    assert manifest.base_env["HEADROOM_TELEMETRY"] == "off"
    assert manifest.tool_envs["claude"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert manifest.tool_envs["copilot"]["COPILOT_PROVIDER_TYPE"] == "anthropic"
    assert "--memory" in manifest.proxy_args
