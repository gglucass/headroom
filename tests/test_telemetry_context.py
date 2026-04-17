"""Tests for headroom.telemetry.context (install_mode + headroom_stack detection)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.telemetry.context import detect_install_mode, detect_stack


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts without our env vars set."""

    monkeypatch.delenv("HEADROOM_STACK", raising=False)
    monkeypatch.delenv("HEADROOM_AGENT_TYPE", raising=False)
    yield


class TestDetectInstallMode:
    def test_wrapped_when_agent_type_set(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_AGENT_TYPE", "claude")
        assert detect_install_mode(8787) == "wrapped"

    def test_on_demand_when_no_env_and_no_manifest(self, monkeypatch):
        monkeypatch.setattr("headroom.install.state.list_manifests", lambda: [])
        assert detect_install_mode(8787) == "on_demand"

    def test_persistent_when_manifest_matches_port(self, monkeypatch):
        manifest = SimpleNamespace(port=8787, profile="default")
        monkeypatch.setattr("headroom.install.state.list_manifests", lambda: [manifest])
        assert detect_install_mode(8787) == "persistent"

    def test_on_demand_when_manifest_port_mismatches(self, monkeypatch):
        manifest = SimpleNamespace(port=9000, profile="other")
        monkeypatch.setattr("headroom.install.state.list_manifests", lambda: [manifest])
        assert detect_install_mode(8787) == "on_demand"

    def test_wrapped_takes_precedence_over_manifest(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_AGENT_TYPE", "codex")
        manifest = SimpleNamespace(port=8787, profile="default")
        monkeypatch.setattr("headroom.install.state.list_manifests", lambda: [manifest])
        assert detect_install_mode(8787) == "wrapped"

    def test_manifest_crash_falls_back_to_on_demand(self, monkeypatch):
        def _boom():
            raise RuntimeError("disk gone")

        monkeypatch.setattr("headroom.install.state.list_manifests", _boom)
        # install_mode should not raise; graceful fallback
        assert detect_install_mode(8787) == "on_demand"


class TestDetectStack:
    def test_explicit_env_wins(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_STACK", "custom_slug")
        assert detect_stack() == "custom_slug"

    def test_explicit_env_overrides_agent_type(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_STACK", "proxy")
        monkeypatch.setenv("HEADROOM_AGENT_TYPE", "claude")
        assert detect_stack() == "proxy"

    def test_wrap_slug_from_agent_type(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_AGENT_TYPE", "claude")
        assert detect_stack() == "wrap_claude"

    def test_unknown_agent_type_rejected(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_AGENT_TYPE", "somebespoke")
        assert detect_stack() == "unknown"

    def test_default_is_proxy(self):
        assert detect_stack() == "proxy"

    def test_default_is_proxy_with_empty_stats(self):
        assert detect_stack({"requests": {"by_stack": {}}}) == "proxy"

    def test_dominant_stack_from_stats(self):
        stats = {"requests": {"by_stack": {"adapter_ts_openai": 90, "adapter_ts_anthropic": 10}}}
        assert detect_stack(stats) == "adapter_ts_openai"

    def test_mixed_when_no_dominant_stack(self):
        stats = {"requests": {"by_stack": {"adapter_ts_openai": 40, "adapter_ts_anthropic": 60}}}
        assert detect_stack(stats) == "mixed"

    def test_single_stack_is_dominant(self):
        stats = {"requests": {"by_stack": {"adapter_ts_openai": 3}}}
        assert detect_stack(stats) == "adapter_ts_openai"

    def test_env_beats_stats(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_STACK", "wrap_claude")
        stats = {"requests": {"by_stack": {"adapter_ts_openai": 100}}}
        assert detect_stack(stats) == "wrap_claude"
