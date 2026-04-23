from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

from headroom.providers.copilot.wrap import (
    build_launch_env,
    detect_running_proxy_backend,
    model_configured,
    provider_key_source,
    query_proxy_config,
    resolve_provider_type,
    validate_configuration,
)


def test_query_proxy_config_handles_success_and_invalid_payload() -> None:
    payload = io.BytesIO(json.dumps({"config": {"backend": "anyllm"}}).encode("utf-8"))
    payload_missing = io.BytesIO(json.dumps({"status": "ok"}).encode("utf-8"))

    with patch("urllib.request.urlopen", return_value=payload):
        assert query_proxy_config(8787) == {"backend": "anyllm"}
    with patch("urllib.request.urlopen", return_value=payload_missing):
        assert query_proxy_config(8787) is None
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        assert query_proxy_config(8787) is None


def test_detect_running_proxy_backend_requires_string_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        "headroom.providers.copilot.wrap.query_proxy_config",
        lambda port: {"backend": 123} if port == 8787 else None,
    )

    assert detect_running_proxy_backend(8787) is None
    assert detect_running_proxy_backend(9999) is None


def test_resolve_provider_type_prefers_explicit_and_env() -> None:
    assert resolve_provider_type("anthropic", "openai") == "openai"
    assert resolve_provider_type(None, "auto", {"HEADROOM_BACKEND": "anthropic"}) == "anthropic"
    assert resolve_provider_type(None, "auto", {"HEADROOM_BACKEND": "anyllm"}) == "openai"


def test_validate_configuration_accepts_supported_combinations() -> None:
    validate_configuration(provider_type="openai", wire_api="responses", backend=None)
    validate_configuration(provider_type="openai", wire_api="completions", backend="anyllm")


def test_provider_key_source_and_build_launch_env_cover_anthropic_and_openai() -> None:
    assert provider_key_source("anthropic") == "ANTHROPIC_API_KEY"
    assert provider_key_source("openai") == "OPENAI_API_KEY"

    anthropic_env, anthropic_lines = build_launch_env(
        port=8787,
        provider_type="anthropic",
        wire_api="responses",
        environ={
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "COPILOT_PROVIDER_WIRE_API": "stale",
        },
    )
    openai_env, openai_lines = build_launch_env(
        port=8787,
        provider_type="openai",
        wire_api=None,
        environ={"OPENAI_API_KEY": "sk-proj-test"},
    )

    assert anthropic_env["COPILOT_PROVIDER_TYPE"] == "anthropic"
    assert anthropic_env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787"
    assert anthropic_env["COPILOT_PROVIDER_API_KEY"] == "sk-ant-test"
    assert "COPILOT_PROVIDER_WIRE_API" not in anthropic_env
    assert anthropic_lines == [
        "COPILOT_PROVIDER_TYPE=anthropic",
        "COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:8787",
    ]

    assert openai_env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert openai_env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert openai_env["COPILOT_PROVIDER_WIRE_API"] == "completions"
    assert openai_env["COPILOT_PROVIDER_API_KEY"] == "sk-proj-test"
    assert openai_lines[-1] == "COPILOT_PROVIDER_WIRE_API=completions"


def test_model_configured_detects_env_and_cli_variants() -> None:
    assert model_configured((), {"COPILOT_MODEL": "gpt-4o"}) is True
    assert model_configured(("--model", "gpt-4o"), {}) is True
    assert model_configured(("--model=gpt-4o",), {}) is True
    assert model_configured(("--other", "value"), {}) is False
