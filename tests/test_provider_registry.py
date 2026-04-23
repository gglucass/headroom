from __future__ import annotations

from headroom.providers.registry import (
    ProviderApiOverrides,
    build_proxy_provider_runtime,
    format_backend_status,
    resolve_api_overrides,
    resolve_api_targets,
)
from headroom.proxy.models import ProxyConfig


def test_resolve_api_overrides_prefers_explicit_values_over_environment(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_TARGET_API_URL", "https://env.anthropic.example/v1")
    monkeypatch.setenv("OPENAI_TARGET_API_URL", "https://env.openai.example/v1")

    overrides = resolve_api_overrides(
        anthropic_api_url="https://cli.anthropic.example/v1",
        openai_api_url=None,
        gemini_api_url=None,
        cloudcode_api_url=None,
    )

    assert overrides == ProviderApiOverrides(
        anthropic="https://cli.anthropic.example/v1",
        openai="https://env.openai.example/v1",
        gemini=None,
        cloudcode=None,
    )


def test_resolve_api_targets_normalizes_trailing_v1() -> None:
    targets = resolve_api_targets(
        ProviderApiOverrides(
            anthropic="https://anthropic.example/v1/",
            openai="https://openai.example/v1",
            gemini="https://gemini.example/v1",
            cloudcode="https://cloudcode.example/v1/",
        )
    )

    assert targets.anthropic == "https://anthropic.example"
    assert targets.openai == "https://openai.example"
    assert targets.gemini == "https://gemini.example"
    assert targets.cloudcode == "https://cloudcode.example"


def test_proxy_config_exposes_provider_api_overrides() -> None:
    config = ProxyConfig(
        anthropic_api_url="https://anthropic.example",
        openai_api_url="https://openai.example",
        gemini_api_url=None,
        cloudcode_api_url="https://cloudcode.example",
    )

    assert config.provider_api_overrides == ProviderApiOverrides(
        anthropic="https://anthropic.example",
        openai="https://openai.example",
        gemini=None,
        cloudcode="https://cloudcode.example",
    )


def test_format_backend_status_for_anyllm() -> None:
    assert (
        format_backend_status(
            backend="anyllm",
            anyllm_provider="groq",
            bedrock_region="us-central1",
        )
        == "Groq via any-llm"
    )


def test_proxy_provider_runtime_routes_model_metadata_and_passthrough() -> None:
    runtime = build_proxy_provider_runtime(ProxyConfig())

    assert runtime.model_metadata_provider({"x-api-key": "test"}) == "anthropic"
    assert runtime.model_metadata_provider({}) == "openai"
    assert (
        runtime.select_passthrough_base_url({"x-goog-api-key": "test"})
        == runtime.api_targets.gemini
    )
