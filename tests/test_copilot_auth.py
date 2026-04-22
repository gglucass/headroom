from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from headroom import copilot_auth


def test_read_cached_oauth_token_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-env")
    assert copilot_auth.read_cached_oauth_token() == "gho-env"


def test_read_cached_oauth_token_reads_hosts_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hosts = tmp_path / "hosts.json"
    hosts.write_text(
        json.dumps(
            {
                "github.com": {
                    "oauth_token": "gho-file",
                    "expires_at": "2999-01-01T00:00:00Z",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("GITHUB_COPILOT_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_FILE", str(hosts))

    assert copilot_auth.read_cached_oauth_token() == "gho-file"


def test_read_cached_oauth_token_skips_expired_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hosts = tmp_path / "hosts.json"
    hosts.write_text(
        json.dumps({"github.com": {"oauthToken": "gho-old", "expiresAt": 1}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_FILE", str(hosts))

    assert copilot_auth.read_cached_oauth_token() is None


def test_resolve_client_bearer_token_prefers_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "copilot-api")
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-oauth")

    assert copilot_auth.resolve_client_bearer_token() == "copilot-api"


def test_is_copilot_api_url_matches_expected_hosts() -> None:
    assert copilot_auth.is_copilot_api_url("https://api.githubcopilot.com/v1/chat/completions")
    assert copilot_auth.is_copilot_api_url("wss://api.githubcopilot.com/v1/responses")
    assert not copilot_auth.is_copilot_api_url("https://api.openai.com/v1/chat/completions")


@pytest.mark.asyncio
async def test_apply_copilot_api_auth_replaces_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_api_token() -> copilot_auth.CopilotAPIToken:
        return copilot_auth.CopilotAPIToken(
            token="copilot-session",
            expires_at=time.time() + 3600,
            api_url=copilot_auth.DEFAULT_API_URL,
        )

    monkeypatch.setattr(
        copilot_auth.get_copilot_token_provider(),
        "get_api_token",
        fake_get_api_token,
    )

    headers = await copilot_auth.apply_copilot_api_auth(
        {"authorization": "Bearer downstream-token"},
        url="https://api.githubcopilot.com/v1/chat/completions",
    )

    assert headers["Authorization"] == "Bearer copilot-session"
    assert "authorization" not in headers


@pytest.mark.asyncio
async def test_token_provider_exchanges_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN", "gho-oauth")

    provider = copilot_auth.CopilotTokenProvider()
    calls = {"count": 0}

    def fake_exchange(headers: dict[str, str]) -> dict[str, object]:
        calls["count"] += 1
        return {
            "token": "copilot-api",
            "expires_at": int(time.time()) + 3600,
            "refresh_in": 1200,
            "endpoints": {"api": "https://api.githubcopilot.com"},
            "sku": "copilot_individual",
        }

    monkeypatch.setattr(provider, "_exchange_token_sync", staticmethod(fake_exchange))

    first = await provider.get_api_token()
    second = await provider.get_api_token()

    assert first.token == "copilot-api"
    assert second.token == "copilot-api"
    assert calls["count"] == 1
