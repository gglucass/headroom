"""Tests for Codex CLI wrapper helpers."""

import pytest

pytest.importorskip("click")

from headroom.cli.wrap import _codex_base_url, _codex_proxy_args  # noqa: E402


def test_codex_base_url_uses_proxy_v1_path():
    assert _codex_base_url(8787) == "http://127.0.0.1:8787/v1"


def test_codex_proxy_args_force_http_responses_transport():
    assert _codex_proxy_args(8787) == (
        "-c",
        'openai_base_url="http://127.0.0.1:8787/v1"',
        "--disable",
        "responses_websockets",
        "--disable",
        "responses_websockets_v2",
    )
