from __future__ import annotations

from headroom.providers.cursor import build_proxy_targets, render_setup_lines


def test_cursor_proxy_targets_use_local_headroom_proxy() -> None:
    targets = build_proxy_targets(9999)

    assert targets.openai_base_url == "http://127.0.0.1:9999/v1"
    assert targets.anthropic_base_url == "http://127.0.0.1:9999"


def test_cursor_setup_lines_include_both_provider_urls() -> None:
    lines = render_setup_lines(8787)
    joined = "\n".join(lines)

    assert "http://127.0.0.1:8787/v1" in joined
    assert "http://127.0.0.1:8787" in joined
