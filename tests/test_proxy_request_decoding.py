"""Regression tests for compressed JSON request bodies in the proxy."""

import gzip
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

pytest.importorskip("fastapi")
zstandard = pytest.importorskip("zstandard")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _response_payload() -> dict:
    return {
        "id": "resp_test_123",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _make_client():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    return TestClient(app)


class TestProxyRequestDecoding:
    def test_openai_responses_accepts_zstd_request_bodies(self):
        payload = {"model": "gpt-4o-mini", "input": "Ping"}
        compressed = zstandard.ZstdCompressor().compress(json.dumps(payload).encode("utf-8"))

        with patch(
            "headroom.proxy.server.HeadroomProxy._retry_request",
            new_callable=AsyncMock,
        ) as mock_retry, patch("headroom.proxy.server.get_tokenizer") as mock_get_tokenizer:
            mock_retry.return_value = httpx.Response(200, json=_response_payload())
            mock_get_tokenizer.return_value.count_messages.return_value = 1

            with _make_client() as client:
                response = client.post(
                    "/v1/responses",
                    headers={
                        "Authorization": "Bearer test-key",
                        "Content-Type": "application/json",
                        "Content-Encoding": "zstd",
                    },
                    content=compressed,
                )

        assert response.status_code == 200, response.text
        assert response.json()["id"] == "resp_test_123"
        mock_retry.assert_awaited_once()

    def test_openai_responses_accepts_gzip_request_bodies(self):
        payload = {"model": "gpt-4o-mini", "input": "Ping"}
        compressed = gzip.compress(json.dumps(payload).encode("utf-8"))

        with patch(
            "headroom.proxy.server.HeadroomProxy._retry_request",
            new_callable=AsyncMock,
        ) as mock_retry, patch("headroom.proxy.server.get_tokenizer") as mock_get_tokenizer:
            mock_retry.return_value = httpx.Response(200, json=_response_payload())
            mock_get_tokenizer.return_value.count_messages.return_value = 1

            with _make_client() as client:
                response = client.post(
                    "/v1/responses",
                    headers={
                        "Authorization": "Bearer test-key",
                        "Content-Type": "application/json",
                        "Content-Encoding": "gzip",
                    },
                    content=compressed,
                )

        assert response.status_code == 200, response.text
        assert response.json()["id"] == "resp_test_123"
        mock_retry.assert_awaited_once()

    def test_openai_responses_rejects_unsupported_request_encoding(self):
        with _make_client() as client:
            response = client.post(
                "/v1/responses",
                headers={
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json",
                    "Content-Encoding": "br",
                },
                content=b"{}",
            )

        assert response.status_code == 415
        assert "Unsupported Content-Encoding" in response.json()["error"]["message"]
