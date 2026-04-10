from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


@pytest.fixture
def client():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as test_client:
        yield test_client


def test_livez_reports_process_health(client):
    response = client.get("/livez")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "headroom-proxy"
    assert data["status"] == "healthy"
    assert data["alive"] is True
    assert data["uptime_seconds"] >= 0


def test_readyz_reports_core_subsystem_checks(client):
    response = client.get("/readyz")

    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["status"] == "healthy"
    assert "config" not in data
    assert data["checks"]["startup"]["status"] == "healthy"
    assert data["checks"]["http_client"]["status"] == "healthy"
    assert data["checks"]["cache"]["status"] == "disabled"
    assert data["checks"]["rate_limiter"]["status"] == "disabled"
    assert data["checks"]["memory"]["status"] == "disabled"


def test_health_preserves_backwards_compatible_config_payload(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ready"] is True
    assert data["config"] == {
        "optimize": False,
        "cache": False,
        "rate_limit": False,
    }


def test_health_remains_200_when_proxy_is_not_ready(client):
    client.app.state.ready = False

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ready"] is False


def test_readyz_reports_memory_backend_when_enabled(tmp_path):
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        memory_enabled=True,
        memory_backend="local",
        memory_db_path=str(tmp_path / "headroom_memory.db"),
        memory_inject_tools=True,
        memory_inject_context=True,
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    data = response.json()
    assert data["checks"]["memory"]["status"] == "healthy"
    assert data["checks"]["memory"]["backend"] == "local"
    assert data["checks"]["memory"]["initialized"] is True


def test_readyz_initializes_qdrant_memory_backend(monkeypatch):
    from headroom.memory.backends import direct_mem0

    init_calls: list[str] = []

    class FakeDirectMem0Adapter:
        def __init__(self, config):
            self.config = config

        async def ensure_initialized(self):
            init_calls.append("initialized")

    monkeypatch.setattr(direct_mem0, "DirectMem0Adapter", FakeDirectMem0Adapter)

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        memory_enabled=True,
        memory_backend="qdrant-neo4j",
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    data = response.json()
    assert init_calls == ["initialized"]
    assert data["checks"]["memory"]["status"] == "healthy"
    assert data["checks"]["memory"]["backend"] == "qdrant-neo4j"
    assert data["checks"]["memory"]["initialized"] is True


def test_shutdown_tolerates_stubbed_memory_handler():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    with TestClient(app) as client:
        client.app.state.proxy.memory_handler = SimpleNamespace(
            health_status=lambda: {
                "enabled": False,
                "backend": None,
                "initialized": False,
                "native_tool": False,
                "bridge_enabled": False,
            }
        )
        response = client.get("/health")

    assert response.status_code == 200
