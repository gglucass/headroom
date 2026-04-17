"""Deployment context detection for telemetry.

Derives two orthogonal identity fields the beacon reports:

* ``install_mode`` — how the proxy process is deployed
  (``persistent`` / ``on_demand`` / ``wrapped`` / ``unknown``).
* ``headroom_stack`` — how Headroom is being invoked
  (``proxy``, ``wrap_claude``, ``adapter_ts_openai``, ...).

Both helpers are best-effort and never raise: telemetry is fire-and-forget and
must not break the proxy.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


_KNOWN_WRAP_AGENTS = frozenset(
    {"claude", "copilot", "codex", "aider", "cursor", "openclaw"}
)


def _slug_from_agent_type(agent_type: str) -> str:
    """Return ``wrap_<agent>`` for known agents, otherwise ``unknown``."""

    agent_type = agent_type.strip().lower()
    if agent_type and agent_type in _KNOWN_WRAP_AGENTS:
        return f"wrap_{agent_type}"
    return "unknown"


def detect_install_mode(port: int) -> str:
    """Classify how the proxy is deployed.

    Resolution order:

    1. ``HEADROOM_AGENT_TYPE`` env var set → ``wrapped`` (spawned by ``headroom wrap``).
    2. A ``DeploymentManifest`` on disk whose port matches ``port`` → ``persistent``.
    3. Otherwise → ``on_demand``.

    Any failure falls back to ``unknown`` so a broken install subsystem
    doesn't silence telemetry.
    """

    try:
        if os.environ.get("HEADROOM_AGENT_TYPE"):
            return "wrapped"

        try:
            from headroom.install.state import list_manifests

            for manifest in list_manifests():
                if getattr(manifest, "port", None) == port:
                    return "persistent"
        except Exception:
            logger.debug(
                "Beacon: manifest lookup failed during install_mode detection",
                exc_info=True,
            )

        return "on_demand"
    except Exception:
        logger.debug("Beacon: detect_install_mode crashed", exc_info=True)
        return "unknown"


def detect_stack(stats: dict[str, Any] | None = None) -> str:
    """Classify how Headroom is being invoked.

    Resolution order:

    1. ``HEADROOM_STACK`` env var set → use that slug verbatim.
    2. ``HEADROOM_AGENT_TYPE`` env var set → ``wrap_<agent>``.
    3. ``stats['requests']['by_stack']`` dict populated →
       pick the stack with >80% of requests, else ``mixed``.
    4. Otherwise → ``proxy``.

    Any failure falls back to ``unknown``.
    """

    try:
        explicit = os.environ.get("HEADROOM_STACK")
        if explicit:
            return explicit.strip().lower()

        agent_type = os.environ.get("HEADROOM_AGENT_TYPE")
        if agent_type:
            return _slug_from_agent_type(agent_type)

        if stats:
            by_stack = (stats.get("requests") or {}).get("by_stack") or {}
            if by_stack:
                total = sum(by_stack.values())
                if total > 0:
                    dominant, count = max(by_stack.items(), key=lambda kv: kv[1])
                    if count / total >= 0.8:
                        return dominant
                    return "mixed"

        return "proxy"
    except Exception:
        logger.debug("Beacon: detect_stack crashed", exc_info=True)
        return "unknown"
