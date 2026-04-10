#!/usr/bin/env bash
set -euo pipefail

profile="${1:-default}"
project_env="${UV_PROJECT_ENVIRONMENT:-/home/vscode/.venvs/headroom}"
sync_extras=(--extra dev)

if [[ "$profile" == "memory-stack" ]]; then
  sync_extras+=(--extra memory-stack)
fi

cd /workspaces/headroom

uv sync --frozen "${sync_extras[@]}" --link-mode copy

uv run pre-commit install

echo "Headroom devcontainer is ready."
if [[ "$profile" == "memory-stack" ]]; then
  echo "Memory stack sidecars are available at qdrant:6333 and neo4j://neo4j:7687."
fi
echo "Run checks with:"
echo "  uv run ruff check ."
echo "  uv run ruff format --check ."
echo "  uv run mypy headroom --ignore-missing-imports"
echo "  uv run pytest -v --tb=short"
