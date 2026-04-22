from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

from headroom.client import HeadroomClient
from headroom.compress import compress
from headroom.config import HeadroomConfig, HeadroomMode, TransformResult
from headroom.hooks import CompressionHooks
from headroom.pipeline import (
    CANONICAL_PIPELINE_STAGES,
    PipelineExtensionManager,
    PipelineStage,
    summarize_routing_markers,
)
from headroom.providers.base import Provider, TokenCounter


class RecordingExtension:
    def __init__(self) -> None:
        self.stages: list[PipelineStage] = []

    def on_pipeline_event(self, event):
        self.stages.append(event.stage)
        return None


class MutatingExtension:
    def on_pipeline_event(self, event):
        if event.stage == PipelineStage.INPUT_RECEIVED:
            event.messages = [{"role": "user", "content": "mutated"}]
        return event


class RecordingHooks(CompressionHooks):
    def __init__(self) -> None:
        self.stages: list[PipelineStage] = []
        self.post_event = None

    def pre_compress(self, messages, ctx):
        return messages

    def compute_biases(self, messages, ctx):
        return {}

    def post_compress(self, event):
        self.post_event = event

    def on_pipeline_event(self, event):
        self.stages.append(event.stage)
        return None


class StubPipeline:
    def apply(self, messages, model, **kwargs):
        return TransformResult(
            messages=messages,
            tokens_before=20,
            tokens_after=8,
            transforms_applied=["router:text:kompress", "kompress:user:0.40"],
        )

    def _get_tokenizer(self, model):
        return StubTokenCounter()


class StubTokenCounter(TokenCounter):
    def count_text(self, text: str) -> int:
        return len(text.split())

    def count_message(self, message: dict[str, Any]) -> int:
        content = message.get("content", "")
        if isinstance(content, str):
            return len(content.split())
        return 1

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        return sum(self.count_message(message) for message in messages)


class StubProvider(Provider):
    @property
    def name(self) -> str:
        return "openai"

    def get_token_counter(self, model: str) -> TokenCounter:
        return StubTokenCounter()

    def get_context_limit(self, model: str) -> int:
        return 128000

    def supports_model(self, model: str) -> bool:
        return True


class DummyCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"id": "resp_123", "messages": kwargs["messages"]}


class DummyOriginalClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=DummyCompletions())


def test_pipeline_extension_manager_uses_canonical_stage_contract():
    recorder = RecordingExtension()
    manager = PipelineExtensionManager(
        extensions=[recorder, MutatingExtension()],
        discover=False,
    )

    event = manager.emit(
        PipelineStage.INPUT_RECEIVED,
        operation="test",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert list(CANONICAL_PIPELINE_STAGES)[0] is PipelineStage.SETUP
    assert summarize_routing_markers(["router:text:kompress", "smart:kept=3"]) == [
        "router:text:kompress"
    ]
    assert recorder.stages == [PipelineStage.INPUT_RECEIVED]
    assert event.messages == [{"role": "user", "content": "mutated"}]


def test_compress_emits_canonical_pipeline_events(monkeypatch):
    hooks = RecordingHooks()
    compress_module = importlib.import_module("headroom.compress")
    monkeypatch.setattr(compress_module, "_get_pipeline", lambda: StubPipeline())

    result = compress(
        [{"role": "user", "content": "hello world"}],
        model="gpt-4o",
        hooks=hooks,
    )

    assert result.tokens_before == 20
    assert result.tokens_after == 2
    assert hooks.post_event is not None
    assert hooks.post_event.tokens_saved == 18
    assert hooks.stages == [
        PipelineStage.INPUT_RECEIVED,
        PipelineStage.INPUT_ROUTED,
        PipelineStage.INPUT_COMPRESSED,
    ]


def test_headroom_client_emits_canonical_pipeline_events(tmp_path):
    recorder = RecordingExtension()
    original = DummyOriginalClient()
    config = HeadroomConfig(
        store_url=f"jsonl://{tmp_path / 'headroom.jsonl'}",
        default_mode=HeadroomMode.OPTIMIZE,
        pipeline_extensions=[recorder],
        discover_pipeline_extensions=False,
    )
    client = HeadroomClient(
        original_client=original,
        provider=StubProvider(),
        store_url=f"jsonl://{tmp_path / 'headroom-client.jsonl'}",
        enable_cache_optimizer=False,
        config=config,
    )
    client._pipeline = StubPipeline()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello world"}],
    )

    assert response["id"] == "resp_123"
    assert recorder.stages == [
        PipelineStage.SETUP,
        PipelineStage.INPUT_RECEIVED,
        PipelineStage.INPUT_ROUTED,
        PipelineStage.INPUT_COMPRESSED,
        PipelineStage.PRE_SEND,
        PipelineStage.POST_SEND,
        PipelineStage.RESPONSE_RECEIVED,
    ]
