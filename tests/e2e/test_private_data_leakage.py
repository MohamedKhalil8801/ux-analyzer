from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from ux_analyzer.domain.attention import AttentionState, ProgressiveObservation
from ux_analyzer.domain.benchmark import Budget
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    PrivateExecutionReference,
    ViewportSnapshot,
)
from ux_analyzer.ports.models import ChatMessage, ModelRole
from ux_analyzer.providers.cognitive import (
    CognitiveModelResponse,
    StructuredCognitiveAgent,
)
from ux_analyzer.providers.scent import (
    CoarseScentResponse,
    FullScentResponse,
    StructuredCoarseScentEvaluator,
    StructuredFullScentEvaluator,
)

_RECORDINGS = Path(__file__).parents[1] / "recordings"


class _RecordingModelClient:
    endpoint_origin = "https://llm.example.test"

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def complete(
        self,
        schema: type[BaseModel],
        messages: tuple[ChatMessage, ...],
        model: str,
        role: ModelRole,
    ) -> BaseModel:
        payload = json.loads(messages[-1].content)
        self.requests.append(
            {
                "role": role.value,
                "model": model,
                "schema": schema.__name__,
                "messages": [message.model_dump() for message in messages],
            }
        )
        recordings = json.loads(
            (_RECORDINGS / "ci-model-responses.json").read_text(encoding="utf-8")
        )
        if schema is CoarseScentResponse:
            return schema.model_validate(
                {
                    "scores": [
                        {"element_id": item["element_id"], "score": recordings["scent"]}
                        for item in payload["elements"]
                    ]
                }
            )
        if schema is FullScentResponse:
            return schema.model_validate(
                {
                    "scores": [
                        {"element_id": item["element_id"], "score": recordings["scent"]}
                        for item in payload["elements"]
                    ]
                }
            )
        if schema is CognitiveModelResponse:
            return schema.model_validate(recordings["cognitive"])
        raise AssertionError(f"unexpected role schema: {schema!r}")


def _snapshot() -> ViewportSnapshot:
    return ViewportSnapshot(
        id="viewport-1",
        provider_id="fixture",
        elements=(
            ElementSnapshot(
                id="target",
                role="button",
                label="Invite teammate",
                bounds=BoundingBox(x=10, y=10, width=100, height=30),
                visibility_fraction=1,
                actionable=True,
                provider_id="fixture",
                execution_reference=PrivateExecutionReference(
                    provider_id="fixture",
                    viewport_id="viewport-1",
                    token="private-token",
                ),
                selector="button[data-testid='invite']",
                test_id="target-test-id",
                hidden_label="private hidden label",
                destination_url="https://fixture.test/private",
            ),
            ElementSnapshot(
                id="other",
                role="link",
                label="Overview",
                bounds=BoundingBox(x=140, y=10, width=100, height=30),
                visibility_fraction=1,
                actionable=True,
                destination_url="https://fixture.test/overview",
            ),
        ),
    )


def _noticed_state(snapshot: ViewportSnapshot) -> AttentionState:
    state = AttentionState.initial(
        Budget(max_steps=5, max_observations=3, max_interactions=2, timeout_seconds=10),
        confidence=0.5,
        frustration=0,
    )
    return state.after_observation(
        ProgressiveObservation.from_snapshot(snapshot, newly_revealed_ids=("target",))
    )


def _forbidden_values() -> dict[str, str]:
    return {
        "fixture-selector": "button[data-testid='invite']",
        "data-testid": "target-test-id",
        "private-control-path": "/__control/state/session-1",
        "destination-url": "https://fixture.test/private",
        "fixture-state-key": "invited_email",
        "sensitive-fixture-value": "demo-secret@example.test",
        "api-key": "api-key-123",
        "execution-token": "private-token",
    }


def _assert_no_leaks(
    event_id: str, value: object, forbidden_values: dict[str, str]
) -> None:
    violations: list[tuple[str, str]] = []

    def visit(current: object, path: str) -> None:
        if isinstance(current, dict):
            for key, item in current.items():
                visit(key, f"{path}.<key>" if path else "<key>")
                visit(item, f"{path}.{key}" if path else str(key))
        elif isinstance(current, (list, tuple)):
            for index, item in enumerate(current):
                visit(item, f"{path}[{index}]")
        elif isinstance(current, str):
            lowered = current.lower()
            for label, forbidden in forbidden_values.items():
                if forbidden.lower() in lowered:
                    violations.append((path, label))

    visit(value, "")
    if violations:
        path, label = violations[0]
        raise AssertionError(f"{event_id} path={path} matched={label}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_model_requests_and_persona_observations_are_leak_free() -> None:
    client = _RecordingModelClient()
    snapshot = _snapshot()
    state = _noticed_state(snapshot)
    observation = ProgressiveObservation.from_snapshot(
        snapshot, newly_revealed_ids=("target",)
    )

    await StructuredCoarseScentEvaluator(client, model="scent-model").evaluate(
        "Find invite", snapshot
    )
    await StructuredFullScentEvaluator(client, model="scent-model").evaluate(
        "Find invite", state, snapshot
    )
    await StructuredCognitiveAgent(client, model="cognitive-model").decide(
        "Find invite", observation
    )

    forbidden = _forbidden_values()
    for index, request in enumerate(client.requests, start=1):
        _assert_no_leaks(f"model-request-{index}", request, forbidden)
    _assert_no_leaks("persona-observation-1", asdict(observation), forbidden)

    assert [request["role"] for request in client.requests] == [
        ModelRole.COARSE_SCENT.value,
        ModelRole.FULL_SCENT.value,
        ModelRole.COGNITIVE.value,
    ]
    assert [request["model"] for request in client.requests] == [
        "scent-model",
        "scent-model",
        "cognitive-model",
    ]
    assert client.requests[-1]["schema"] == CognitiveModelResponse.__name__


def test_leakage_failure_names_event_and_redacted_field_path() -> None:
    with pytest.raises(
        AssertionError,
        match=r"event-42 path=messages\[0\]\.content matched=api-key",
    ):
        _assert_no_leaks(
            "event-42",
            {"messages": [{"content": "api-key-123"}]},
            _forbidden_values(),
        )
