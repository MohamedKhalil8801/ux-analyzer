from __future__ import annotations

import json

import pytest

from ux_analyzer.domain.attention import AttentionState, ProgressiveObservation
from ux_analyzer.domain.benchmark import Budget
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    PrivateExecutionReference,
    ViewportSnapshot,
)
from ux_analyzer.ports.models import ModelRole
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


class RecordingClient:
    endpoint_origin = "https://llm.example.test"

    def __init__(self) -> None:
        self.calls: list[tuple[type[object], str, ModelRole, str]] = []

    async def complete(
        self,
        schema: type[object],
        messages: object,
        model: str,
        role: ModelRole,
    ) -> object:
        user_message = list(messages)[-1]
        content = user_message.content
        self.calls.append((schema, model, role, content))
        if schema is CoarseScentResponse:
            payload = {"scores": [{"element_id": "target", "score": 0.9}]}
            return schema.model_validate(payload)
        if schema is FullScentResponse:
            payload = {"scores": [{"element_id": "target", "score": 0.8}]}
            return schema.model_validate(payload)
        if schema is CognitiveModelResponse:
            payload = {
                "action": "inspect",
                "element_id": "target",
                "reason": "Visible control matches goal.",
            }
            return schema.model_validate(payload)
        raise AssertionError(f"unexpected schema: {schema!r}")


def _snapshot() -> ViewportSnapshot:
    return ViewportSnapshot(
        id="viewport-1",
        elements=(
            ElementSnapshot(
                id="target",
                role="button",
                label="Invite teammate",
                bounds=BoundingBox(x=1, y=2, width=10, height=10),
                visibility_fraction=1.0,
                actionable=True,
                provider_id="fixture",
                execution_reference=PrivateExecutionReference(
                    provider_id="fixture",
                    viewport_id="viewport-1",
                    token="private-token",
                ),
                selector="[data-testid='target']",
                test_id="target",
                hidden_label="private hidden label",
                destination_url="https://fixture.invalid/private",
            ),
            ElementSnapshot(
                id="unnoticed",
                role="link",
                label="Private destination",
                bounds=BoundingBox(x=20, y=2, width=10, height=10),
                visibility_fraction=1.0,
                actionable=True,
                destination_url="https://fixture.invalid/unnoticed",
            ),
        ),
    )


def _noticed_state(snapshot: ViewportSnapshot) -> AttentionState:
    state = AttentionState.initial(
        Budget(max_steps=5, max_observations=3, max_interactions=2, timeout_seconds=10),
        confidence=0.5,
        frustration=0.0,
    )
    observation = ProgressiveObservation.from_snapshot(
        snapshot, newly_revealed_ids=("target",)
    )
    return state.after_observation(observation)


def test_model_environment_loads_separate_role_models_without_logging_key() -> None:
    from ux_analyzer.adapters.openai import OpenAICompatibleSettings

    settings = OpenAICompatibleSettings.from_env(
        {
            "UXA_LLM_BASE_URL": "https://llm.example.test/v1/",
            "UXA_LLM_API_KEY": "secret-key",
            "UXA_SCENT_MODEL": "scent-model",
            "UXA_COGNITIVE_MODEL": "cognitive-model",
        }
    )

    assert settings.base_url == "https://llm.example.test/v1"
    assert settings.scent_model == "scent-model"
    assert settings.cognitive_model == "cognitive-model"
    assert "secret-key" not in repr(settings)


@pytest.mark.asyncio
async def test_role_providers_keep_models_and_payloads_separate() -> None:
    client = RecordingClient()
    snapshot = _snapshot()

    coarse = StructuredCoarseScentEvaluator(client, model="scent-model")
    full = StructuredFullScentEvaluator(client, model="scent-model")
    cognitive = StructuredCognitiveAgent(
        client,
        model="cognitive-model",
        fixture_keys=("invite_email", "totp_code"),
    )

    await coarse.evaluate("Find invite", snapshot)
    await full.evaluate("Find invite", _noticed_state(snapshot), snapshot)
    observation = ProgressiveObservation.from_snapshot(
        snapshot, newly_revealed_ids=("target",)
    )
    await cognitive.decide("Find invite", observation)

    assert [call[2] for call in client.calls] == [
        ModelRole.COARSE_SCENT,
        ModelRole.FULL_SCENT,
        ModelRole.COGNITIVE,
    ]
    assert [call[1] for call in client.calls] == [
        "scent-model",
        "scent-model",
        "cognitive-model",
    ]

    coarse_payload = json.loads(client.calls[0][3])
    assert set(coarse_payload["elements"][0]) == {
        "element_id",
        "role",
        "label",
        "region_label",
        "actionable",
    }

    full_payload = json.loads(client.calls[1][3])
    assert [item["element_id"] for item in full_payload["elements"]] == ["target"]
    assert "unnoticed" not in client.calls[1][3]

    cognitive_payload = json.loads(client.calls[2][3])
    assert "unnoticed" not in client.calls[2][3]
    assert cognitive_payload["fixture_keys"] == ["invite_email", "totp_code"]
    cognitive_text = client.calls[2][3].lower()
    for forbidden in (
        "data-testid",
        "private-token",
        "private hidden label",
        "fixture.invalid",
        "score",
        "prominence",
    ):
        assert forbidden not in cognitive_text
    assert set(cognitive_payload["newly_revealed_elements"][0]) == {
        "element_id",
        "role",
        "label",
        "actionable",
        "disabled",
        "region_label",
    }


@pytest.mark.asyncio
async def test_full_scent_rejects_snapshot_without_current_notice_state() -> None:
    client = RecordingClient()
    provider = StructuredFullScentEvaluator(client, model="scent-model")
    snapshot = _snapshot()
    state = AttentionState.initial(
        Budget(max_steps=5, max_observations=3, max_interactions=2, timeout_seconds=10),
        confidence=0.5,
        frustration=0.0,
    )

    with pytest.raises(ValueError, match="current viewport"):
        await provider.evaluate("Find invite", state, snapshot)
