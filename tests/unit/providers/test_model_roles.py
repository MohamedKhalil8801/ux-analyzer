from __future__ import annotations

import json
from pathlib import Path

import pytest

from ux_analyzer.domain.attention import AttentionState, ProgressiveObservation
from ux_analyzer.domain.benchmark import Budget
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    PrivateExecutionReference,
    ViewportSnapshot,
)
from ux_analyzer.ports.models import (
    CognitiveRunContext,
    ModelResponseValidationError,
    ModelRole,
)
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
            element_id = json.loads(content)["elements"][0]["element_id"]
            payload = {"scores": [{"element_id": element_id, "score": 0.9}]}
            return schema.model_validate(payload)
        if schema is FullScentResponse:
            element_id = json.loads(content)["elements"][0]["element_id"]
            payload = {"scores": [{"element_id": element_id, "score": 0.8}]}
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


def test_cognitive_prompt_requires_grounded_navigation() -> None:
    prompt = (
        Path(__file__).parents[3]
        / "src"
        / "ux_analyzer"
        / "prompts"
        / "cognitive-v2.txt"
    ).read_text(encoding="utf-8")

    assert "Do not infer unseen destinations" in prompt
    assert "unrelated navigation" in prompt


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


def test_role_manifests_use_selected_client_provider_metadata() -> None:
    class CodexRecordingClient(RecordingClient):
        endpoint_origin = "codex-cli"
        provider_id = "codex-cli"
        provider_version = "codex-cli"

    client = CodexRecordingClient()
    coarse = StructuredCoarseScentEvaluator(client, model="scent-model")
    full = StructuredFullScentEvaluator(client, model="scent-model")
    cognitive = StructuredCognitiveAgent(client, model="cognitive-model")

    assert coarse.manifest.provider_id == "codex-cli"
    assert coarse.manifest.provider_version == "codex-cli"
    assert full.manifest.provider_id == "codex-cli"
    assert full.manifest.provider_version == "codex-cli"
    assert cognitive.manifest.provider_id == "codex-cli"
    assert cognitive.manifest.provider_version == "codex-cli"


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
    assert [item["element_id"] for item in coarse_payload["elements"]] == [
        "e0",
        "e1",
    ]

    full_payload = json.loads(client.calls[1][3])
    assert [item["element_id"] for item in full_payload["elements"]] == ["e0"]
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


@pytest.mark.asyncio
async def test_unknown_coarse_scent_element_is_classified_model_failure() -> None:
    class UnknownElementClient(RecordingClient):
        async def complete(self, schema, messages, model, role):
            del messages, model, role
            return schema.model_validate(
                {"scores": [{"element_id": "unknown", "score": 0.7}]}
            )

    with pytest.raises(
        ModelResponseValidationError, match="unknown element ID"
    ) as failure:
        await StructuredCoarseScentEvaluator(
            UnknownElementClient(), model="scent-model"
        ).evaluate("Find invite", _snapshot())

    assert failure.value.role is ModelRole.COARSE_SCENT
    assert failure.value.response_summary == {"element_id": "unknown"}


@pytest.mark.asyncio
async def test_cognitive_inspect_without_element_is_classified_model_failure() -> None:
    class IncompleteActionClient(RecordingClient):
        async def complete(self, schema, messages, model, role):
            del messages, model, role
            return schema.model_validate(
                {"action": "inspect", "reason": "Inspect the target."}
            )

    observation = ProgressiveObservation.from_snapshot(
        _snapshot(), newly_revealed_ids=("target",)
    )
    with pytest.raises(
        ModelResponseValidationError, match="inspect action requires element_id"
    ) as failure:
        await StructuredCognitiveAgent(
            IncompleteActionClient(), model="cognitive-model"
        ).decide("Find invite", observation)

    assert failure.value.role is ModelRole.COGNITIVE
    assert failure.value.response_summary["action"] == "inspect"


@pytest.mark.asyncio
async def test_cognitive_unknown_element_is_classified_model_failure() -> None:
    class UnknownElementClient(RecordingClient):
        async def complete(self, schema, messages, model, role):
            del messages, model, role
            return schema.model_validate(
                {
                    "action": "interact",
                    "element_id": "unknown",
                    "reason": "Select it.",
                }
            )

    observation = ProgressiveObservation.from_snapshot(
        _snapshot(), newly_revealed_ids=("target",)
    )
    with pytest.raises(
        ModelResponseValidationError, match="unknown element ID"
    ) as failure:
        await StructuredCognitiveAgent(
            UnknownElementClient(), model="cognitive-model"
        ).decide("Find invite", observation)

    assert failure.value.role is ModelRole.COGNITIVE
    assert failure.value.response_summary["element_id"] == "unknown"


@pytest.mark.asyncio
async def test_cognitive_unconfigured_fixture_key_is_classified_model_failure() -> None:
    class UnknownFixtureClient(RecordingClient):
        async def complete(self, schema, messages, model, role):
            del messages, model, role
            return schema.model_validate(
                {
                    "action": "type-fixture",
                    "element_id": "target",
                    "fixture_key": "unknown_fixture",
                    "reason": "Enter it.",
                }
            )

    observation = ProgressiveObservation.from_snapshot(
        _snapshot(), newly_revealed_ids=("target",)
    )
    with pytest.raises(
        ModelResponseValidationError, match="unconfigured fixture key"
    ) as failure:
        await StructuredCognitiveAgent(
            UnknownFixtureClient(),
            model="cognitive-model",
            fixture_keys=("invite_email",),
        ).decide("Find invite", observation)

    assert failure.value.role is ModelRole.COGNITIVE
    assert failure.value.response_summary["fixture_key"] == "unknown_fixture"


@pytest.mark.asyncio
async def test_cognitive_payload_includes_safe_progress_and_persona_context() -> None:
    client = RecordingClient()
    agent = StructuredCognitiveAgent(
        client, model="cognitive-model", fixture_keys=("invite_email",)
    )
    agent.update_context(
        CognitiveRunContext(
            viewport_id="viewport-1",
            previous_action={
                "kind": "type-fixture",
                "element_id": "target",
                "fixture_key": "invite_email",
            },
            previous_action_result={"succeeded": True, "state_changed": True},
            completed_fixture_keys=("invite_email",),
            fixture_input_complete=True,
            working_memory_capacity=3,
            confidence=0.6,
            frustration=0.1,
            abandonment_threshold=0.8,
            attention_temperature=1.2,
        )
    )
    observation = ProgressiveObservation.from_snapshot(
        _snapshot(), newly_revealed_ids=("target",)
    )

    await agent.decide("Find invite", observation)

    payload = json.loads(client.calls[-1][3])
    assert payload["current_viewport_id"] == "viewport-1"
    assert payload["previous_action_result"]["succeeded"] is True
    assert payload["completed_fixture_keys"] == ["invite_email"]
    assert payload["fixture_input_complete"] is True
    assert payload["available_controls"][0]["element_id"] == "target"
    assert payload["persona_behavior"] == {
        "abandonment_threshold": 0.8,
        "attention_temperature": 1.2,
        "confidence": 0.6,
        "frustration": 0.1,
        "working_memory_capacity": 3,
    }
    serialized = json.dumps(payload).lower()
    for forbidden in (
        "private-token",
        "data-testid",
        "person@example.com",
        "verifier",
        "prominence",
        "scent_score",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_action", "expected_kind"),
    (("click", "interact"), ("type", "type-fixture")),
)
async def test_flat_cognitive_provider_response_normalizes_to_domain_action(
    provider_action: str,
    expected_kind: str,
) -> None:
    class AliasResponseClient(RecordingClient):
        async def complete(self, schema, messages, model, role):
            del messages, model, role
            payload = {
                "action": provider_action,
                "element_id": "target",
                "fixture_key": "invite_email" if provider_action == "type" else None,
                "reason": "Provider alias response.",
            }
            return schema.model_validate(payload)

    observation = ProgressiveObservation.from_snapshot(
        _snapshot(), newly_revealed_ids=("target",)
    )
    decision = await StructuredCognitiveAgent(
        AliasResponseClient(),
        model="cognitive-model",
        fixture_keys=(("invite_email",) if provider_action == "type" else ()),
    ).decide("Find invite", observation)

    assert decision.action.kind == expected_kind
    assert decision.action.element_id == "target"
