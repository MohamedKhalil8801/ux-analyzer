from __future__ import annotations

import os

import httpx
import pytest

from ux_analyzer.adapters.openai import (
    OpenAICompatibleSettings,
    OpenAICompatibleStructuredClient,
)
from ux_analyzer.domain.attention import AttentionState, ProgressiveObservation
from ux_analyzer.domain.benchmark import Budget
from ux_analyzer.domain.interface import BoundingBox, ElementSnapshot, ViewportSnapshot
from ux_analyzer.ports.models import ModelRole
from ux_analyzer.providers.cognitive import CognitiveDecision, StructuredCognitiveAgent
from ux_analyzer.providers.scent import (
    StructuredCoarseScentEvaluator,
    StructuredFullScentEvaluator,
)


def _snapshot() -> ViewportSnapshot:
    return ViewportSnapshot(
        id="live-viewport",
        elements=(
            ElementSnapshot(
                id="target",
                role="button",
                label="Invite teammate",
                bounds=BoundingBox(x=10, y=10, width=100, height=30),
                visibility_fraction=1,
                actionable=True,
            ),
        ),
    )


@pytest.mark.live
@pytest.mark.asyncio
async def test_openai_compatible_endpoint_supports_all_structured_roles() -> None:
    if os.environ.get("UXA_RUN_LIVE_TESTS") != "1":
        pytest.skip("set UXA_RUN_LIVE_TESTS=1 to run live endpoint tests")
    required = (
        "UXA_LLM_BASE_URL",
        "UXA_LLM_API_KEY",
        "UXA_SCENT_MODEL",
        "UXA_COGNITIVE_MODEL",
    )
    missing = tuple(name for name in required if not os.environ.get(name))
    if missing:
        pytest.skip("missing live model environment variables: " + ", ".join(missing))
    settings = OpenAICompatibleSettings.from_env()
    http_client = httpx.AsyncClient(timeout=settings.timeout_seconds)
    client = OpenAICompatibleStructuredClient(settings, http_client=http_client)
    snapshot = _snapshot()
    state = AttentionState.initial(
        Budget(max_steps=5, max_observations=3, max_interactions=2, timeout_seconds=10),
        confidence=0.5,
        frustration=0,
    ).after_observation(
        ProgressiveObservation.from_snapshot(snapshot, newly_revealed_ids=("target",))
    )
    try:
        coarse = await StructuredCoarseScentEvaluator(
            client, model=settings.scent_model
        ).evaluate("Find invite", snapshot)
        full = await StructuredFullScentEvaluator(
            client, model=settings.scent_model
        ).evaluate("Find invite", state, snapshot)
        cognitive = await StructuredCognitiveAgent(
            client, model=settings.cognitive_model
        ).decide(
            "Find invite",
            ProgressiveObservation.from_snapshot(
                snapshot, newly_revealed_ids=("target",)
            ),
        )
    finally:
        await http_client.aclose()

    assert coarse and 0 <= coarse[0].score <= 1
    assert full and 0 <= full[0].score <= 1
    assert isinstance(cognitive, CognitiveDecision)
    assert [record.role for record in client.records] == [
        ModelRole.COARSE_SCENT,
        ModelRole.FULL_SCENT,
        ModelRole.COGNITIVE,
    ]
    assert all(
        record.endpoint_origin == settings.endpoint_origin for record in client.records
    )
    assert all(record.request and record.response for record in client.records)
