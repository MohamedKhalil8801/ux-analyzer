from __future__ import annotations

import os
from typing import Protocol, cast

import httpx
import pytest

from ux_analyzer.adapters.openai import (
    ModelConfigurationError,
    OpenAICompatibleSettings,
    create_structured_model_client,
)
from ux_analyzer.domain.attention import AttentionState, ProgressiveObservation
from ux_analyzer.domain.benchmark import Budget
from ux_analyzer.domain.interface import BoundingBox, ElementSnapshot, ViewportSnapshot
from ux_analyzer.ports.models import ModelCallRecord, ModelRole, StructuredModelClient
from ux_analyzer.providers.cognitive import CognitiveDecision, StructuredCognitiveAgent
from ux_analyzer.providers.scent import (
    StructuredCoarseScentEvaluator,
    StructuredFullScentEvaluator,
)


class _RecordedStructuredModelClient(StructuredModelClient, Protocol):
    @property
    def records(self) -> tuple[ModelCallRecord, ...]: ...


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
    try:
        settings = OpenAICompatibleSettings.from_env()
    except ModelConfigurationError as error:
        if str(error).startswith("missing model environment variables:"):
            pytest.skip(str(error))
        raise
    http_client = (
        httpx.AsyncClient(timeout=settings.timeout_seconds)
        if settings.mode == "api"
        else None
    )
    client = cast(
        _RecordedStructuredModelClient,
        create_structured_model_client(settings, http_client=http_client),
    )
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
        if http_client is not None:
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
