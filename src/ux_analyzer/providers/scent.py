"""Structured coarse and full scent providers with separate contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ux_analyzer.domain.attention import AttentionState, CoarseScent, FullScent
from ux_analyzer.domain.interface import ElementRole, ElementSnapshot, ViewportSnapshot
from ux_analyzer.ports.models import (
    ChatMessage,
    ModelManifest,
    ModelResponseValidationError,
    ModelRole,
    StructuredModelClient,
)


class _RoleSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CoarseScentElement(_RoleSchema):
    """Glance-level fields available before an element is noticed."""

    element_id: str
    role: str
    label: str
    region_label: str | None = None
    actionable: bool


class CoarseScentRequest(_RoleSchema):
    goal: str
    elements: tuple[CoarseScentElement, ...]


class CoarseScentScore(_RoleSchema):
    element_id: str
    score: float = Field(ge=0, le=1)


class CoarseScentResponse(_RoleSchema):
    schema_version: ClassVar[str] = "scent-coarse-v1"

    scores: list[CoarseScentScore]


class FullScentElement(_RoleSchema):
    """Persona-visible fields for one already-noticed element."""

    element_id: str
    role: str
    label: str
    region_label: str | None = None
    actionable: bool
    disabled: bool


class FullScentRequest(_RoleSchema):
    goal: str
    elements: tuple[FullScentElement, ...]


class FullScentScore(_RoleSchema):
    element_id: str
    score: float = Field(ge=0, le=1)


class FullScentResponse(_RoleSchema):
    schema_version: ClassVar[str] = "scent-full-v1"

    scores: list[FullScentScore]


def _prompt(filename: str) -> str:
    path = Path(__file__).resolve().parents[1] / "prompts" / filename
    return path.read_text(encoding="utf-8").strip()


def _messages(prompt: str, payload: BaseModel) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(role="system", content=prompt),
        ChatMessage(
            role="user",
            content=json.dumps(
                payload.model_dump(mode="json"),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )


def _region_label(snapshot: ViewportSnapshot, element: ElementSnapshot) -> str | None:
    if element.region_id is None:
        return None
    for region in snapshot.regions:
        if region.id == element.region_id:
            return region.label
    return None


def _manifest(
    client: StructuredModelClient,
    role: ModelRole,
    model: str,
    prompt_version: str,
    schema_version: str,
) -> ModelManifest:
    return ModelManifest(
        provider_id=str(getattr(client, "provider_id", "openai-compatible-structured")),
        role=role,
        model_id=model,
        endpoint_origin=client.endpoint_origin,
        prompt_version=prompt_version,
        schema_version=schema_version,
        provider_version=str(
            getattr(client, "provider_version", "openai-compatible-v1")
        ),
    )


class StructuredCoarseScentEvaluator:
    """Evaluate pre-notice scent from glance cues only."""

    role = ModelRole.COARSE_SCENT
    prompt_version = "scent-coarse-v1"

    def __init__(self, client: StructuredModelClient, *, model: str) -> None:
        self.client = client
        self.model = model

    @property
    def manifest(self) -> ModelManifest:
        return _manifest(
            self.client,
            self.role,
            self.model,
            self.prompt_version,
            CoarseScentResponse.schema_version,
        )

    async def evaluate(
        self, goal: str, snapshot: ViewportSnapshot
    ) -> tuple[CoarseScent, ...]:
        model_elements = tuple(
            (f"e{index}", element) for index, element in enumerate(snapshot.elements)
        )
        payload = CoarseScentRequest(
            goal=goal,
            elements=tuple(
                CoarseScentElement(
                    element_id=model_id,
                    role=ElementRole(element.role).value,
                    label=element.label,
                    region_label=_region_label(snapshot, element),
                    actionable=element.actionable,
                )
                for model_id, element in model_elements
            ),
        )
        response = await self.client.complete(
            CoarseScentResponse,
            _messages(_prompt("scent-coarse-v1.txt"), payload),
            model=self.model,
            role=self.role,
        )
        known = dict(model_elements)
        seen: set[str] = set()
        results: list[CoarseScent] = []
        for item in response.scores:
            if item.element_id in seen:
                raise ModelResponseValidationError(
                    self.role,
                    "response contains duplicate element ID",
                    response_summary={"element_id": item.element_id},
                )
            element = known.get(item.element_id)
            if element is None:
                raise ModelResponseValidationError(
                    self.role,
                    "response references unknown element ID",
                    response_summary={"element_id": item.element_id},
                )
            seen.add(item.element_id)
            results.append(CoarseScent.from_element(snapshot, element, item.score))
        return tuple(results)


class StructuredFullScentEvaluator:
    """Evaluate full scent only after notice and only for noticed elements."""

    role = ModelRole.FULL_SCENT
    prompt_version = "scent-full-v1"

    def __init__(self, client: StructuredModelClient, *, model: str) -> None:
        self.client = client
        self.model = model

    @property
    def manifest(self) -> ModelManifest:
        return _manifest(
            self.client,
            self.role,
            self.model,
            self.prompt_version,
            FullScentResponse.schema_version,
        )

    async def evaluate(
        self, goal: str, state: AttentionState, snapshot: ViewportSnapshot
    ) -> tuple[FullScent, ...]:
        if state.current_viewport_id is None:
            raise ValueError("full scent requires current viewport")
        if state.current_viewport_id != snapshot.id:
            raise ValueError("full scent targets stale viewport")
        noticed = [
            element for element in snapshot.elements if element.id in state.noticed_ids
        ]
        model_elements = tuple(
            (f"e{index}", element) for index, element in enumerate(noticed)
        )
        payload = FullScentRequest(
            goal=goal,
            elements=tuple(
                FullScentElement(
                    element_id=model_id,
                    role=ElementRole(element.role).value,
                    label=element.label,
                    region_label=_region_label(snapshot, element),
                    actionable=element.actionable,
                    disabled=element.disabled,
                )
                for model_id, element in model_elements
            ),
        )
        response = await self.client.complete(
            FullScentResponse,
            _messages(_prompt("scent-full-v1.txt"), payload),
            model=self.model,
            role=self.role,
        )
        noticed_by_model_id = dict(model_elements)
        seen: set[str] = set()
        results: list[FullScent] = []
        for item in response.scores:
            if item.element_id in seen:
                raise ModelResponseValidationError(
                    self.role,
                    "response contains duplicate element ID",
                    response_summary={"element_id": item.element_id},
                )
            element = noticed_by_model_id.get(item.element_id)
            if element is None:
                raise ModelResponseValidationError(
                    self.role,
                    "response references unknown or unnoticed element ID",
                    response_summary={"element_id": item.element_id},
                )
            seen.add(item.element_id)
            results.append(FullScent.for_element(state, element.id, item.score))
        return tuple(results)


CoarseScentProvider = StructuredCoarseScentEvaluator
FullScentProvider = StructuredFullScentEvaluator
