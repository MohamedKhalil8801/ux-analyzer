"""Structured qualitative cognitive agent provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from ux_analyzer.domain.attention import PersonaObservation
from ux_analyzer.domain.interface import PersonaVisibleElement
from ux_analyzer.ports.models import (
    ChatMessage,
    ModelManifest,
    ModelRole,
    StructuredModelClient,
)


class _RoleSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CognitiveElement(_RoleSchema):
    """Only qualitative persona-visible fields used for action choice."""

    element_id: str
    role: str
    label: str
    actionable: bool
    disabled: bool
    region_label: str | None = None


class CognitiveObservation(_RoleSchema):
    goal: str
    fixture_keys: tuple[str, ...]
    newly_revealed_elements: tuple[CognitiveElement, ...]
    remembered_elements: tuple[CognitiveElement, ...]
    region_label: str | None = None


class InspectAction(_RoleSchema):
    kind: Literal["inspect"] = "inspect"
    element_id: str


class InteractAction(_RoleSchema):
    kind: Literal["interact"] = "interact"
    element_id: str


class TypeFixtureAction(_RoleSchema):
    kind: Literal["type-fixture"] = "type-fixture"
    element_id: str
    fixture_key: str


class ScrollAction(_RoleSchema):
    kind: Literal["scroll"] = "scroll"
    direction: Literal["up", "down"] = "down"


class WaitAction(_RoleSchema):
    kind: Literal["wait"] = "wait"


class BackAction(_RoleSchema):
    kind: Literal["back"] = "back"


class AbandonAction(_RoleSchema):
    kind: Literal["abandon"] = "abandon"
    reason: str = Field(min_length=1)


CognitiveAction = Annotated[
    InspectAction
    | InteractAction
    | TypeFixtureAction
    | ScrollAction
    | WaitAction
    | BackAction
    | AbandonAction,
    Field(discriminator="kind"),
]


class CognitiveDecision(_RoleSchema):
    schema_version: ClassVar[str] = "cognitive-v1"

    action: CognitiveAction
    reason: str = Field(min_length=1)


def _prompt() -> str:
    path = Path(__file__).resolve().parents[1] / "prompts" / "cognitive-v1.txt"
    return path.read_text(encoding="utf-8").strip()


def _element_payload(
    element: PersonaVisibleElement, region_label: str | None
) -> CognitiveElement:
    return CognitiveElement(
        element_id=element.id,
        role=element.role.value,
        label=element.label,
        actionable=element.actionable,
        disabled=element.disabled,
        region_label=region_label,
    )


def _manifest(client: StructuredModelClient, model: str) -> ModelManifest:
    return ModelManifest(
        provider_id="openai-compatible-structured",
        role=ModelRole.COGNITIVE,
        model_id=model,
        endpoint_origin=client.endpoint_origin,
        prompt_version="cognitive-v1",
        schema_version=CognitiveDecision.schema_version,
    )


class StructuredCognitiveAgent:
    """Choose one qualitative action from persona-visible observations."""

    role = ModelRole.COGNITIVE
    prompt_version = "cognitive-v1"

    def __init__(
        self,
        client: StructuredModelClient,
        *,
        model: str,
        fixture_keys: tuple[str, ...] = (),
    ) -> None:
        self.client = client
        self.model = model
        keys = tuple(fixture_keys)
        if any(not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("fixture keys must be unique non-empty names")
        self.fixture_keys = keys

    @property
    def manifest(self) -> ModelManifest:
        return _manifest(self.client, self.model)

    async def decide(
        self, goal: str, observation: PersonaObservation
    ) -> CognitiveDecision:
        region_label = (
            observation.region_context.label
            if observation.region_context is not None
            else None
        )
        payload = CognitiveObservation(
            goal=goal,
            fixture_keys=self.fixture_keys,
            newly_revealed_elements=tuple(
                _element_payload(element, region_label)
                for element in observation.newly_revealed_elements
            ),
            remembered_elements=tuple(
                _element_payload(element, region_label)
                for element in observation.remembered_elements
            ),
            region_label=region_label,
        )
        messages = (
            ChatMessage(role="system", content=_prompt()),
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
        return await self.client.complete(
            CognitiveDecision,
            messages,
            model=self.model,
            role=self.role,
        )
