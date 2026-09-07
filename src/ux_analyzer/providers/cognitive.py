"""Structured qualitative cognitive agent provider."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ux_analyzer.domain.attention import PersonaObservation
from ux_analyzer.domain.interface import PersonaVisibleElement
from ux_analyzer.ports.models import (
    ChatMessage,
    CognitiveRunContext,
    ModelManifest,
    ModelResponseValidationError,
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
    current_viewport_id: str | None = None
    previous_action: dict[str, object] | None = None
    previous_action_result: dict[str, object] | None = None
    completed_fixture_keys: tuple[str, ...] = ()
    fixture_input_complete: bool = False
    available_controls: tuple[CognitiveElement, ...] = ()
    persona_behavior: dict[str, float | int] = Field(default_factory=dict)


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


class CompleteAction(_RoleSchema):
    kind: Literal["complete"] = "complete"


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
    | CompleteAction
    | AbandonAction,
    Field(discriminator="kind"),
]


class CognitiveDecision(_RoleSchema):
    schema_version: ClassVar[str] = "cognitive-v1"

    action: CognitiveAction
    reason: str = Field(min_length=1)


class CognitiveModelResponse(_RoleSchema):
    """Provider-facing flat response accepted by less strict model APIs."""

    schema_version: ClassVar[str] = "cognitive-v1"

    action: str | None = None
    element_id: str | None = None
    fixture_key: str | None = None
    direction: str | None = None
    reason: str | None = None


def _prompt() -> str:
    path = Path(__file__).resolve().parents[1] / "prompts" / "cognitive-v3.txt"
    return path.read_text(encoding="utf-8").strip()


def _element_payload(
    element: PersonaVisibleElement,
    region_label: str | None,
    *,
    model_element_id: str | None = None,
) -> CognitiveElement:
    return CognitiveElement(
        element_id=model_element_id or element.id,
        role=element.role.value,
        label=element.label,
        actionable=element.actionable,
        disabled=element.disabled,
        region_label=region_label,
    )


def _model_element_aliases(
    elements: tuple[PersonaVisibleElement, ...],
) -> dict[str, str]:
    reserved_ids = {element.id for element in elements}
    aliases: dict[str, str] = {}
    next_index = 0
    for element in elements:
        if len(element.id) <= 32:
            continue
        while f"e{next_index}" in reserved_ids or f"e{next_index}" in aliases.values():
            next_index += 1
        aliases[element.id] = f"e{next_index}"
        next_index += 1
    return aliases


def _model_action(
    action: Mapping[str, object] | None,
    aliases: dict[str, str],
) -> dict[str, object] | None:
    if action is None:
        return None
    result = dict(action)
    element_id = result.get("element_id")
    if isinstance(element_id, str):
        result["element_id"] = aliases.get(element_id, element_id)
    return result


def _normalize_model_response(response: CognitiveModelResponse) -> CognitiveDecision:
    action_name = (response.action or "").strip().lower()
    action_name = {
        "click": "interact",
        "tap": "interact",
        "press": "interact",
        "input": "type-fixture",
        "type": "type-fixture",
    }.get(action_name, action_name)

    action_data: dict[str, object] = {"kind": action_name}
    if (
        action_name in {"inspect", "interact", "type-fixture"}
        and response.element_id is not None
    ):
        action_data["element_id"] = response.element_id
    if action_name == "type-fixture" and response.fixture_key is not None:
        action_data["fixture_key"] = response.fixture_key
    if action_name == "scroll" and response.direction is not None:
        action_data["direction"] = response.direction
    if action_name == "abandon" and response.reason is not None:
        action_data["reason"] = response.reason

    reason = response.reason or "Provider-compatible cognitive decision"
    try:
        return CognitiveDecision.model_validate(
            {"action": action_data, "reason": reason}
        )
    except ValidationError as error:
        if action_name in {"inspect", "interact"} and response.element_id is None:
            failure = f"{action_name} action requires element_id"
        elif action_name == "type-fixture" and response.element_id is None:
            failure = "type-fixture action requires element_id"
        elif action_name == "type-fixture" and response.fixture_key is None:
            failure = "type-fixture action requires fixture_key"
        elif not action_name:
            failure = "response requires a valid action kind"
        else:
            failure = "response action is incomplete or invalid"
        raise ModelResponseValidationError(
            ModelRole.COGNITIVE,
            failure,
            response_summary=response.model_dump(mode="json"),
        ) from error


def _manifest(client: StructuredModelClient, model: str) -> ModelManifest:
    return ModelManifest(
        provider_id=str(getattr(client, "provider_id", "openai-compatible-structured")),
        role=ModelRole.COGNITIVE,
        model_id=model,
        endpoint_origin=client.endpoint_origin,
        prompt_version="cognitive-v3",
        schema_version=CognitiveDecision.schema_version,
        provider_version=str(
            getattr(client, "provider_version", "openai-compatible-v1")
        ),
    )


class StructuredCognitiveAgent:
    """Choose one qualitative action from persona-visible observations."""

    role = ModelRole.COGNITIVE
    prompt_version = "cognitive-v3"

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
        self._run_context: CognitiveRunContext | None = None

    def update_context(self, context: CognitiveRunContext) -> None:
        self._run_context = context

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
        visible_elements = (
            *observation.newly_revealed_elements,
            *observation.remembered_elements,
        )
        aliases = _model_element_aliases(visible_elements)
        reverse_aliases = {alias: element_id for element_id, alias in aliases.items()}
        payload = CognitiveObservation(
            goal=goal,
            fixture_keys=self.fixture_keys,
            newly_revealed_elements=tuple(
                _element_payload(
                    element,
                    region_label,
                    model_element_id=aliases.get(element.id),
                )
                for element in observation.newly_revealed_elements
            ),
            remembered_elements=tuple(
                _element_payload(
                    element,
                    region_label,
                    model_element_id=aliases.get(element.id),
                )
                for element in observation.remembered_elements
            ),
            region_label=region_label,
            current_viewport_id=(
                self._run_context.viewport_id if self._run_context is not None else None
            ),
            previous_action=(
                _model_action(self._run_context.previous_action, aliases)
                if self._run_context is not None
                and self._run_context.previous_action is not None
                else None
            ),
            previous_action_result=(
                _model_action(self._run_context.previous_action_result, aliases)
                if self._run_context is not None
                and self._run_context.previous_action_result is not None
                else None
            ),
            completed_fixture_keys=(
                self._run_context.completed_fixture_keys
                if self._run_context is not None
                else ()
            ),
            fixture_input_complete=(
                self._run_context.fixture_input_complete
                if self._run_context is not None
                else False
            ),
            available_controls=tuple(
                _element_payload(
                    element,
                    region_label,
                    model_element_id=aliases.get(element.id),
                )
                for element in (
                    *observation.newly_revealed_elements,
                    *observation.remembered_elements,
                )
                if element.actionable and not element.disabled
            ),
            persona_behavior=(
                {
                    "working_memory_capacity": self._run_context.working_memory_capacity,
                    "confidence": self._run_context.confidence,
                    "frustration": self._run_context.frustration,
                    "abandonment_threshold": self._run_context.abandonment_threshold,
                    "attention_temperature": self._run_context.attention_temperature,
                }
                if self._run_context is not None
                else {}
            ),
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
        response = await self.client.complete(
            CognitiveModelResponse,
            messages,
            model=self.model,
            role=self.role,
        )
        if response.element_id is not None:
            response = response.model_copy(
                update={
                    "element_id": reverse_aliases.get(
                        response.element_id, response.element_id
                    )
                }
            )
        decision = _normalize_model_response(response)
        visible_ids = {
            element.id
            for element in (
                *observation.newly_revealed_elements,
                *observation.remembered_elements,
            )
        }
        element_id = getattr(decision.action, "element_id", None)
        if isinstance(element_id, str) and element_id not in visible_ids:
            raise ModelResponseValidationError(
                ModelRole.COGNITIVE,
                f"response references unknown element ID {element_id!r}",
                response_summary=response.model_dump(mode="json"),
            )
        fixture_key = getattr(decision.action, "fixture_key", None)
        if isinstance(fixture_key, str) and fixture_key not in self.fixture_keys:
            raise ModelResponseValidationError(
                ModelRole.COGNITIVE,
                f"response references unconfigured fixture key {fixture_key!r}",
                response_summary=response.model_dump(mode="json"),
            )
        return decision
