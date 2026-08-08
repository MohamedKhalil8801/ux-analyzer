"""Immutable attention state, observations, scent records, and actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ux_analyzer.domain.benchmark import Budget
from ux_analyzer.domain.interface import (
    ElementRole,
    ElementSnapshot,
    PersonaVisibleElement,
    RegionSnapshot,
    ViewportSnapshot,
)


@dataclass(frozen=True, slots=True)
class AttentionBudgets:
    """Remaining runtime resources; zero is valid after consumption."""

    steps: int
    observations: int
    interactions: int

    def __post_init__(self) -> None:
        for name, value in (
            ("steps", self.steps),
            ("observations", self.observations),
            ("interactions", self.interactions),
        ):
            if value < 0:
                raise ValueError(f"remaining {name} must not be negative")

    def consume_steps(self, amount: int = 1) -> AttentionBudgets:
        if amount <= 0 or amount > self.steps:
            raise ValueError("step budget exhausted or amount invalid")
        return AttentionBudgets(
            steps=self.steps - amount,
            observations=self.observations,
            interactions=self.interactions,
        )

    def consume_observation(self) -> AttentionBudgets:
        if self.observations <= 0:
            raise ValueError("observation budget exhausted")
        return AttentionBudgets(
            steps=self.steps,
            observations=self.observations - 1,
            interactions=self.interactions,
        )

    def consume_interaction(self) -> AttentionBudgets:
        if self.interactions <= 0:
            raise ValueError("interaction budget exhausted")
        return AttentionBudgets(
            steps=self.steps,
            observations=self.observations,
            interactions=self.interactions - 1,
        )


@dataclass(frozen=True, slots=True)
class RememberedElement:
    """Persona-visible element retained in simulated working memory."""

    element_id: str
    viewport_id: str
    label: str


@dataclass(frozen=True, slots=True)
class PersonaVisibleRegion:
    """Safe projection of captured region context."""

    id: str
    label: str

    @classmethod
    def from_snapshot(cls, region: RegionSnapshot) -> PersonaVisibleRegion:
        return cls(
            id=region.id,
            label=(
                region.rendered_label
                if region.rendered_label is not None
                else region.label
            ),
        )

    def model_dump(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label}


@dataclass(frozen=True, slots=True)
class ProgressiveObservation:
    """Bounded persona-visible observation from one viewport capture."""

    viewport_id: str
    newly_revealed_elements: tuple[PersonaVisibleElement, ...]
    remembered_elements: tuple[PersonaVisibleElement, ...] = ()
    region_context: PersonaVisibleRegion | None = None

    def __post_init__(self) -> None:
        newly_revealed = tuple(self.newly_revealed_elements)
        remembered = tuple(self.remembered_elements)
        if not 1 <= len(newly_revealed) <= 3:
            raise ValueError("observation must reveal 1 to 3 elements")
        ids = [element.id for element in newly_revealed]
        if len(ids) != len(set(ids)):
            raise ValueError("observation contains duplicate newly revealed element")
        remembered_ids = [element.id for element in remembered]
        if len(remembered_ids) != len(set(remembered_ids)):
            raise ValueError("observation contains duplicate remembered element")
        if set(ids) & set(remembered_ids):
            raise ValueError("newly revealed elements cannot already be remembered")
        object.__setattr__(self, "newly_revealed_elements", newly_revealed)
        object.__setattr__(self, "remembered_elements", remembered)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ViewportSnapshot,
        newly_revealed_ids: Sequence[str],
        remembered_ids: Sequence[str] = (),
        region_id: str | None = None,
    ) -> ProgressiveObservation:
        """Build observation only from elements in captured viewport."""

        element_by_id = {element.id: element for element in snapshot.elements}
        for element_id in (*newly_revealed_ids, *remembered_ids):
            if element_id not in element_by_id:
                raise ValueError(
                    f"observation element {element_id!r} not present in captured viewport"
                )
        region_context: PersonaVisibleRegion | None = None
        if region_id is not None:
            for region in snapshot.regions:
                if region.id == region_id:
                    region_context = PersonaVisibleRegion.from_snapshot(region)
                    break
            if region_context is None:
                raise ValueError(
                    f"region {region_id!r} not present in captured viewport"
                )
        return cls(
            viewport_id=snapshot.id,
            newly_revealed_elements=tuple(
                PersonaVisibleElement.from_snapshot(element_by_id[element_id])
                for element_id in newly_revealed_ids
                if (
                    element_by_id[element_id].visibility_fraction > 0
                    and (
                        element_by_id[element_id].rendered_text is None
                        or bool(element_by_id[element_id].rendered_text.strip())
                    )
                )
            ),
            remembered_elements=tuple(
                PersonaVisibleElement.from_snapshot(element_by_id[element_id])
                for element_id in remembered_ids
                if (
                    element_by_id[element_id].visibility_fraction > 0
                    and (
                        element_by_id[element_id].rendered_text is None
                        or bool(element_by_id[element_id].rendered_text.strip())
                    )
                )
            ),
            region_context=region_context,
        )


@dataclass(frozen=True, slots=True)
class CompleteObservation:
    """Complete persona-safe visible list used only by unrestricted policies."""

    viewport_id: str
    newly_revealed_elements: tuple[PersonaVisibleElement, ...]
    remembered_elements: tuple[PersonaVisibleElement, ...] = ()
    region_context: PersonaVisibleRegion | None = None

    def __post_init__(self) -> None:
        newly_revealed = tuple(self.newly_revealed_elements)
        remembered = tuple(self.remembered_elements)
        if not newly_revealed:
            raise ValueError("complete observation needs at least one visible element")
        ids = [element.id for element in newly_revealed]
        if len(ids) != len(set(ids)):
            raise ValueError("complete observation contains duplicate elements")
        remembered_ids = [element.id for element in remembered]
        if len(remembered_ids) != len(set(remembered_ids)):
            raise ValueError(
                "complete observation contains duplicate remembered element"
            )
        if set(ids) & set(remembered_ids):
            raise ValueError("complete elements cannot already be remembered")
        object.__setattr__(self, "newly_revealed_elements", newly_revealed)
        object.__setattr__(self, "remembered_elements", remembered)

    @classmethod
    def from_snapshot(cls, snapshot: ViewportSnapshot) -> CompleteObservation:
        return cls(
            viewport_id=snapshot.id,
            newly_revealed_elements=tuple(
                PersonaVisibleElement.from_snapshot(element)
                for element in snapshot.elements
                if element.visibility_fraction > 0
            ),
        )


type PersonaObservation = ProgressiveObservation | CompleteObservation


@dataclass(frozen=True, slots=True)
class AttentionRecoveryMiss:
    """Bounded miss count for one provider-stable element lineage."""

    provider_id: str
    lineage_id: str
    consecutive_misses: int

    def __post_init__(self) -> None:
        if not self.provider_id:
            raise ValueError("recovery provider_id must not be empty")
        if not self.lineage_id:
            raise ValueError("recovery lineage_id must not be empty")
        if self.consecutive_misses < 1:
            raise ValueError("recovery misses must be positive")

    @property
    def key(self) -> tuple[str, str]:
        return self.provider_id, self.lineage_id


@dataclass(frozen=True, slots=True)
class AttentionState:
    """All mutable-looking runtime attention state represented immutably."""

    noticed_ids: frozenset[str]
    inspected_ids: frozenset[str]
    focus_region: str | None
    budgets: AttentionBudgets
    memory: tuple[RememberedElement, ...]
    memory_capacity: int
    confidence: float
    frustration: float
    failed_candidates: frozenset[str]
    current_subgoal: str | None
    current_viewport_id: str | None
    current_observation_ids: frozenset[str] = frozenset()
    recovery_misses: tuple[AttentionRecoveryMiss, ...] = ()

    def __post_init__(self) -> None:
        if self.memory_capacity <= 0:
            raise ValueError("memory_capacity must be greater than zero")
        for name, value in (
            ("confidence", self.confidence),
            ("frustration", self.frustration),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        object.__setattr__(self, "noticed_ids", frozenset(self.noticed_ids))
        object.__setattr__(self, "inspected_ids", frozenset(self.inspected_ids))
        object.__setattr__(self, "failed_candidates", frozenset(self.failed_candidates))
        object.__setattr__(self, "memory", tuple(self.memory))
        object.__setattr__(
            self, "current_observation_ids", frozenset(self.current_observation_ids)
        )
        recovery_misses = tuple(self.recovery_misses)
        recovery_keys = [miss.key for miss in recovery_misses]
        if len(recovery_keys) != len(set(recovery_keys)):
            raise ValueError("attention recovery contains duplicate lineage keys")
        object.__setattr__(self, "recovery_misses", recovery_misses)
        if len(self.memory) > self.memory_capacity:
            raise ValueError("memory exceeds configured capacity")

    @classmethod
    def initial(
        cls,
        budget: Budget,
        confidence: float,
        frustration: float,
        *,
        memory_capacity: int = 100,
        current_subgoal: str | None = None,
    ) -> AttentionState:
        return cls(
            noticed_ids=frozenset(),
            inspected_ids=frozenset(),
            focus_region=None,
            budgets=AttentionBudgets(
                steps=budget.max_steps,
                observations=budget.max_observations,
                interactions=budget.max_interactions,
            ),
            memory=(),
            memory_capacity=memory_capacity,
            confidence=confidence,
            frustration=frustration,
            failed_candidates=frozenset(),
            current_subgoal=current_subgoal,
            current_viewport_id=None,
            current_observation_ids=frozenset(),
            recovery_misses=(),
        )

    @property
    def remembered_ids(self) -> frozenset[str]:
        return frozenset(item.element_id for item in self.memory)

    def after_observation(self, observation: PersonaObservation) -> AttentionState:
        """Notice new elements and retain bounded persona-visible memory."""

        budgets = self.budgets.consume_observation().consume_steps()
        new_elements = observation.newly_revealed_elements
        remembered = list(self.memory)
        retained_ids = {item.element_id for item in remembered}
        for element in (*observation.remembered_elements, *new_elements):
            if element.id in retained_ids:
                remembered = [
                    item for item in remembered if item.element_id != element.id
                ]
            remembered.append(
                RememberedElement(
                    element_id=element.id,
                    viewport_id=observation.viewport_id,
                    label=element.label,
                )
            )
            retained_ids.add(element.id)
        remembered = remembered[-self.memory_capacity :]
        return AttentionState(
            noticed_ids=self.noticed_ids | {element.id for element in new_elements},
            inspected_ids=self.inspected_ids,
            focus_region=(
                observation.region_context.id
                if observation.region_context is not None
                else self.focus_region
            ),
            budgets=budgets,
            memory=tuple(remembered),
            memory_capacity=self.memory_capacity,
            confidence=self.confidence,
            frustration=self.frustration,
            failed_candidates=self.failed_candidates,
            current_subgoal=self.current_subgoal,
            current_viewport_id=observation.viewport_id,
            current_observation_ids=frozenset(element.id for element in new_elements),
            recovery_misses=self.recovery_misses,
        )

    def after_action(self, action: AttentionAction) -> AttentionState:
        """Apply deterministic budget and inspection effects for one action."""

        budgets = self.budgets.consume_steps()
        inspected_ids = self.inspected_ids
        if isinstance(action, InspectElement):
            if action.element_id not in self.noticed_ids:
                raise ValueError("cannot inspect unnoticed element")
            if action.element_id not in self.remembered_ids:
                raise ValueError("cannot inspect unremembered element")
            inspected_ids = inspected_ids | {action.element_id}
        if isinstance(action, InteractWithElement):
            budgets = budgets.consume_interaction()
        return AttentionState(
            noticed_ids=self.noticed_ids,
            inspected_ids=inspected_ids,
            focus_region=self.focus_region,
            budgets=budgets,
            memory=self.memory,
            memory_capacity=self.memory_capacity,
            confidence=self.confidence,
            frustration=self.frustration,
            failed_candidates=self.failed_candidates,
            current_subgoal=self.current_subgoal,
            current_viewport_id=self.current_viewport_id,
            current_observation_ids=self.current_observation_ids,
            recovery_misses=self.recovery_misses,
        )

    def validate_action(
        self,
        action: AttentionAction,
        snapshot: ViewportSnapshot,
    ) -> None:
        """Validate action against current viewport and remembered attention."""

        if isinstance(action, (InspectElement, InteractWithElement)):
            if action.element_id not in self.noticed_ids:
                raise ValueError("interaction target must be noticed")
            if (
                action.element_id not in self.remembered_ids
                and action.element_id not in self.current_observation_ids
            ):
                raise ValueError("interaction target must be remembered")
        if self.current_viewport_id != snapshot.id:
            raise ValueError("action targets stale viewport")
        if isinstance(action, (InspectElement, InteractWithElement)):
            element = snapshot.element(action.element_id)
            if isinstance(action, InteractWithElement) and (
                not element.actionable or element.disabled
            ):
                raise ValueError("interaction target must be actionable")

    def mark_failed_candidate(self, element_id: str) -> AttentionState:
        return AttentionState(
            noticed_ids=self.noticed_ids,
            inspected_ids=self.inspected_ids,
            focus_region=self.focus_region,
            budgets=self.budgets,
            memory=self.memory,
            memory_capacity=self.memory_capacity,
            confidence=self.confidence,
            frustration=self.frustration,
            failed_candidates=self.failed_candidates | {element_id},
            current_subgoal=self.current_subgoal,
            current_viewport_id=self.current_viewport_id,
            current_observation_ids=self.current_observation_ids,
            recovery_misses=self.recovery_misses,
        )


@dataclass(frozen=True, slots=True)
class CoarseScent:
    """Pre-notice scent based only on public glance-level cues."""

    element_id: str
    viewport_id: str
    score: float
    glance_cues: tuple[str, ...]

    @classmethod
    def from_element(
        cls, snapshot: ViewportSnapshot, element: ElementSnapshot, score: float
    ) -> CoarseScent:
        if not 0 <= score <= 1:
            raise ValueError("coarse scent score must be between 0 and 1")
        snapshot.element(element.id)
        return cls(
            element_id=element.id,
            viewport_id=snapshot.id,
            score=score,
            glance_cues=(ElementRole(element.role).value, element.label),
        )


@dataclass(frozen=True, slots=True)
class FullScent:
    """Post-notice scent; creation requires noticed attention state."""

    element_id: str
    viewport_id: str
    score: float

    @classmethod
    def for_element(
        cls, state: AttentionState, element_id: str, score: float
    ) -> FullScent:
        if element_id not in state.noticed_ids:
            raise ValueError("full scent requires element to be noticed")
        if state.current_viewport_id is None:
            raise ValueError("full scent requires current viewport")
        if not 0 <= score <= 1:
            raise ValueError("full scent score must be between 0 and 1")
        return cls(
            element_id=element_id,
            viewport_id=state.current_viewport_id,
            score=score,
        )


@dataclass(frozen=True, slots=True)
class NoticeElements:
    kind: Literal["notice-elements"] = "notice-elements"
    element_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = tuple(self.element_ids)
        if not ids:
            raise ValueError("notice action needs at least one element")
        if len(ids) != len(set(ids)):
            raise ValueError("notice action contains duplicate element")
        object.__setattr__(self, "element_ids", ids)


@dataclass(frozen=True, slots=True)
class InspectElement:
    kind: Literal["inspect-element"] = "inspect-element"
    element_id: str = ""

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("inspect action needs element ID")


@dataclass(frozen=True, slots=True)
class InteractWithElement:
    kind: Literal["interact-with-element"] = "interact-with-element"
    element_id: str = ""

    def __post_init__(self) -> None:
        if not self.element_id:
            raise ValueError("interaction action needs element ID")


@dataclass(frozen=True, slots=True)
class Scroll:
    kind: Literal["scroll"] = "scroll"
    direction: Literal["up", "down"] = "down"


@dataclass(frozen=True, slots=True)
class Wait:
    kind: Literal["wait"] = "wait"
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.duration_seconds < 0:
            raise ValueError("wait duration must not be negative")


@dataclass(frozen=True, slots=True)
class Back:
    kind: Literal["back"] = "back"


@dataclass(frozen=True, slots=True)
class Complete:
    kind: Literal["complete"] = "complete"


@dataclass(frozen=True, slots=True)
class Abandon:
    kind: Literal["abandon"] = "abandon"
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("abandon action needs reason")


type AttentionAction = (
    NoticeElements
    | InspectElement
    | InteractWithElement
    | Scroll
    | Wait
    | Back
    | Complete
    | Abandon
)
