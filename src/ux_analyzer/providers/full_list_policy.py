"""Unrestricted persona-visible element exposure policy."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ux_analyzer.domain.attention import AttentionState
from ux_analyzer.domain.interface import PersonaVisibleElement, ViewportSnapshot


@dataclass(frozen=True, slots=True)
class ListObservation:
    """Persona-visible observation used by unrestricted list policies."""

    viewport_id: str
    newly_revealed_elements: tuple[PersonaVisibleElement, ...]
    remembered_elements: tuple[PersonaVisibleElement, ...] = ()

    def __post_init__(self) -> None:
        elements = tuple(self.newly_revealed_elements)
        remembered = tuple(self.remembered_elements)
        if not elements:
            raise ValueError("list observation needs at least one visible element")
        ids = [element.id for element in elements]
        if len(ids) != len(set(ids)):
            raise ValueError("list observation contains duplicate elements")
        object.__setattr__(self, "newly_revealed_elements", elements)
        object.__setattr__(self, "remembered_elements", remembered)


@dataclass(frozen=True, slots=True)
class ListObservationSelection:
    """Selection result shared by full and ranked list policies."""

    observation: ListObservation
    selection_mode: str

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return tuple(element.id for element in self.observation.newly_revealed_elements)

    @property
    def newly_revealed_elements(self) -> tuple[PersonaVisibleElement, ...]:
        return self.observation.newly_revealed_elements

    @property
    def element_probabilities(self) -> Mapping[str, float]:
        """Expose no prominence values; all elements are directly available."""

        return MappingProxyType({})


def visible_elements(snapshot: ViewportSnapshot) -> tuple[PersonaVisibleElement, ...]:
    """Project all rendered, non-zero-visibility elements into safe data."""

    return tuple(
        PersonaVisibleElement.from_snapshot(element)
        for element in snapshot.elements
        if element.visibility_fraction > 0
    )


class FullListPolicy:
    """Reveal every visible persona-safe element in capture order."""

    selection_mode = "full-list"

    def select(self, snapshot: ViewportSnapshot) -> ListObservationSelection:
        elements = visible_elements(snapshot)
        return ListObservationSelection(
            observation=ListObservation(
                viewport_id=snapshot.id,
                newly_revealed_elements=elements,
            ),
            selection_mode=self.selection_mode,
        )

    def next_observation(
        self,
        state: AttentionState,
        snapshot: ViewportSnapshot,
        scores: object = (),
        coarse_scent: object = (),
        rng: random.Random | None = None,
    ) -> ListObservationSelection:
        del state, scores, coarse_scent, rng
        return self.select(snapshot)


FullListAttentionPolicy = FullListPolicy
