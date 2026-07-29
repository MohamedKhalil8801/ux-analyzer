"""Unrestricted persona-visible element exposure policy."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ux_analyzer.domain.attention import AttentionState, CompleteObservation
from ux_analyzer.domain.interface import PersonaVisibleElement, ViewportSnapshot

ListObservation = CompleteObservation


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

    @property
    def region_id(self) -> None:
        return None

    @property
    def region_probabilities(self) -> Mapping[None, float]:
        return MappingProxyType({None: 1.0})


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
    id = "full-list"
    version = "full-list-v1"

    def select(self, snapshot: ViewportSnapshot) -> ListObservationSelection:
        return ListObservationSelection(
            observation=CompleteObservation.from_snapshot(snapshot),
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
