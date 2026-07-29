"""Prominence-ranked unrestricted persona-visible element policy."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import cast

from ux_analyzer.domain.attention import AttentionState
from ux_analyzer.domain.interface import PersonaVisibleElement, ViewportSnapshot
from ux_analyzer.providers.full_list_policy import (
    FullListPolicy,
    ListObservation,
    ListObservationSelection,
    visible_elements,
)
from ux_analyzer.providers.prominence import (
    HeuristicProminenceProvider,
    ProminenceResult,
)


class ProminenceRankedListPolicy(FullListPolicy):
    """Reveal full persona-safe list sorted by heuristic prominence."""

    selection_mode = "prominence-ranked-list"
    id = "prominence-ranked-list"
    version = "prominence-ranked-list-v1"

    def __init__(
        self, prominence_provider: HeuristicProminenceProvider | None = None
    ) -> None:
        self.prominence_provider = prominence_provider or HeuristicProminenceProvider()

    def select(
        self,
        snapshot: ViewportSnapshot,
        scores: Sequence[ProminenceResult] = (),
    ) -> ListObservationSelection:
        visible = visible_elements(snapshot)
        score_by_id = self._score_by_id(
            snapshot, scores or self.prominence_provider.score(snapshot)
        )
        order = {element.id: index for index, element in enumerate(visible)}
        ranked = tuple(
            sorted(
                visible,
                key=lambda element: (
                    -score_by_id[element.id].raw_score,
                    -score_by_id[element.id].normalized_probability,
                    order[element.id],
                ),
            )
        )
        return ListObservationSelection(
            observation=self._observation(snapshot, ranked),
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
        del state, coarse_scent, rng
        return self.select(snapshot, _score_sequence(scores))

    @staticmethod
    def _score_by_id(
        snapshot: ViewportSnapshot, scores: Sequence[ProminenceResult]
    ) -> dict[str, ProminenceResult]:
        known_ids = {element.id for element in snapshot.elements}
        result: dict[str, ProminenceResult] = {}
        for score in scores:
            if score.element_id not in known_ids:
                raise ValueError(
                    f"prominence score references unknown element {score.element_id!r}"
                )
            if score.element_id in result:
                raise ValueError(f"duplicate prominence score for {score.element_id!r}")
            result[score.element_id] = score
        missing = known_ids.difference(result)
        if missing:
            raise ValueError(
                f"missing prominence score for elements: {sorted(missing)}"
            )
        return result

    @staticmethod
    def _observation(
        snapshot: ViewportSnapshot, elements: Sequence[PersonaVisibleElement]
    ) -> ListObservation:
        return ListObservation(
            viewport_id=snapshot.id,
            newly_revealed_elements=tuple(elements),
        )


RankedListPolicy = ProminenceRankedListPolicy


def _score_sequence(value: object) -> Sequence[ProminenceResult]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    sequence = cast(Sequence[object], value)
    values: tuple[object, ...] = tuple(sequence)
    if not all(isinstance(item, ProminenceResult) for item in values):
        raise TypeError("ranked list scores must contain ProminenceResult values")
    return tuple(cast(ProminenceResult, item) for item in values)
