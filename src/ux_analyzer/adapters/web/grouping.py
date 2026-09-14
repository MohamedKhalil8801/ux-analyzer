"""Region and relationship normalization for rendered web elements."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import TYPE_CHECKING

from ux_analyzer.adapters.web.visibility import RawRect
from ux_analyzer.domain.interface import (
    ElementSnapshot,
    GraphEdge,
    GraphRelation,
    RegionSnapshot,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class RawRegionFact:
    """Safe-to-normalize region facts returned by browser evaluation."""

    ordinal: int
    kind: str
    label: str
    rendered_label: str
    bounds: RawRect
    ancestor_ordinals: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RawElementFact:
    """Rendered element facts before domain normalization."""

    ordinal: int
    tag: str
    role: str
    label: str
    rendered_text: str
    hidden_label: str | None
    bounds: RawRect
    visible_bounds: RawRect | None
    geometric_fraction: float
    occlusion_fraction: float
    actionable: bool
    disabled: bool
    selector: str
    dom_id: str | None
    test_id: str | None
    destination_url: str | None
    region_ordinals: tuple[int, ...]
    label_for: str | None
    has_visible_graphic: bool | None = None


def build_regions_and_edges(
    viewport_id: str,
    regions: Iterable[RawRegionFact],
    elements: Iterable[RawElementFact],
    snapshots: Iterable[ElementSnapshot],
) -> tuple[tuple[RegionSnapshot, ...], tuple[GraphEdge, ...]]:
    """Create public regions and graph edges from private ordinal references."""

    raw_regions = tuple(regions)
    raw_elements = tuple(elements)
    snapshot_by_ordinal = {
        element.ordinal: snapshot
        for element, snapshot in zip(raw_elements, snapshots, strict=True)
    }
    region_ids = {
        region.ordinal: f"{viewport_id}-region-{region.ordinal}"
        for region in raw_regions
    }
    element_ids = {
        ordinal: snapshot.id for ordinal, snapshot in snapshot_by_ordinal.items()
    }
    region_elements: dict[int, list[str]] = {
        region.ordinal: [] for region in raw_regions
    }
    for element in raw_elements:
        for region_ordinal in element.region_ordinals:
            if region_ordinal in region_elements:
                region_elements[region_ordinal].append(element_ids[element.ordinal])
    normalized_regions = tuple(
            RegionSnapshot(
                id=region_ids[region.ordinal],
                label=region.label,
                rendered_label=region.rendered_label,
                element_ids=tuple(region_elements[region.ordinal]),
        )
        for region in raw_regions
    )

    edges: list[GraphEdge] = []
    for region in raw_regions:
        region_id = region_ids[region.ordinal]
        for element_id in region_elements[region.ordinal]:
            edges.append(GraphEdge(region_id, element_id, GraphRelation.CONTAINS))
        for ancestor_ordinal in region.ancestor_ordinals:
            ancestor_id = region_ids.get(ancestor_ordinal)
            if ancestor_id is not None:
                edges.append(GraphEdge(ancestor_id, region_id, GraphRelation.CONTAINS))

    dom_id_to_ordinal = {
        element.dom_id: element.ordinal for element in raw_elements if element.dom_id
    }
    for element in raw_elements:
        if element.label_for is None:
            continue
        target_ordinal = dom_id_to_ordinal.get(element.label_for)
        if target_ordinal is not None:
            edges.append(
                GraphEdge(
                    element_ids[element.ordinal],
                    element_ids[target_ordinal],
                    GraphRelation.LABELS,
                )
            )

    for region_ordinal in region_elements:
        members = [
            element
            for element in raw_elements
            if region_ordinal in element.region_ordinals
        ]
        for first, second in zip(members, members[1:]):
            edges.append(
                GraphEdge(
                    element_ids[first.ordinal],
                    element_ids[second.ordinal],
                    GraphRelation.FOLLOWS,
                )
            )
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                if _near(first.bounds, second.bounds):
                    edges.append(
                        GraphEdge(
                            element_ids[first.ordinal],
                            element_ids[second.ordinal],
                            GraphRelation.NEAR,
                        )
                    )

    return normalized_regions, _deduplicate_edges(edges)


def _near(first: RawRect, second: RawRect) -> bool:
    first_center = (first.x + first.width / 2, first.y + first.height / 2)
    second_center = (second.x + second.width / 2, second.y + second.height / 2)
    distance = hypot(
        first_center[0] - second_center[0], first_center[1] - second_center[1]
    )
    return distance <= max(180.0, first.width + second.width + 24.0)


def _deduplicate_edges(edges: Iterable[GraphEdge]) -> tuple[GraphEdge, ...]:
    seen: set[tuple[str, str, GraphRelation]] = set()
    unique: list[GraphEdge] = []
    for edge in edges:
        key = (edge.source_id, edge.target_id, GraphRelation(edge.relation))
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return tuple(unique)
