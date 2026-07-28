"""Immutable, platform-neutral interface snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ElementRole(StrEnum):
    """Coarse semantic roles used by domain and provider code."""

    BUTTON = "button"
    LINK = "link"
    INPUT = "input"
    CHECKBOX = "checkbox"
    TAB = "tab"
    MENU = "menu"
    TEXT = "text"
    OTHER = "other"


class GraphRelation(StrEnum):
    """Supported relationships in the interface graph."""

    CONTAINS = "contains"
    NEAR = "near"
    LABELS = "labels"
    FOLLOWS = "follows"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Positive, finite rendered rectangle."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name, value in (
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.width <= 0:
            raise ValueError("width must be greater than zero")
        if self.height <= 0:
            raise ValueError("height must be greater than zero")

    def model_dump(self) -> dict[str, float]:
        """Return JSON-compatible public rectangle data."""

        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


Rect = BoundingBox
Bounds = BoundingBox


@dataclass(frozen=True, slots=True)
class PrivateExecutionReference:
    """Provider-only handle bound to one captured viewport."""

    provider_id: str
    viewport_id: str
    token: str

    def __post_init__(self) -> None:
        if not self.provider_id:
            raise ValueError("provider_id must not be empty")
        if not self.viewport_id:
            raise ValueError("viewport_id must not be empty")
        if not self.token:
            raise ValueError("execution token must not be empty")

    def with_viewport(self, viewport_id: str) -> PrivateExecutionReference:
        """Create deliberately stale reference useful for validation boundaries."""

        return PrivateExecutionReference(
            provider_id=self.provider_id,
            viewport_id=viewport_id,
            token=self.token,
        )


@dataclass(frozen=True, slots=True)
class RegionSnapshot:
    """Persona-visible region context captured with a viewport."""

    id: str
    label: str
    element_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("region id must not be empty")
        if not self.label:
            raise ValueError("region label must not be empty")
        ids = tuple(self.element_ids)
        if len(ids) != len(set(ids)):
            raise ValueError(f"region {self.id!r} has duplicate element IDs")
        object.__setattr__(self, "element_ids", ids)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """Directed relationship between captured interface nodes."""

    source_id: str
    target_id: str
    relation: GraphRelation | str

    def __post_init__(self) -> None:
        if not self.source_id or not self.target_id:
            raise ValueError("graph edge endpoints must not be empty")
        object.__setattr__(self, "relation", GraphRelation(self.relation))


@dataclass(frozen=True, slots=True)
class ElementSnapshot:
    """Immutable rendered element with private execution metadata."""

    id: str
    role: ElementRole | str
    label: str
    bounds: BoundingBox
    visibility_fraction: float
    actionable: bool
    disabled: bool = False
    region_id: str | None = None
    provider_id: str | None = None
    execution_reference: PrivateExecutionReference | None = None
    selector: str | None = None
    test_id: str | None = None
    hidden_label: str | None = None
    destination_url: str | None = None
    lineage_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("element id must not be empty")
        if not self.label:
            raise ValueError("element label must not be empty")
        object.__setattr__(self, "role", ElementRole(self.role))
        if not 0 <= self.visibility_fraction <= 1:
            raise ValueError("visibility_fraction must be between 0 and 1")
        if self.provider_id is not None and not self.provider_id:
            raise ValueError("provider_id must not be empty")

    @property
    def element_id(self) -> str:
        """Explicit vocabulary alias for snapshot identity."""

        return self.id


@dataclass(frozen=True, slots=True)
class PersonaVisibleElement:
    """Safe projection sent to persona/model code.

    Private provider handles and DOM identifiers are intentionally absent from
    this representation and its serialization method.
    """

    id: str
    role: ElementRole
    label: str
    bounds: BoundingBox
    visibility_fraction: float
    actionable: bool
    disabled: bool
    region_id: str | None

    @property
    def element_id(self) -> str:
        """Explicit vocabulary alias for persona-visible identity."""

        return self.id

    @classmethod
    def from_snapshot(cls, snapshot: ElementSnapshot) -> PersonaVisibleElement:
        """Project one private snapshot into persona-visible data."""

        return cls(
            id=snapshot.id,
            role=ElementRole(snapshot.role),
            label=snapshot.label,
            bounds=snapshot.bounds,
            visibility_fraction=snapshot.visibility_fraction,
            actionable=snapshot.actionable,
            disabled=snapshot.disabled,
            region_id=snapshot.region_id,
        )

    def model_dump(self) -> dict[str, Any]:
        """Return only fields safe for persona/model consumers."""

        return {
            "id": self.id,
            "role": self.role.value,
            "label": self.label,
            "bounds": self.bounds.model_dump(),
            "visibility_fraction": self.visibility_fraction,
            "actionable": self.actionable,
            "disabled": self.disabled,
            "region_id": self.region_id,
        }


@dataclass(frozen=True, slots=True)
class ViewportSnapshot:
    """Immutable interface capture and its provider-private references."""

    id: str
    elements: tuple[ElementSnapshot, ...]
    regions: tuple[RegionSnapshot, ...] = ()
    graph_edges: tuple[GraphEdge, ...] = ()
    screenshot_artifact: str | None = None
    provider_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("viewport id must not be empty")
        elements = tuple(self.elements)
        regions = tuple(self.regions)
        graph_edges = tuple(self.graph_edges)
        element_ids = [element.id for element in elements]
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("duplicate element ID in viewport snapshot")
        region_ids = {region.id for region in regions}
        if len(region_ids) != len(regions):
            raise ValueError("duplicate region ID in viewport snapshot")
        if self.provider_id is not None and not self.provider_id:
            raise ValueError("provider_id must not be empty")
        for element in elements:
            if element.region_id is not None and element.region_id not in region_ids:
                raise ValueError(
                    f"element {element.id!r} references unknown region "
                    f"{element.region_id!r}"
                )
            reference = element.execution_reference
            if reference is None:
                continue
            if reference.viewport_id != self.id:
                raise ValueError(
                    f"execution reference for element {element.id!r} belongs to "
                    "different viewport"
                )
            if (
                self.provider_id is not None
                and reference.provider_id != self.provider_id
            ):
                raise ValueError(
                    f"execution reference for element {element.id!r} belongs to "
                    "different provider"
                )
        known_ids = set(element_ids) | region_ids
        for edge in graph_edges:
            if edge.source_id not in known_ids or edge.target_id not in known_ids:
                raise ValueError("graph edge references unknown snapshot node")
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "graph_edges", graph_edges)

    @property
    def viewport_id(self) -> str:
        """Explicit vocabulary alias for snapshot identity."""

        return self.id

    @property
    def element_snapshots(self) -> tuple[ElementSnapshot, ...]:
        """Compatibility name for callers using explicit snapshot vocabulary."""

        return self.elements

    def element(self, element_id: str) -> ElementSnapshot:
        """Get captured element or reject unknown identity."""

        for element in self.elements:
            if element.id == element_id:
                return element
        raise ValueError(f"element {element_id!r} not present in viewport {self.id!r}")

    def persona_visible_elements(self) -> tuple[PersonaVisibleElement, ...]:
        """Return public projections in capture order."""

        return tuple(
            PersonaVisibleElement.from_snapshot(element) for element in self.elements
        )

    def validate_execution_reference(
        self,
        element_id: str,
        reference: PrivateExecutionReference,
    ) -> None:
        """Reject references not created by this exact viewport capture."""

        element = self.element(element_id)
        if reference.viewport_id != self.id:
            raise ValueError("execution reference belongs to stale viewport")
        if self.provider_id is not None and reference.provider_id != self.provider_id:
            raise ValueError("execution reference belongs to different provider")
        if element.execution_reference != reference:
            raise ValueError("execution reference does not belong to captured element")
