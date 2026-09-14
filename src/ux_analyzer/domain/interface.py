"""Immutable, platform-neutral interface snapshots."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
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
    rendered_label: str | None = None

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
    local_contrast: float | None = None
    occlusion_fraction: float | None = None
    rendered_text: str | None = None
    # Whether this element paints a non-text graphic a sighted user can see
    # (an icon). ``None`` means the capture did not report it (older
    # payloads). Used only to decide whether an element with no rendered text
    # is still perceivable; it never contributes to the persona label.
    has_visible_graphic: bool | None = None

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
        for name, value in (
            ("local_contrast", self.local_contrast),
            ("occlusion_fraction", self.occlusion_fraction),
        ):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be between 0 and 1")

    @property
    def element_id(self) -> str:
        """Explicit vocabulary alias for snapshot identity."""

        return self.id


_PERCEIVABLE_ACTIONS: Mapping[ElementRole, str] = MappingProxyType(
    {
        ElementRole.BUTTON: "button",
        ElementRole.LINK: "link",
        ElementRole.CHECKBOX: "checkbox",
        ElementRole.INPUT: "field",
        ElementRole.TAB: "tab",
        ElementRole.MENU: "menu item",
    }
)


def _bearing(bounds: BoundingBox, viewport_width: float | None = None) -> str:
    """Describe where an element sits the way a person would say it out loud.

    Only horizontal placement is inferred here: the capture does not
    guarantee a viewport height, so vertical placement is deliberately left
    unstated rather than guessed wrong.
    """

    if viewport_width is None or viewport_width <= 0:
        return "on the page"
    center = bounds.x + bounds.width / 2
    if center < viewport_width / 3:
        return "on the left"
    if center > viewport_width * 2 / 3:
        return "on the right"
    return "in the middle"


def _perceivable_identity(snapshot: ElementSnapshot) -> str:
    """Describe an unlabelled control using only sighted-user information.

    Used when an element renders no text of its own (a typical icon-only
    button). It reports the control's role, its approximate bearing, and its
    size — all of which a sighted user perceives directly. It must never
    substitute ``snapshot.label`` (the author's accessible name) or any other
    markup-only attribute, because those are unavailable to a sighted user.
    """

    action = _PERCEIVABLE_ACTIONS.get(ElementRole(snapshot.role), "control")
    width = round(snapshot.bounds.width)
    height = round(snapshot.bounds.height)
    return (
        f"unlabelled {action} {_bearing(snapshot.bounds)} "
        f"({width}x{height} px)"
    )


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
        """Project one private snapshot into persona-visible data.

        The projection must never hand the persona knowledge a sighted user
        would not have. In particular the author's accessible name is *not*
        a substitute for visible text: it exists for assistive technology and
        is invisible to a sighted user, so leaking it would make the simulated
        user smarter than the human it stands in for.

        ``rendered_text`` distinguishes "renders no text" (``""``/``None``)
        from "renders text". When a control renders no text at all, the
        persona is given a description of what a sighted user would actually
        perceive — its role and its position on the page — instead of a name
        only the markup knows. This keeps an icon-only control identifiable
        without revealing anything hidden.
        """

        rendered = (snapshot.rendered_text or "").strip()
        label = rendered or _perceivable_identity(snapshot)
        return cls(
            id=snapshot.id,
            role=ElementRole(snapshot.role),
            label=label,
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
