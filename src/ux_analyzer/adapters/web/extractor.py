"""Rendered web extraction into platform-neutral interface snapshots."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import cast

from playwright.async_api import Page

from ux_analyzer.adapters.web.grouping import (
    RawElementFact,
    RawRegionFact,
    build_regions_and_edges,
)
from ux_analyzer.adapters.web.visibility import (
    RawRect,
    decode_screenshot,
    effective_visibility,
    geometric_visibility,
    screenshot_local_contrast,
)
from ux_analyzer.domain.interface import (
    ElementRole,
    ElementSnapshot,
    PrivateExecutionReference,
    ViewportSnapshot,
)

WEB_PROVIDER_ID = "playwright-web"
StageMeasure = Callable[[str], AbstractContextManager[None]]


@dataclass(frozen=True, slots=True)
class ExtractionDiagnostics:
    """Measurements useful to later prominence providers, kept outside public snapshot."""

    local_contrast: Mapping[str, float]
    occlusion_fraction: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Snapshot plus private measurement diagnostics from one capture."""

    snapshot: ViewportSnapshot
    diagnostics: ExtractionDiagnostics


async def capture(
    page: Page,
    viewport_id: str,
    *,
    measure: StageMeasure | None = None,
) -> ViewportSnapshot:
    """Capture one rendered page using one browser evaluation payload."""

    return (
        await capture_with_diagnostics(
            page,
            viewport_id,
            measure=measure,
        )
    ).snapshot


async def capture_with_diagnostics(
    page: Page,
    viewport_id: str,
    *,
    measure: StageMeasure | None = None,
) -> ExtractionResult:
    """Capture snapshot and derived private measurements in deterministic order."""

    if not viewport_id:
        raise ValueError("viewport ID must not be empty")
    with _measure(measure, "dom.page_evaluate"):
        payload = await page.evaluate(EVALUATION_PAYLOAD)
    with _measure(measure, "dom.normalize"):
        raw_regions, raw_elements, viewport_width, viewport_height = _normalize_payload(
            payload
        )
    with _measure(measure, "dom.screenshot"):
        captured_screenshot = await page.screenshot(type="png")
    snapshots: list[ElementSnapshot] = []
    contrast: dict[str, float] = {}
    occlusion: dict[str, float] = {}
    lineage_ids = _lineage_ids(raw_elements, raw_regions)
    with _measure(measure, "dom.local_contrast"):
        decoded_screenshot = decode_screenshot(captured_screenshot)
        for raw_element, lineage_id in zip(raw_elements, lineage_ids, strict=True):
            bounds = raw_element.bounds.to_domain()
            element_id = f"{viewport_id}-element-{raw_element.ordinal}"
            local_contrast = screenshot_local_contrast(
                decoded_screenshot,
                bounds,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
            )
            occlusion_fraction = max(0.0, min(1.0, raw_element.occlusion_fraction))
            effective_fraction = effective_visibility(
                geometric_visibility(raw_element.bounds, raw_element.visible_bounds),
                occlusion_fraction,
            )
            snapshots.append(
                ElementSnapshot(
                    id=element_id,
                    role=_domain_role(raw_element.role, raw_element.tag),
                    label=raw_element.label,
                    bounds=bounds,
                    visibility_fraction=effective_fraction,
                    actionable=raw_element.actionable,
                    disabled=raw_element.disabled,
                    region_id=(
                        f"{viewport_id}-region-{raw_element.region_ordinals[0]}"
                        if raw_element.region_ordinals
                        else None
                    ),
                    provider_id=WEB_PROVIDER_ID,
                    execution_reference=PrivateExecutionReference(
                        provider_id=WEB_PROVIDER_ID,
                        viewport_id=viewport_id,
                        token=_execution_token(viewport_id, raw_element),
                    ),
                    selector=raw_element.selector,
                    test_id=raw_element.test_id,
                    hidden_label=raw_element.hidden_label,
                    destination_url=raw_element.destination_url,
                    lineage_id=lineage_id,
                    local_contrast=local_contrast,
                    occlusion_fraction=occlusion_fraction,
                )
            )
            contrast[element_id] = local_contrast
            occlusion[element_id] = occlusion_fraction
    with _measure(measure, "dom.grouping"):
        regions, graph_edges = build_regions_and_edges(
            viewport_id, raw_regions, raw_elements, snapshots
        )
    snapshot = ViewportSnapshot(
        id=viewport_id,
        elements=tuple(snapshots),
        regions=regions,
        graph_edges=graph_edges,
        provider_id=WEB_PROVIDER_ID,
    )
    return ExtractionResult(
        snapshot=snapshot,
        diagnostics=ExtractionDiagnostics(
            local_contrast=contrast,
            occlusion_fraction=occlusion,
        ),
    )


def _measure(
    measure: StageMeasure | None,
    stage: str,
) -> AbstractContextManager[None]:
    return measure(stage) if measure is not None else nullcontext()


def _execution_token(viewport_id: str, element: RawElementFact) -> str:
    material = f"{viewport_id}\0{element.ordinal}\0{element.selector}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _lineage_ids(
    elements: tuple[RawElementFact, ...],
    regions: tuple[RawRegionFact, ...],
) -> tuple[str | None, ...]:
    """Link semantically unchanged controls without selectors or public identity."""

    region_by_ordinal = {region.ordinal: region for region in regions}
    signatures: list[str] = []
    for element in elements:
        region_context = tuple(
            f"{region_by_ordinal[ordinal].kind}:{region_by_ordinal[ordinal].label}"
            for ordinal in element.region_ordinals
            if ordinal in region_by_ordinal
        )
        signature = "\0".join(
            (
                "element-lineage-v1",
                element.tag,
                element.role,
                element.label,
                str(element.actionable),
                str(element.disabled),
                *region_context,
            )
        )
        signatures.append(signature)
    counts = Counter(signatures)
    return tuple(
        (
            f"lineage-v1-{hashlib.sha256(signature.encode('utf-8')).hexdigest()}"
            if counts[signature] == 1
            else None
        )
        for signature in signatures
    )


def _domain_role(role: str, tag: str) -> ElementRole:
    if role in {item.value for item in ElementRole}:
        return ElementRole(role)
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "label", "p"}:
        return ElementRole.TEXT
    return ElementRole.OTHER


def _normalize_payload(
    payload: object,
) -> tuple[tuple[RawRegionFact, ...], tuple[RawElementFact, ...], int, int]:
    root = _mapping(payload, "evaluation payload")
    viewport = _mapping(root.get("viewport"), "viewport payload")
    width = _positive_int(viewport.get("width"), "viewport width")
    height = _positive_int(viewport.get("height"), "viewport height")
    raw_regions = tuple(_region_fact(item) for item in _list(root.get("regions")))
    raw_elements = tuple(_element_fact(item) for item in _list(root.get("elements")))
    _validate_ordinals(raw_regions, raw_elements)
    known_regions = {region.ordinal for region in raw_regions}
    for element in raw_elements:
        unknown = set(element.region_ordinals).difference(known_regions)
        if unknown:
            raise ValueError(f"element references unknown regions: {sorted(unknown)}")
    return raw_regions, raw_elements, width, height


def _region_fact(payload: object) -> RawRegionFact:
    item = _mapping(payload, "region fact")
    return RawRegionFact(
        ordinal=_positive_or_zero_int(item.get("ordinal"), "region ordinal"),
        kind=_required_string(item.get("kind"), "region kind"),
        label=_required_string(item.get("label"), "region label"),
        bounds=RawRect.from_payload(item.get("bounds")),
        ancestor_ordinals=tuple(
            _positive_or_zero_int(value, "region ancestor ordinal")
            for value in _list(item.get("ancestorOrdinals"))
        ),
    )


def _element_fact(payload: object) -> RawElementFact:
    item = _mapping(payload, "element fact")
    bounds = RawRect.from_payload(item.get("bounds"))
    visible_bounds_payload = item.get("visibleBounds")
    return RawElementFact(
        ordinal=_positive_or_zero_int(item.get("ordinal"), "element ordinal"),
        tag=_required_string(item.get("tag"), "element tag"),
        role=_required_string(item.get("role"), "element role"),
        label=_required_string(item.get("label"), "element label"),
        hidden_label=_optional_string(item.get("hiddenLabel")),
        bounds=bounds,
        visible_bounds=(
            None
            if visible_bounds_payload is None
            else RawRect.from_payload(visible_bounds_payload)
        ),
        geometric_fraction=geometric_visibility(
            bounds,
            None
            if visible_bounds_payload is None
            else RawRect.from_payload(visible_bounds_payload),
        ),
        occlusion_fraction=_fraction(
            item.get("occlusionFraction"), "occlusion fraction"
        ),
        actionable=_required_bool(item.get("actionable"), "actionable"),
        disabled=_required_bool(item.get("disabled"), "disabled"),
        selector=_required_string(item.get("selector"), "element selector"),
        dom_id=_optional_string(item.get("domId")),
        test_id=_optional_string(item.get("testId")),
        destination_url=_optional_string(item.get("destinationUrl")),
        region_ordinals=tuple(
            _positive_or_zero_int(value, "element region ordinal")
            for value in _list(item.get("regionOrdinals"))
        ),
        label_for=_optional_string(item.get("labelFor")),
    )


def _validate_ordinals(
    regions: tuple[RawRegionFact, ...], elements: tuple[RawElementFact, ...]
) -> None:
    region_ordinals = [region.ordinal for region in regions]
    element_ordinals = [element.ordinal for element in elements]
    if len(region_ordinals) != len(set(region_ordinals)):
        raise ValueError("duplicate region ordinal")
    if len(element_ordinals) != len(set(element_ordinals)):
        raise ValueError("duplicate element ordinal")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _list(value: object) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("evaluation collection must be a list")
    return cast(list[object], value)


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return " ".join(value.split())


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional text must be string or null")
    normalized = " ".join(value.split())
    return normalized or None


def _required_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _positive_or_zero_int(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _positive_or_zero_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return result


EVALUATION_PAYLOAD = r"""
() => {
  const clean = (value) => (value || '').replace(/\s+/g, ' ').trim();
  const viewport = {width: window.innerWidth, height: window.innerHeight};
  const viewportRect = {x: 0, y: 0, width: viewport.width, height: viewport.height};
  const area = (rect) => Math.max(0, rect.width) * Math.max(0, rect.height);
  const intersect = (first, second) => {
    const left = Math.max(first.x, second.x);
    const top = Math.max(first.y, second.y);
    const right = Math.min(first.x + first.width, second.x + second.width);
    const bottom = Math.min(first.y + first.height, second.y + second.height);
    if (right <= left || bottom <= top) return null;
    return {x: left, y: top, width: right - left, height: bottom - top};
  };
  const styleVisible = (node) => {
    if (!(node instanceof Element)) return false;
    if (node.hasAttribute('hidden') || node.getAttribute('aria-hidden') === 'true') return false;
    for (let ancestor = node; ancestor instanceof Element; ancestor = ancestor.parentElement) {
      const style = getComputedStyle(ancestor);
      if (style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse') return false;
      if (Number.parseFloat(style.opacity) === 0) return false;
    }
    return true;
  };
  const visibleGeometry = (node) => {
    const bounds = node.getBoundingClientRect();
    const original = {x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height};
    if (area(original) <= 0 || !styleVisible(node)) return {bounds: original, visibleBounds: null};
    let visible = intersect(original, viewportRect);
    for (let ancestor = node.parentElement; ancestor instanceof Element && visible; ancestor = ancestor.parentElement) {
      const style = getComputedStyle(ancestor);
      const clipsX = ['hidden', 'clip', 'scroll', 'auto'].includes(style.overflowX);
      const clipsY = ['hidden', 'clip', 'scroll', 'auto'].includes(style.overflowY);
      if (clipsX || clipsY) {
        const ancestorBounds = ancestor.getBoundingClientRect();
        const clip = {
          x: clipsX ? ancestorBounds.x : -1e9,
          y: clipsY ? ancestorBounds.y : -1e9,
          width: clipsX ? ancestorBounds.width : 2e9,
          height: clipsY ? ancestorBounds.height : 2e9
        };
        visible = intersect(visible, clip);
      }
    }
    return {bounds: original, visibleBounds: visible};
  };
  const probePoints = (rect) => {
    if (!rect) return [];
    const insetX = Math.min(rect.width / 4, 1);
    const insetY = Math.min(rect.height / 4, 1);
    const left = rect.x + insetX;
    const right = rect.x + rect.width - insetX;
    const top = rect.y + insetY;
    const bottom = rect.y + rect.height - insetY;
    return [[left, top], [right, top], [rect.x + rect.width / 2, rect.y + rect.height / 2], [left, bottom], [right, bottom]];
  };
  const isOpaqueBlocker = (candidate) => {
    if (!(candidate instanceof Element)) return false;
    const style = getComputedStyle(candidate);
    return style.display !== 'none' && style.visibility !== 'hidden' && Number.parseFloat(style.opacity) > 0.02;
  };
  const occlusionFraction = (node, visibleBounds) => {
    const points = probePoints(visibleBounds);
    if (!points.length) return 1;
    let blocked = 0;
    for (const [x, y] of points) {
      const stack = document.elementsFromPoint(x, y);
      const visibleAtPoint = stack.some((candidate) => candidate === node || node.contains(candidate));
      const firstOpaque = stack.find((candidate) => isOpaqueBlocker(candidate));
      const blocker = firstOpaque && firstOpaque !== node && !node.contains(firstOpaque);
      if (!visibleAtPoint || blocker) blocked += 1;
    }
    return blocked / points.length;
  };
  const selectorFor = (node) => {
    if (node.id) return `#${CSS.escape(node.id)}`;
    const path = [];
    for (let current = node; current instanceof Element && current !== document.body; current = current.parentElement) {
      let step = current.tagName.toLowerCase();
      let sibling = current;
      let position = 1;
      while ((sibling = sibling.previousElementSibling)) {
        if (sibling.tagName === current.tagName) position += 1;
      }
      step += `:nth-of-type(${position})`;
      path.unshift(step);
    }
    return path.join(' > ') || 'body';
  };
  const visibleText = (node) => {
    const text = clean(node.innerText);
    return /[\p{L}\p{N}]/u.test(text) ? text : '';
  };
  const referencedText = (node) => {
    const ids = clean(node.getAttribute('aria-labelledby')).split(' ').filter(Boolean);
    return clean(ids.map((id) => document.getElementById(id)?.innerText || '').join(' '));
  };
  const associatedLabel = (node) => {
    if (!(node instanceof HTMLInputElement || node instanceof HTMLSelectElement || node instanceof HTMLTextAreaElement)) return '';
    return clean(Array.from(node.labels || []).map((label) => label.innerText).join(' '));
  };
  const semanticRole = (node) => {
    const explicit = clean(node.getAttribute('role')).toLowerCase();
    if (explicit === 'button' || explicit === 'link' || explicit === 'tab' || explicit === 'menu' || explicit === 'menuitem' || explicit === 'checkbox') return explicit;
    const tag = node.tagName.toLowerCase();
    if (tag === 'button' || tag === 'summary') return 'button';
    if (tag === 'a' && node.hasAttribute('href')) return 'link';
    if (tag === 'input' && ['checkbox', 'radio'].includes(node.type)) return 'checkbox';
    if (['input', 'select', 'textarea'].includes(tag)) return 'input';
    if (['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'label'].includes(tag)) return 'text';
    if (node.getAttribute('role') === 'status' || node.getAttribute('role') === 'alert') return 'text';
    return 'other';
  };
  const isCandidate = (node, role, text) => {
    const tag = node.tagName.toLowerCase();
    return ['button', 'a', 'input', 'select', 'textarea', 'summary', 'label', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tag)
      || ['button', 'link', 'tab', 'menu', 'menuitem', 'checkbox'].includes(role)
      || ['status', 'alert'].includes(node.getAttribute('role'))
      || (text && ['div', 'p', 'span', 'section'].includes(tag)
        && !Array.from(node.children).some((child) => isCandidate(child, semanticRole(child), visibleText(child))));
  };
  const regionNode = (node) => {
    const tag = node.tagName.toLowerCase();
    const role = clean(node.getAttribute('role')).toLowerCase();
    return ['header', 'nav', 'main', 'aside', 'footer', 'section', 'article', 'form', 'ul', 'ol', 'dialog'].includes(tag)
      || ['banner', 'navigation', 'main', 'complementary', 'contentinfo', 'region', 'form', 'dialog', 'list'].includes(role);
  };
  const regionLabel = (node) => {
    const tag = node.tagName.toLowerCase();
    const aria = clean(node.getAttribute('aria-label'));
    const labelled = referencedText(node);
    const heading = clean(Array.from(node.querySelectorAll('h1,h2,h3,h4,h5,h6')).map((item) => item.innerText).join(' '));
    const defaults = {header: 'Header', nav: 'Navigation', main: 'Main', aside: 'Sidebar', footer: 'Footer', section: 'Section', article: 'Section', form: 'Form', ul: 'List', ol: 'List', dialog: 'Dialog'};
    return aria || labelled || heading || defaults[tag] || 'Region';
  };
  const allNodes = Array.from(document.querySelectorAll('*'));
  const regions = [];
  const regionOrdinalByNode = new Map();
  for (const node of allNodes) {
    if (!regionNode(node)) continue;
    const geometry = visibleGeometry(node);
    if (!geometry.visibleBounds) continue;
    const ordinal = regions.length;
    regionOrdinalByNode.set(node, ordinal);
    const ancestors = [];
    for (let ancestor = node.parentElement; ancestor instanceof Element; ancestor = ancestor.parentElement) {
      const ancestorOrdinal = regionOrdinalByNode.get(ancestor);
      if (ancestorOrdinal !== undefined) ancestors.push(ancestorOrdinal);
    }
    regions.push({ordinal, kind: node.tagName.toLowerCase(), label: regionLabel(node), bounds: geometry.bounds, ancestorOrdinals: ancestors});
  }
  const elements = [];
  for (const node of allNodes) {
    if (!styleVisible(node)) continue;
    const role = semanticRole(node);
    const geometry = visibleGeometry(node);
    if (!geometry.visibleBounds) continue;
    const visible = visibleText(node);
    if (!isCandidate(node, role, visible)) continue;
    const accessible = clean(node.getAttribute('aria-label'));
    const label = visible || associatedLabel(node) || accessible || clean(node.getAttribute('placeholder')) || role || 'control';
    const regionOrdinals = [];
    for (let ancestor = node; ancestor instanceof Element; ancestor = ancestor.parentElement) {
      const regionOrdinal = regionOrdinalByNode.get(ancestor);
      if (regionOrdinal !== undefined) regionOrdinals.push(regionOrdinal);
    }
    const disabled = node.hasAttribute('disabled') || node.getAttribute('aria-disabled') === 'true';
    const tag = node.tagName.toLowerCase();
    const actionable = !disabled && (['button', 'a', 'input', 'select', 'textarea', 'summary'].includes(tag) || ['button', 'link', 'tab', 'menuitem', 'checkbox'].includes(role));
    elements.push({
      ordinal: elements.length,
      tag,
      role,
      label,
      hiddenLabel: clean(node.getAttribute('data-hidden-label')) || null,
      bounds: geometry.bounds,
      visibleBounds: geometry.visibleBounds,
      occlusionFraction: occlusionFraction(node, geometry.visibleBounds),
      actionable,
      disabled,
      selector: selectorFor(node),
      domId: node.id || null,
      testId: node.getAttribute('data-testid'),
      destinationUrl: node instanceof HTMLAnchorElement ? node.href : null,
      regionOrdinals,
      labelFor: tag === 'label' ? node.htmlFor || null : null
    });
  }
  return {viewport, regions, elements};
}
"""
