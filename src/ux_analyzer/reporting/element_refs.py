"""Element reference resolution for report prose (``e{n}`` hover chips).

Synthesis models cite evidence through compact aliases (``e674``) that are
minted per attempt as *the index of the entry in the attempt's corpus
manifest* (``providers/report_synthesis.py::_provider_evidence_handle_map``).
The manifest is persisted next to every synthesis attempt, so the renderer
can rebuild the alias -> evidence map offline and make every ``e{n}`` token
in finding prose hoverable:

- hover shows what the alias stands for (kind, surface, recorded value);
- ``kind=element`` aliases additionally show a screenshot crop of the exact
  recorded element and offer a verified-unique DOM locator for the live site.

All element data here is operator-facing report data only; none of it is
fed back to any model role (ADR: models never see DOM identifiers).
"""

from __future__ import annotations

import base64
import io
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast

from PIL import Image

from ux_analyzer.application.evidence_corpus import EvidenceCorpus

_ALIAS_PATTERN = re.compile(r"\be(\d{1,4})\b")
_CROP_MAX_EDGE = 320
_CROP_PADDING = 8
_CROP_JPEG_QUALITY = 85
# Max px distance (Manhattan, capture-time viewport coords) between a cited
# element's recorded bounds and an inventoried node before they are
# considered different elements.
_LOCATOR_MATCH_TOLERANCE_PX = 64


def alias_index_from_corpus(corpus: EvidenceCorpus) -> dict[str, str]:
    """Rebuild the model-facing alias map: ``e{index}`` -> evidence ID.

    Mirrors ``_provider_evidence_handle_map``: aliases are positions in the
    corpus manifest's entry order, which is exactly what the attempt's
    persisted ``corpus-manifest.json`` froze.
    """

    return {
        f"e{index}": entry.ref.evidence_id
        for index, entry in enumerate(corpus.entries)
    }


def evidence_detail_for_alias(
    corpus: EvidenceCorpus, evidence_id: str
) -> dict[str, Any] | None:
    """Human-facing detail card for one evidence entry."""

    try:
        entry = corpus.require(evidence_id)
    except ValueError:
        return None
    payload = entry.payload
    detail: dict[str, Any] = {
        "evidence_id": entry.ref.evidence_id,
        "kind": entry.ref.kind,
        "run_id": entry.ref.run_id,
    }
    for key in ("surface", "name", "value", "unit"):
        value = payload.get(key)
        if isinstance(value, (str, int, float, bool)):
            detail[key] = value
    if entry.ref.element_id:
        detail["element_id"] = entry.ref.element_id
    evidence_class = payload.get("evidence_class")
    if isinstance(evidence_class, str):
        detail["evidence_class"] = evidence_class
    return detail


def prose_alias_segments(
    text: str,
    known: frozenset[str] | set[str],
) -> list[str | dict[str, str]]:
    """Split prose into render segments around known ``e{n}`` aliases.

    Plain segments stay ``str``; each known alias becomes
    ``{"alias": "e674"}`` for the template to emit a chip. Adjacent
    segments are never empty.
    """

    segments: list[str | dict[str, str]] = []
    cursor = 0
    for match in _ALIAS_PATTERN.finditer(text):
        token = f"e{int(match.group(1))}"
        if token not in known:
            continue
        if match.start() > cursor:
            segments.append(text[cursor : match.start()])
        segments.append({"alias": token})
        cursor = match.end()
    if cursor < len(text):
        segments.append(text[cursor:])
    return segments


def aliases_in_prose(texts: list[str] | tuple[str, ...]) -> set[str]:
    """Distinct ``e{n}`` tokens appearing anywhere in the given prose.

    Intentionally unfiltered: the raw model vocabulary is the universe for
    chip-payload construction, resolved against the corpus index later.
    """

    return {
        f"e{int(match.group(1))}"
        for text in texts
        for match in _ALIAS_PATTERN.finditer(text)
    }


# ---------------------------------------------------------------------------
# Element-level resolution (kind=element aliases)
# ---------------------------------------------------------------------------


def _element_by_id(
    snapshots: Any, element_id: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not isinstance(snapshots, list):
        return None
    for snapshot_item in cast("list[object]", snapshots):
        if not isinstance(snapshot_item, dict):
            continue
        snapshot = cast(dict[str, Any], snapshot_item)
        elements = snapshot.get("elements")
        if not isinstance(elements, list):
            continue
        for element_item in cast("list[object]", elements):
            if isinstance(element_item, dict):
                element = cast(dict[str, Any], element_item)
                if element.get("id") == element_id:
                    return snapshot, element
    return None


def _crop_element_png(
    png_bytes: bytes, bounds: dict[str, Any]
) -> str | None:
    """Crop one element out of a viewport screenshot; JPEG data URL.

    Returns ``None`` for malformed bounds or undecodable images instead of
    raising: a broken preview must never fail the report.
    """

    try:
        x = float(bounds["x"])
        y = float(bounds["y"])
        width = float(bounds["width"])
        height = float(bounds["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(value == value and abs(value) != float("inf") for value in (x, y, width, height)):
        return None
    try:
        image = Image.open(io.BytesIO(png_bytes))
        image.load()
    except Exception:  # noqa: BLE001 - any decode failure degrades to no preview
        return None
    left = max(0, int(x) - _CROP_PADDING)
    top = max(0, int(y) - _CROP_PADDING)
    right = min(image.width, int(x + width) + _CROP_PADDING)
    bottom = min(image.height, int(y + height) + _CROP_PADDING)
    if right - left < 2 or bottom - top < 2:
        return None
    crop = image.crop((left, top, right, bottom))
    if max(crop.size) > _CROP_MAX_EDGE:
        scale = _CROP_MAX_EDGE / max(crop.size)
        crop = crop.resize(
            (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
        )
    buffer = io.BytesIO()
    crop.convert("RGB").save(buffer, format="JPEG", quality=_CROP_JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def _inventory_locator_candidates(
    page: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """All selectors and xpaths recorded in one capture's node inventory."""

    selectors: list[str] = []
    xpaths: list[str] = []
    for key in ("links", "buttons", "inputs", "headings", "sections"):
        nodes = page.get(key)
        if not isinstance(nodes, list):
            continue
        for node_item in cast("list[object]", nodes):
            if not isinstance(node_item, dict):
                continue
            node = cast(Mapping[str, Any], node_item)
            selector = node.get("selector")
            xpath = node.get("xpath")
            if isinstance(selector, str) and selector:
                selectors.append(selector)
            if isinstance(xpath, str) and xpath:
                xpaths.append(xpath)
    return selectors, xpaths


def _shortest_unique_css(selector: str, all_selectors: list[str]) -> bool:
    """True when ``selector`` matches exactly one recorded node.

    Uniqueness is judged against the whole inventoried selector set of the
    page (the same set an operator's live page produced), not by syntax
    alone. A bare id selector is unique by construction.
    """

    if selector.startswith("#"):
        return True
    return all_selectors.count(selector) == 1


def locator_for_box(
    captures_by_url: Mapping[str, Mapping[str, Any]],
    *,
    box: Mapping[str, Any] | None,
    page_url_hint: str | None = None,
    tolerance_px: float = _LOCATOR_MATCH_TOLERANCE_PX,
) -> dict[str, Any] | None:
    """Verified-unique locator for any recorded box on the page.

    Matches geometrically against the operator-facing capture inventory
    (same tag-family distance rule as :func:`_locator_for_element`) and
    returns the best node's locator, uniqueness-checked against every
    inventoried node of the winning page. ``None`` when no inventoried
    node sits within ``tolerance_px``.
    """

    target = _box_tuple(box)
    if target is None:
        return None
    pages: list[tuple[str, Mapping[str, Any]]] = []
    if page_url_hint and page_url_hint in captures_by_url:
        pages.append((page_url_hint, captures_by_url[page_url_hint]))
    pages.extend(
        (url, page)
        for url, page in captures_by_url.items()
        if url != page_url_hint
    )
    best: tuple[float, str, Mapping[str, Any], list[str], list[str]] | None = None
    for page_url, page in pages:
        selectors, xpaths = _inventory_locator_candidates(page)
        if not selectors:
            continue
        for key in ("links", "buttons", "inputs", "headings", "sections"):
            nodes = page.get(key)
            if not isinstance(nodes, list):
                continue
            for node_item in cast("list[object]", nodes):
                if not isinstance(node_item, dict):
                    continue
                node = cast(Mapping[str, Any], node_item)
                raw_selector = node.get("selector")
                if not isinstance(raw_selector, str) or not raw_selector:
                    continue
                node_box = _box_tuple(cast(Mapping[str, Any], node.get("box")))
                if node_box is None:
                    continue
                distance = abs(node_box[0] - target[0]) + abs(
                    node_box[1] - target[1]
                )
                if best is None or distance < best[0]:
                    best = (distance, page_url, node, selectors, xpaths)
    if best is None or best[0] > tolerance_px:
        return None
    distance, page_url, node, selectors, xpaths = best
    selector = str(node.get("selector"))
    xpath = str(node.get("xpath") or "")
    result: dict[str, Any] = {
        "page_url": page_url,
        "match_distance_px": int(distance),
    }
    if _shortest_unique_css(selector, selectors):
        result["selector"] = selector
    if xpath and (xpath.startswith("//*[@id=") or xpaths.count(xpath) == 1):
        result["xpath"] = xpath
    if "selector" not in result and "xpath" not in result:
        return None
    return result


def element_chip_payload(
    *,
    bundle_path: Path,
    snapshots: Any,
    element_id: str,
    captures_by_url: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Preview + locator payload for one recorded element.

    Combines the run bundle's viewport snapshot (geometry, screenshot
    artifact) with the operator-facing capture inventory (locators). Any
    missing piece degrades that sub-feature only.
    """

    found = _element_by_id(snapshots, element_id)
    if found is None:
        return None
    snapshot, element = found
    payload: dict[str, Any] = {
        "element_id": element_id,
        "label": str(element.get("label") or "")[:160],
        "role": str(element.get("role") or ""),
        "bounds": element.get("bounds") or {},
        "viewport_id": snapshot.get("id"),
    }
    bounds = cast(dict[str, Any] | None, element.get("bounds"))
    screenshot_ref = cast(
        str | None, snapshot.get("screenshot_artifact") or snapshot.get("artifact")
    )
    if isinstance(bounds, dict) and isinstance(screenshot_ref, str) and screenshot_ref:
        png_bytes = _read_bundle_artifact(bundle_path, screenshot_ref)
        if png_bytes is not None:
            data_url = _crop_element_png(png_bytes, bounds)
            if data_url is not None:
                payload["preview"] = data_url
    if captures_by_url:
        locator = _locator_for_element(captures_by_url, element, bounds)
        if locator is not None:
            payload["locator"] = locator
    return payload


def _read_bundle_artifact(bundle_path: Path, reference: str) -> bytes | None:
    """Read one content-addressed artifact from a run bundle safely.

    Run-bundle screenshots live under ``<bundle>/artifacts/<sha256>``;
    references may be recorded with or without that directory prefix.
    """

    try:
        relative = PurePosixPath(reference)
    except ValueError:
        return None
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return None
    candidates = [bundle_path / Path(*relative.parts)]
    if len(relative.parts) == 1:
        candidates.append(bundle_path / "artifacts" / Path(*relative.parts))
    for candidate in candidates:
        try:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            return candidate.read_bytes()
        except OSError:
            continue
    return None


def _locator_for_element(
    captures_by_url: dict[str, dict[str, Any]],
    element: dict[str, Any],
    bounds: object,
) -> dict[str, Any] | None:
    """Pick a verified-unique locator for a recorded element.

    The element came from the run-agent extractor (which has no xpath), so
    we match it back to the capture inventory geometrically (nearest box of
    same tag family per page) and return that node's locator, uniqueness-
    checked against every inventoried node of the winning page.
    """

    if not isinstance(bounds, dict):
        return None
    target = _box_tuple(cast(Mapping[str, Any], bounds))
    if target is None:
        return None
    best: tuple[float, str, Mapping[str, Any], list[str], list[str]] | None = None
    for page_url, page in captures_by_url.items():
        selectors, xpaths = _inventory_locator_candidates(page)
        if not selectors:
            continue
        for key in ("links", "buttons", "inputs", "headings", "sections"):
            nodes = page.get(key)
            if not isinstance(nodes, list):
                continue
            for node_item in cast("list[object]", nodes):
                if not isinstance(node_item, dict):
                    continue
                node = cast(Mapping[str, Any], node_item)
                raw_selector = node.get("selector")
                if not isinstance(raw_selector, str) or not raw_selector:
                    continue
                box = _box_tuple(cast(Mapping[str, Any], node.get("box")))
                if box is None:
                    continue
                distance = abs(box[0] - target[0]) + abs(box[1] - target[1])
                if best is None or distance < best[0]:
                    best = (distance, page_url, node, selectors, xpaths)
    if best is None or best[0] > 64:
        return None
    distance, page_url, node, selectors, xpaths = best
    selector = str(node.get("selector"))
    xpath = str(node.get("xpath") or "")
    result: dict[str, Any] = {"page_url": page_url, "match_distance_px": int(distance)}
    if _shortest_unique_css(selector, selectors):
        result["selector"] = selector
    if xpath and (xpath.startswith("//*[@id=") or xpaths.count(xpath) == 1):
        result["xpath"] = xpath
    if "selector" not in result and "xpath" not in result:
        return None
    return result


def _box_tuple(bounds: object) -> tuple[float, float] | None:
    if not isinstance(bounds, Mapping):
        return None
    typed = cast(Mapping[str, Any], bounds)
    try:
        x = float(typed["x"])
        y = float(typed["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if x != x or y != y or abs(x) == float("inf") or abs(y) == float("inf"):
        return None
    return x, y


__all__ = [
    "alias_index_from_corpus",
    "aliases_in_prose",
    "element_chip_payload",
    "evidence_detail_for_alias",
    "locator_for_box",
    "prose_alias_segments",
]
