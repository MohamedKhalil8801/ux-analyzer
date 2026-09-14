"""Redesign pass orchestration: captures → proposals → validated attempt.

Two isolated model roles (ADR 0007): a per-page Proposer and one cross-page
Critic/Merger whose output deterministically overrides the per-page proposals.
Before publication every surviving proposal passes deterministic validation:
schema invariants, section references resolving into the persisted capture,
known principle ids, deliberate-choice presence, and bounded counts. Failures
never raise into the caller's run — they become attempt statuses.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from ux_analyzer.domain.redesign import (
    DeliberateChoiceCheck,
    DesignCategory,
    DesignProposal,
    Effort,
    Impact,
    KilledProposal,
    PageUnderstanding,
    RedesignAttempt,
    RedesignAttemptStatus,
    SectionReference,
)
from ux_analyzer.ports.models import ModelAttachment, ModelCallRecord

MAX_PROPOSALS_PER_PAGE = 12
MAX_PROPOSALS_TOTAL = 40
_MAX_REASON_LENGTH = 300
# Text-only payload budget for one proposer request: with segment pixels
# stripped into image attachments the page metadata stays far below this;
# only a regression that re-embeds data URLs would approach it.
_REDESIGN_JSON_PAYLOAD_MAX_BYTES = 1_000_000
_BOX_TOLERANCE_PX = 2.0
_MIN_TARGET_SIZE_PX = 44.0
_TARGET_WORDS = (
    "target",
    "tap",
    "tappable",
    "hit area",
    "hit-area",
    "touch target",
    "touch area",
    "hitbox",
    "hit box",
    "click area",
    "clickable area",
)
# A size judgment is required in addition to a target word so that
# accessibility write-ups which merely mention a tap target in passing
# (e.g. a contrast proposal) are not routed into the target-size guard.
_SIZE_CLAIM_WORDS = (
    "too small",
    "very small",
    "tiny",
    "undersized",
    "small",
    "larger",
    "bigger",
    "enlarge",
    "expand",
    "grow",
    "increase",
    "minimum",
    "at least",
    "44px",
    "45px",
    "48px",
)
_IMPACT_ORDER = {"high": 0, "medium": 1, "low": 2}
_EFFORT_ORDER = {"small": 0, "medium": 1, "large": 2}
_SAFE_DIAGNOSTIC_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class RedesignUnderstandingView(Protocol):
    """Structural view of a provider page-understanding value."""

    page_url: str
    intent: str
    audience_inference: str
    section_relationships: str


class RedesignProposalView(Protocol):
    """Structural view of a provider proposal (validated pydantic model)."""

    def model_dump(self) -> dict[str, object]: ...


class RedesignKilledView(Protocol):
    """Structural view of a critic-killed proposal."""

    proposal_id: str
    reason: str


class RedesignProposerPort(Protocol):
    """Port the pipeline needs from a proposer provider."""

    async def analyze(
        self,
        page_payload: dict[str, object],
        *,
        audience: str,
        principles: list[dict[str, object]],
        attachments: Sequence[ModelAttachment] = (),
    ) -> object: ...


class RedesignCriticPort(Protocol):
    """Port the pipeline needs from a critic/merger provider."""

    async def review(
        self,
        consolidated: list[dict[str, object]],
        page_payloads_digests: list[dict[str, object]],
        *,
        audience: str,
        principles: list[dict[str, object]],
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class RedesignPassOutcome:
    """One pass result: the attempt plus the capture digest it is tied to."""

    attempt: RedesignAttempt
    captures_digest: str
    # Sanitized transport audit records (role, model, attempts, retries,
    # failure diagnostics). Populated by the caller that owns the client;
    # persisted next to the payload so terminal model failures stay
    # debuggable after the run.
    model_call_records: tuple[ModelCallRecord, ...] = ()


def _reason(text: str) -> str:
    return text[:_MAX_REASON_LENGTH]


def _model_failure_details(error: BaseException) -> str:
    """Safe suffix for a ModelFailureError; never includes response content.

    The transport raises ModelFailureError with provider metadata (status
    code, error code/type, request id) and sanitized structural diagnostics
    (stage, mode, validation paths, content shape). Those fields are exactly
    what an operator needs to tell a misbehaving provider from a model that
    cannot satisfy the schema; the raw text stays in the transport records.
    """

    details: list[str] = []
    status_code = getattr(error, "status_code", None)
    if type(status_code) is int:
        details.append(f"status_code={status_code}")
    for attribute in ("error_code", "error_type", "request_id"):
        value = getattr(error, attribute, None)
        if (
            isinstance(value, str)
            and 0 < len(value) <= 128
            and "\r" not in value
            and "\n" not in value
        ):
            details.append(f"{attribute}={value}")
    diagnostics = getattr(error, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        safe = _safe_failure_diagnostics(cast(Mapping[object, object], diagnostics))
        if safe:
            details.append(f"diagnostics={safe}")
    if not details:
        return ""
    return f" ({'; '.join(details)})"


def _safe_failure_diagnostics(diagnostics: Mapping[object, object]) -> str:
    """Bounded safe projection of transport structural diagnostics."""

    parts: list[str] = []
    stage = diagnostics.get("stage")
    if isinstance(stage, str) and _SAFE_DIAGNOSTIC_TOKEN.fullmatch(stage):
        parts.append(f"stage={stage}")
    mode = diagnostics.get("response_mode")
    if isinstance(mode, str) and _SAFE_DIAGNOSTIC_TOKEN.fullmatch(mode):
        parts.append(f"mode={mode}")
    attempts = diagnostics.get("attempt_count")
    if type(attempts) is int and attempts > 0:
        parts.append(f"attempts={attempts}")
    finish_reason = diagnostics.get("finish_reason")
    if isinstance(finish_reason, str) and _SAFE_DIAGNOSTIC_TOKEN.fullmatch(
        finish_reason
    ):
        parts.append(f"finish_reason={finish_reason}")
    return " ".join(parts[:8])


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _segment_digest(capture: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    segments = capture.get("segments")
    if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)):
        for segment in cast(Sequence[object], segments):
            if isinstance(segment, Mapping):
                mapping = cast(Mapping[object, object], segment)
                digest.update(
                    str(mapping.get("data_url", "")).encode("utf-8", "replace")
                )
    return digest.hexdigest()


def captures_digest(captures: Mapping[str, Mapping[str, object]]) -> str:
    """Stable digest binding an attempt to the exact captures it used."""

    digest = hashlib.sha256()
    for url in sorted(captures):
        capture = captures[url]
        digest.update(
            _canonical(
                {
                    "url": url,
                    "title": capture.get("title", ""),
                    "document_height": capture.get("document_height", 0),
                    "segment_digest": _segment_digest(capture),
                }
            )
        )
    return digest.hexdigest()


def page_payload_digests(
    captures: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compact per-page digests for the critic/merger (no full images)."""

    digests: list[dict[str, object]] = []
    for url in sorted(captures):
        capture = captures[url]
        section_labels: list[str] = []
        sections = capture.get("sections")
        if isinstance(sections, Sequence) and not isinstance(sections, (str, bytes)):
            for section in cast(Sequence[object], sections):
                if isinstance(section, Mapping):
                    mapping = cast(Mapping[object, object], section)
                    if mapping.get("label"):
                        section_labels.append(str(mapping["label"])[:120])
        digests.append(
            {
                "url": url,
                "title": capture.get("title", ""),
                "document_height": capture.get("document_height", 0),
                "segment_count": _sequence_len(capture.get("segments")),
                "section_labels": section_labels[:50],
                "digest": _segment_digest(capture),
            }
        )
    return digests


def _sequence_len(value: object) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(cast(Sequence[object], value))
    return 0


def _segment_payload_bytes(segment: Mapping[object, object]) -> bytes | None:
    """Decode one persisted capture segment's JPEG payload; None when unusable."""

    data_url = segment.get("data_url")
    if not isinstance(data_url, str) or not data_url.startswith(
        "data:image/jpeg;base64,"
    ):
        return None
    try:
        encoded = data_url.split(",", maxsplit=1)[1]
        return base64.b64decode(encoded, validate=False)
    except (ValueError, TypeError):
        return None


def _json_capture_view(capture: Mapping[str, object]) -> dict[str, object]:
    """Text-only page payload for the proposer (pixels ride as attachments).

    The persisted capture embeds every segment as a base64 JPEG data URL;
    embedding those bytes again inside the JSON request would roughly double
    the transport body. The JSON therefore carries ordered segment metadata
    (index, y-offset, height, digest), and the pixel bytes travel exactly
    once, as image attachments (ADR 0007: "segment images attached in order
    with y-offsets").
    """

    view = dict(capture)
    segments = capture.get("segments")
    metadata: list[dict[str, object]] = []
    if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)):
        for index, raw in enumerate(cast(Sequence[object], segments)):
            if not isinstance(raw, Mapping):
                continue
            segment = cast(Mapping[object, object], raw)
            payload = _segment_payload_bytes(segment)
            if payload is None:
                digest = hashlib.sha256(
                    str(segment.get("data_url", "")).encode("utf-8", "replace")
                ).hexdigest()
            else:
                digest = hashlib.sha256(payload).hexdigest()
            metadata.append(
                {
                    "index": index,
                    "y_offset": segment.get("y_offset", 0),
                    "height": segment.get("height", 0),
                    "digest": digest,
                }
            )
    view["segments"] = metadata
    return view


def _segment_attachments(
    capture: Mapping[str, object],
    scratch_dir: Path,
) -> tuple[ModelAttachment, ...]:
    """Materialize ordered capture segments as model attachments.

    The capture persists each ~2000px segment as a bounded JPEG data URL;
    the transport reads real files, so segments are written (in capture
    order, which is y-offset order) into the per-pass scratch directory.
    Unusable segments are skipped rather than failing the pass, matching
    the best-effort contract.
    """

    segments = capture.get("segments")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        return ()
    attachments: list[ModelAttachment] = []
    for index, raw in enumerate(cast(Sequence[object], segments)):
        if not isinstance(raw, Mapping):
            continue
        payload = _segment_payload_bytes(cast(Mapping[object, object], raw))
        if payload is None:
            continue
        digest = hashlib.sha256(payload).hexdigest()
        path = scratch_dir / f"segment-{index:03d}.jpg"
        try:
            path.write_bytes(payload)
        except OSError:
            continue
        attachments.append(
            ModelAttachment(
                evidence_id=f"capture:segment:{digest[:16]}",
                path=path,
                media_type="image/jpeg",
                sha256=digest,
            )
        )
    return tuple(attachments)


def _normalized_label(value: str) -> str:
    return " ".join(value.lower().split())


def _inventory_labels_and_boxes(
    capture: Mapping[str, object],
) -> tuple[list[str], list[dict[str, float]]]:
    labels: list[str] = []
    boxes: list[dict[str, float]] = []
    for key in ("sections", "headings", "forms", "buttons", "inputs", "links"):
        entries = capture.get(key)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in cast(Sequence[object], entries):
            if not isinstance(entry, Mapping):
                continue
            item = cast(Mapping[object, object], entry)
            label = item.get("label")
            if isinstance(label, str) and label.strip():
                labels.append(_normalized_label(label))
            box = item.get("box")
            if isinstance(box, Mapping):
                try:
                    boxes.append(
                        {
                            "x": float(box.get("x", 0)),  # type: ignore[arg-type]
                            "y": float(box.get("y", 0)),  # type: ignore[arg-type]
                            "w": float(box.get("w", box.get("width", 0))),  # type: ignore[arg-type]
                            "h": float(box.get("h", box.get("height", 0))),  # type: ignore[arg-type]
                        }
                    )
                except (TypeError, ValueError):
                    continue
    return labels, boxes


def _mentions_target_size(texts: Sequence[str]) -> bool:
    """True when the proposal is actually about hit-target size.

    ``category: accessibility`` also covers color contrast, keyboard focus,
    and other non-target issues; the deterministic guard must only gate
    claims that ask for a bigger tappable surface. A target word alone is
    not enough (a contrast write-up can mention "tap target" in passing):
    the proposal must also make a size judgment about it.
    """

    combined = " ".join(texts).lower()
    mentions_target = any(word in combined for word in _TARGET_WORDS)
    makes_size_judgment = any(word in combined for word in _SIZE_CLAIM_WORDS)
    return mentions_target and makes_size_judgment


def _interactive_inventory_entries(
    capture: Mapping[str, object],
) -> list[tuple[str, dict[str, float], dict[str, float]]]:
    """Label / own-box / effective tap-box triples for tappable controls.

    Only buttons, inputs, and links count as interactive; sections and
    headings describe regions, not targets. The pairs feed the deterministic
    target-size guard so hit-target claims are checked against the real
    control geometry instead of whatever the model inferred from pixels.
    """

    entries: list[tuple[str, dict[str, float], dict[str, float]]] = []
    for key in ("buttons", "inputs", "links"):
        raw = capture.get(key)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            continue
        for item in cast(Sequence[object], raw):
            if not isinstance(item, Mapping):
                continue
            entry = cast(Mapping[object, object], item)
            label = _normalized_label(str(entry.get("label", "")))
            box_value = entry.get("box")
            box = (
                _box_from_mapping(cast(Mapping[object, object], box_value))
                if isinstance(box_value, Mapping)
                else None
            )
            if box is None:
                continue
            # The effective tappable surface (capture-time: the control
            # itself or its closest tap-reacting ancestor) is the real
            # measure; older sidecars without tap_box fall back to the
            # control's own box.
            raw_tap_box = entry.get("tap_box")
            tap_box = (
                _box_from_mapping(cast(Mapping[object, object], raw_tap_box))
                if isinstance(raw_tap_box, Mapping)
                else box
            )
            entries.append((label, box, tap_box or box))
    return entries


def _accessibility_target_size_reason(
    ref: Mapping[object, object], capture: Mapping[str, object]
) -> str | None:
    """Deterministic guard for hit-target (WCAG 2.5.8-like) claims.

    A ``category: accessibility`` proposal that wants a bigger tap target
    must point at a control whose *effective tappable surface* is genuinely
    below ``_MIN_TARGET_SIZE_PX`` on at least one axis. The effective
    surface is the control itself, or the closest ancestor that reacts to
    user taps (a card whose click handler wraps a small child button, a
    label wrapping a checkbox); the capture inventory records it as
    ``tap_box``. When the referral resolves to a control with an already
    large effective target -- e.g. a whole video card that happens to carry
    a small caption label -- the claim is contradicted by the capture and
    the proposal is rejected here rather than published. The referral is
    resolved to the single best-matching control (label match wins; box
    overlap is scored by intersection-over-union so a ref that spans several
    controls measures the one it most overlaps). Refs that resolve to no
    interactive control at all are not target-size claims and are left
    alone.
    """

    label = _normalized_label(str(ref.get("section_label", "")))
    box_value = ref.get("box")
    ref_box = (
        _box_from_mapping(cast(Mapping[object, object], box_value))
        if isinstance(box_value, Mapping)
        else None
    )
    best: tuple[str, dict[str, float]] | None = None
    best_iou = 0.0
    for entry_label, entry_box, entry_tap_box in _interactive_inventory_entries(capture):
        if label and entry_label and label in entry_label:
            # The model named the control; that is the strongest signal.
            best = (entry_label, entry_tap_box)
            break
        if ref_box is not None:
            iou = _box_iou(ref_box, entry_box)
            if iou > best_iou:
                best_iou = iou
                best = (entry_label, entry_tap_box)
    if best is None:
        return None
    entry_label, tap_box = best
    if tap_box["w"] >= _MIN_TARGET_SIZE_PX and tap_box["h"] >= _MIN_TARGET_SIZE_PX:
        surface = "effective tap target"
        return (
            "hit-target claim references a control whose "
            f"{surface} is already at least "
            f"{_MIN_TARGET_SIZE_PX:.0f}px on both axes "
            f"({tap_box['w']:.0f}x{tap_box['h']:.0f} in the capture "
            f"inventory{_tap_surface_note(entry_label)})"
        )
    return None


def _tap_surface_note(label: str) -> str:
    """Short human suffix naming which inventoried control matched."""

    if not label:
        return ", unnamed control"
    return f', "{label[:44]}"'


def _box_iou(ref: dict[str, float], box: dict[str, float]) -> float:
    """Intersection-over-union of two axis-aligned boxes ([0, 1])."""

    overlap_x = min(ref["x"] + ref["w"], box["x"] + box["w"]) - max(ref["x"], box["x"])
    overlap_y = min(ref["y"] + ref["h"], box["y"] + box["h"]) - max(ref["y"], box["y"])
    if overlap_x <= 0 or overlap_y <= 0:
        return 0.0
    intersection = overlap_x * overlap_y
    area_ref = max(ref["w"] * ref["h"], 0.01)
    area_box = max(box["w"] * box["h"], 0.01)
    return intersection / (area_ref + area_box - intersection)


def _boxes_overlap(ref: dict[str, float], box: dict[str, float]) -> bool:
    overlap_x = min(ref["x"] + ref["w"], box["x"] + box["w"]) - max(ref["x"], box["x"])
    overlap_y = min(ref["y"] + ref["h"], box["y"] + box["h"]) - max(ref["y"], box["y"])
    return overlap_x > 0 and overlap_y > 0


def _box_from_mapping(box: Mapping[object, object]) -> dict[str, float] | None:
    try:
        return {
            "x": float(box.get("x", 0)),  # type: ignore[arg-type]
            "y": float(box.get("y", 0)),  # type: ignore[arg-type]
            "w": float(box.get("w", box.get("width", 0))),  # type: ignore[arg-type]
            "h": float(box.get("h", box.get("height", 0))),  # type: ignore[arg-type]
        }
    except (TypeError, ValueError):
        return None


def _normalize_ref_box(box: object) -> dict[str, float] | None:
    """Normalize a model-emitted ref box to the canonical key set.

    Proposers sometimes mirror the capture inventory's ``w``/``h`` spelling
    (or add extra keys) when echoing a control's bounds; the persisted
    contract wants exactly ``x, y, width, height``. This maps the aliases,
    drops unknown keys, and returns ``None`` when the result still misses a
    required key so the strict domain check keeps rejecting genuinely
    malformed boxes.
    """

    if not isinstance(box, Mapping):
        return None
    raw = {str(key): item for key, item in cast(Mapping[object, object], box).items()}
    aliased = {
        {"w": "width", "h": "height"}.get(key, key): item for key, item in raw.items()
    }
    canonical = {
        key: item
        for key, item in aliased.items()
        if key in {"x", "y", "width", "height"}
    }
    if set(canonical) != {"x", "y", "width", "height"}:
        return None
    return cast(dict[str, float], canonical)


def section_ref_resolves(
    ref: Mapping[object, object], capture: Mapping[str, object]
) -> bool:
    """Deterministic check that a reference exists in the persisted capture.

    A reference resolves when its label appears in the capture's inventory
    labels or its box overlaps an inventoried element box, and its vertical
    extent stays within the captured document.
    """

    labels, boxes = _inventory_labels_and_boxes(capture)
    label = _normalized_label(str(ref.get("section_label", "")))
    label_match = bool(label) and any(label in entry for entry in labels)
    box_value = ref.get("box")
    box = (
        _box_from_mapping(cast(Mapping[object, object], box_value))
        if isinstance(box_value, Mapping)
        else None
    )
    box_match = box is not None and any(
        _boxes_overlap(box, entry) for entry in boxes
    )
    if not (label_match or box_match):
        return False
    if box is None:
        return True
    document_height = capture.get("document_height")
    if isinstance(document_height, (int, float)) and not isinstance(
        document_height, bool
    ):
        return box["y"] + box["h"] <= float(document_height) + _BOX_TOLERANCE_PX
    return True


def _attempt(
    *,
    attempt_id: str,
    status: RedesignAttemptStatus,
    proposals: tuple[DesignProposal, ...] = (),
    killed: tuple[KilledProposal, ...] = (),
    understanding: tuple[PageUnderstanding, ...] = (),
    consistency_notes: tuple[str, ...] = (),
    audience: str = "",
    unavailable_reason: str = "",
    rejection_reasons: tuple[str, ...] = (),
    pack_version: str = "",
) -> RedesignAttempt:
    return RedesignAttempt(
        attempt_id=attempt_id,
        status=status,
        proposals=proposals,
        killed=killed,
        page_understanding=understanding,
        consistency_notes=consistency_notes,
        pack_version=pack_version,
        audience=audience,
        created_at=datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        unavailable_reason=unavailable_reason,
        rejection_reasons=rejection_reasons,
    )


def _sort_key(proposal: DesignProposal) -> tuple[int, int, str]:
    return (
        _IMPACT_ORDER.get(proposal.impact.value, 3),
        _EFFORT_ORDER.get(proposal.effort.value, 3),
        proposal.proposal_id,
    )


def _validate_final_proposals(
    final_proposals: Sequence[Mapping[str, object]],
    captures: Mapping[str, Mapping[str, object]],
    *,
    known_ids: frozenset[str],
) -> tuple[tuple[DesignProposal, ...], tuple[str, ...]]:
    """Deterministic validation gate; returns (proposals, rejection reasons)."""

    reasons: list[str] = []
    validated: list[DesignProposal] = []
    per_page_counts: dict[str, int] = {}
    for proposal in final_proposals:
        proposal_id = str(proposal.get("proposal_id", ""))
        page_url = str(proposal.get("page_url", ""))
        principle_ids = proposal.get("principle_ids")
        if not isinstance(principle_ids, Sequence) or isinstance(
            principle_ids, (str, bytes)
        ):
            reasons.append(_reason(f"{proposal_id}: principle_ids must be a list"))
            continue
        unknown = [
            str(item)
            for item in cast(Sequence[object], principle_ids)
            if str(item) not in known_ids
        ]
        if unknown:
            # A proposal bound to principles outside the pack cannot be
            # published; record why and drop it (same as dangling refs).
            reasons.append(
                _reason(f"{proposal_id}: unknown principle ids: {unknown}")
            )
            continue
        capture = captures.get(page_url)
        if capture is None:
            reasons.append(
                _reason(f"{proposal_id}: no persisted capture for {page_url}")
            )
            continue
        refs = proposal.get("section_refs")
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
            reasons.append(_reason(f"{proposal_id}: section_refs must be a list"))
            continue
        typed_refs = tuple(
            cast(Mapping[object, object], ref)
            for ref in cast(Sequence[object], refs)
            if isinstance(ref, Mapping)
        )
        dangling = False
        for ref in typed_refs:
            if not section_ref_resolves(ref, capture):
                reasons.append(
                    _reason(
                        f"{proposal_id}: dangling section reference on {page_url}"
                    )
                )
                dangling = True
                break
        if dangling:
            continue
        # Deterministic guard: hit-target (accessibility) claims must not
        # point at a control the capture inventory already sizes at or above
        # the 44px minimum (e.g. a whole video card carrying a caption text).
        # Only proposals whose own text asks for a bigger tappable surface
        # are gated; color-contrast and other non-target accessibility
        # proposals pass through.
        if str(proposal.get("category", "")) == "accessibility" and (
            _mentions_target_size(
                (
                    str(proposal.get("title", "")),
                    str(proposal.get("observation", "")),
                    str(proposal.get("change", "")),
                    str(proposal.get("rationale", "")),
                )
            )
        ):
            target_violation = False
            for ref in typed_refs:
                target_reason = _accessibility_target_size_reason(ref, capture)
                if target_reason is not None:
                    reasons.append(
                        _reason(f"{proposal_id}: {target_reason}")
                    )
                    target_violation = True
                    break
            if target_violation:
                continue
        try:
            deliberate = proposal.get("deliberate_choice_check")
            typed_deliberate = (
                cast(Mapping[object, object], deliberate)
                if isinstance(deliberate, Mapping)
                else None
            )
            also_affects = proposal.get("also_affects", [])
            typed_also_affects = (
                cast(Sequence[object], also_affects)
                if isinstance(also_affects, Sequence)
                and not isinstance(also_affects, (str, bytes))
                else ()
            )
            validated.append(
                DesignProposal(
                    proposal_id=proposal_id,
                    page_url=page_url,
                    category=DesignCategory(str(proposal.get("category", ""))),
                    title=str(proposal.get("title", "")),
                    observation=str(proposal.get("observation", "")),
                    rationale=str(proposal.get("rationale", "")),
                    change=str(proposal.get("change", "")),
                    principle_ids=tuple(
                        str(item) for item in cast(Sequence[object], principle_ids)
                    ),
                    impact=Impact(str(proposal.get("impact", ""))),
                    effort=Effort(str(proposal.get("effort", ""))),
                    section_refs=tuple(
                        SectionReference(
                            url=str(ref.get("url", "")),
                            section_label=str(ref.get("section_label", "")),
                            box=cast(
                                "dict[str, float]",
                                (
                                    normalized_box
                                    if (
                                        normalized_box := _normalize_ref_box(
                                            ref.get("box")
                                        )
                                    )
                                    is not None
                                    else (
                                        dict(
                                            cast(
                                                Mapping[object, object],
                                                ref.get("box"),
                                            )
                                        )
                                        if isinstance(ref.get("box"), Mapping)
                                        else {}
                                    )
                                ),
                            ),
                            summary=str(ref.get("summary", "")),
                        )
                        for ref in typed_refs
                    ),
                    also_affects=tuple(
                        str(item) for item in typed_also_affects
                    ),
                    deliberate_choice_check=(
                        None
                        if typed_deliberate is None
                        else DeliberateChoiceCheck(
                            pattern=str(typed_deliberate.get("pattern", "")),
                            rationale=str(typed_deliberate.get("rationale", "")),
                        )
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            reasons.append(_reason(f"{proposal_id}: {error}"))
            continue
        # Only successfully validated proposals count toward the per-page
        # bound, so malformed entries never inflate the count of a page whose
        # valid set is in bounds (they are rejected with their own reasons).
        per_page_counts[page_url] = per_page_counts.get(page_url, 0) + 1
    for page_url, count in per_page_counts.items():
        if count > MAX_PROPOSALS_PER_PAGE:
            reasons.append(
                _reason(
                    f"{page_url}: {count} proposals exceed the per-page bound "
                    f"of {MAX_PROPOSALS_PER_PAGE}"
                )
            )
    validated_total = sum(per_page_counts.values())
    if validated_total > MAX_PROPOSALS_TOTAL:
        reasons.append(
            _reason(
                f"{validated_total} proposals exceed the total bound of "
                f"{MAX_PROPOSALS_TOTAL}"
            )
        )
    return tuple(validated), tuple(reasons)


async def run_redesign_pass(
    captures: Mapping[str, Mapping[str, object]],
    *,
    audience: str,
    proposer: RedesignProposerPort,
    critic: RedesignCriticPort,
    attempt_id: str,
    principle_pack: Sequence[Mapping[str, object]] = (),
    principle_ids: Sequence[str] = (),
    principle_pack_version: str = "",
) -> RedesignPassOutcome:
    """Run one redesign pass; never raises into the caller's run.

    The principle pack is injected (as payload dicts, the known-id tuple,
    and the pack version) so this module stays provider-agnostic; the
    caller assembles it from the versioned static pack.
    """

    if not captures:
        return RedesignPassOutcome(
            _attempt(
                attempt_id=attempt_id,
                status=RedesignAttemptStatus.UNAVAILABLE,
                audience=audience,
                unavailable_reason="no page captures available",
            ),
            "",
        )
    digest = captures_digest(captures)
    pack_payload = [dict(item) for item in principle_pack]
    known_ids = frozenset(principle_ids)

    consolidated: list[dict[str, object]] = []
    understanding: list[PageUnderstanding] = []
    with tempfile.TemporaryDirectory(
        prefix="uxa-redesign-attachments-"
    ) as scratch_dir_name:
        scratch_dir = Path(scratch_dir_name)
        for url in sorted(captures):
            capture: dict[str, object] = dict(captures[url])
            try:
                view = _json_capture_view(capture)
                if len(_canonical(view)) > _REDESIGN_JSON_PAYLOAD_MAX_BYTES:
                    return RedesignPassOutcome(
                        _attempt(
                            attempt_id=attempt_id,
                            status=RedesignAttemptStatus.UNAVAILABLE,
                            audience=audience,
                            unavailable_reason=_reason(
                                f"page payload for {url} exceeds the redesign "
                                "JSON budget; segment pixels must ride as image "
                                "attachments"
                            ),
                        ),
                        digest,
                    )
                attachments = _segment_attachments(capture, scratch_dir)
                response = await proposer.analyze(
                    view,
                    audience=audience,
                    principles=pack_payload,
                    attachments=attachments,
                )
            except Exception as error:  # noqa: BLE001 - best-effort by contract
                return RedesignPassOutcome(
                    _attempt(
                        attempt_id=attempt_id,
                        status=RedesignAttemptStatus.UNAVAILABLE,
                        audience=audience,
                        unavailable_reason=_reason(
                            f"proposer failed for {url}: "
                            f"{type(error).__name__}: {error}"
                            f"{_model_failure_details(error)}"
                        ),
                    ),
                    digest,
                )
            understanding_value = getattr(response, "page_understanding", None)
            proposals_value = getattr(response, "proposals", None)
            if understanding_value is None or not isinstance(
                proposals_value, Sequence
            ) or isinstance(proposals_value, (str, bytes)):
                return RedesignPassOutcome(
                    _attempt(
                        attempt_id=attempt_id,
                        status=RedesignAttemptStatus.UNAVAILABLE,
                        audience=audience,
                        unavailable_reason=_reason(
                            f"proposer returned an unusable response for {url}"
                        ),
                    ),
                    digest,
                )
            understanding_view = cast(RedesignUnderstandingView, understanding_value)
            understanding.append(
                PageUnderstanding(
                    # The canonical capture key is the only trusted page URL;
                    # the model-echoed URL is never validated against the
                    # capture set, so the capture key wins.
                    page_url=url,
                    intent=understanding_view.intent,
                    audience_inference=understanding_view.audience_inference,
                    section_relationships=understanding_view.section_relationships,
                )
            )
            for proposal in cast(Sequence[object], proposals_value):
                consolidated.append(
                    cast(RedesignProposalView, proposal).model_dump()
                )

    # No pre-critic per-page bound here: the critic is charged with killing
    # and merging duplicates, so an over-producing page must reach it and be
    # trimmed. The deterministic per-page and total bounds are enforced on the
    # critic's final proposals in _validate_final_proposals.
    try:
        critic_response = await critic.review(
            consolidated=consolidated,
            page_payloads_digests=page_payload_digests(captures),
            audience=audience,
            principles=pack_payload,
        )
    except Exception as error:  # noqa: BLE001 - best-effort by contract
        return RedesignPassOutcome(
            _attempt(
                attempt_id=attempt_id,
                status=RedesignAttemptStatus.UNAVAILABLE,
                understanding=tuple(understanding),
                audience=audience,
                unavailable_reason=_reason(
                    f"critic/merger failed: {type(error).__name__}: {error}"
                    f"{_model_failure_details(error)}"
                ),
            ),
            digest,
        )

    final_value = getattr(critic_response, "final_proposals", None)
    killed_value = getattr(critic_response, "killed", None)
    notes_value = getattr(critic_response, "consistency_notes", None)
    if not isinstance(final_value, Sequence) or isinstance(final_value, (str, bytes)):
        return RedesignPassOutcome(
            _attempt(
                attempt_id=attempt_id,
                status=RedesignAttemptStatus.UNAVAILABLE,
                understanding=tuple(understanding),
                audience=audience,
                unavailable_reason="critic/merger returned an unusable response",
            ),
            digest,
        )
    final_dicts = [
        cast(RedesignProposalView, item).model_dump()
        for item in cast(Sequence[object], final_value)
    ]
    validated, reasons = _validate_final_proposals(
        final_dicts, captures, known_ids=known_ids
    )
    killed_sequence: Sequence[object] = (
        cast(Sequence[object], killed_value)
        if isinstance(killed_value, Sequence)
        and not isinstance(killed_value, (str, bytes))
        else ()
    )
    killed = tuple(
        KilledProposal(
            proposal_id=cast(RedesignKilledView, item).proposal_id,
            reason=cast(RedesignKilledView, item).reason,
        )
        for item in killed_sequence
    )
    notes_sequence: Sequence[object] = (
        cast(Sequence[object], notes_value)
        if isinstance(notes_value, Sequence)
        and not isinstance(notes_value, (str, bytes))
        else ()
    )
    consistency_notes = tuple(
        str(note) for note in notes_sequence if str(note).strip()
    )
    if reasons:
        # A rejected attempt keeps the proposals that did pass the gate on
        # record (each surviving item is fully validated); the rejection
        # reasons explain what was dropped and why. This preserves
        # observability when a weak model mixes one malformed proposal into
        # an otherwise valid set.
        return RedesignPassOutcome(
            _attempt(
                attempt_id=attempt_id,
                status=RedesignAttemptStatus.REJECTED,
                proposals=tuple(sorted(validated, key=_sort_key)),
                understanding=tuple(understanding),
                consistency_notes=consistency_notes,
                audience=audience,
                rejection_reasons=reasons,
            ),
            digest,
        )
    ordered = tuple(sorted(validated, key=_sort_key))
    status = (
        RedesignAttemptStatus.ACCEPTED
        if ordered
        else RedesignAttemptStatus.NO_PROPOSALS
    )
    return RedesignPassOutcome(
        _attempt(
            attempt_id=attempt_id,
            status=status,
            proposals=ordered,
            killed=killed,
            understanding=tuple(understanding),
            consistency_notes=consistency_notes,
            audience=audience,
            pack_version=principle_pack_version,
        ),
        digest,
    )


# Re-exported for pipeline consumers (base type imported for typing only).
__all__ = [
    "MAX_PROPOSALS_PER_PAGE",
    "MAX_PROPOSALS_TOTAL",
    "RedesignCriticPort",
    "RedesignKilledView",
    "RedesignPassOutcome",
    "RedesignProposalView",
    "RedesignProposerPort",
    "RedesignUnderstandingView",
    "captures_digest",
    "page_payload_digests",
    "run_redesign_pass",
    "section_ref_resolves",
]
