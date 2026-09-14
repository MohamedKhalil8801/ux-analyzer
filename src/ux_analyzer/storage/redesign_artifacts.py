"""Immutable experiment-scoped redesign attempt storage (plan Task 5).

Layout: ``<output>/redesign/<attempt-id>/index.json`` plus ``payload.json``.
The index records the SHA-256 digest of the canonical payload bytes so
corruption is detectable. Attempts are never overwritten; regeneration
creates a new attempt. Selection returns the newest *valid* attempt and
records every skipped (corrupt) newer attempt — corruption is never silent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

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
from ux_analyzer.storage.run_bundle import (
    secure_is_link_or_reparse,
    secure_read_bytes,
    secure_write_bytes,
)

REDESIGN_INDEX_SCHEMA = "redesign-index-v1"
REDESIGN_PAYLOAD_SCHEMA = "redesign-attempt-payload-v1"
REDESIGN_ATTEMPT_DIRNAME = "redesign"
MAX_REDESIGN_JSON_BYTES = 8 * 1024 * 1024
_REDESIGN_INDEX_MAX_BYTES = 64 * 1024
_ATTEMPT_ID_PATTERN = re.compile(
    r"^redesign-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"
)


class RedesignArtifactError(ValueError):
    """Raised when redesign artifact state is invalid or cannot be trusted."""


@dataclass(frozen=True, slots=True)
class RedesignAttemptSelection:
    """Newest valid attempt plus any corruption notes from newer attempts."""

    attempt: RedesignAttempt | None
    skipped: tuple[str, ...] = ()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def new_attempt_id() -> str:
    """Chronologically sortable, collision-safe attempt id."""

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"redesign-{stamp}-{uuid4().hex[:8]}"


def _check_deliberate(value: Mapping[str, object]) -> DeliberateChoiceCheck:
    return DeliberateChoiceCheck(
        pattern=str(value.get("pattern", "")),
        rationale=str(value.get("rationale", "")),
    )


def _section_ref_from_dict(value: object) -> SectionReference:
    if not isinstance(value, Mapping):
        raise RedesignArtifactError("section ref must be an object")
    mapping = cast(Mapping[str, object], value)
    box = mapping.get("box")
    if not isinstance(box, Mapping):
        raise RedesignArtifactError("section ref box must be an object")
    return SectionReference(
        url=str(mapping.get("url", "")),
        section_label=str(mapping.get("section_label", "")),
        box={
            "x": float(cast(Mapping[str, object], box).get("x", 0)),  # type: ignore[arg-type]
            "y": float(cast(Mapping[str, object], box).get("y", 0)),  # type: ignore[arg-type]
            "width": float(cast(Mapping[str, object], box).get("width", 0)),  # type: ignore[arg-type]
            "height": float(cast(Mapping[str, object], box).get("height", 0)),  # type: ignore[arg-type]
        },
        summary=str(mapping.get("summary", "")),
    )


def _proposal_from_dict(value: object) -> DesignProposal:
    if not isinstance(value, Mapping):
        raise RedesignArtifactError("proposal must be an object")
    mapping = cast(Mapping[str, object], value)
    refs = mapping.get("section_refs")
    if not isinstance(refs, list):
        raise RedesignArtifactError("section_refs must be a list")
    deliberate = mapping.get("deliberate_choice_check")
    also_affects = mapping.get("also_affects", [])
    return DesignProposal(
        proposal_id=str(mapping.get("proposal_id", "")),
        page_url=str(mapping.get("page_url", "")),
        category=DesignCategory(str(mapping.get("category", ""))),
        title=str(mapping.get("title", "")),
        observation=str(mapping.get("observation", "")),
        rationale=str(mapping.get("rationale", "")),
        change=str(mapping.get("change", "")),
        principle_ids=tuple(
            str(item) for item in cast(list[object], mapping.get("principle_ids", []))
        ),
        impact=Impact(str(mapping.get("impact", ""))),
        effort=Effort(str(mapping.get("effort", ""))),
        section_refs=tuple(
            _section_ref_from_dict(item) for item in cast(list[object], refs)
        ),
        also_affects=tuple(str(item) for item in cast(list[object], also_affects)),
        deliberate_choice_check=(
            None
            if deliberate is None
            else _check_deliberate(cast(Mapping[str, object], deliberate))
        ),
    )


def _proposal_to_dict(proposal: DesignProposal) -> dict[str, object]:
    return {
        "proposal_id": proposal.proposal_id,
        "page_url": proposal.page_url,
        "category": proposal.category.value,
        "title": proposal.title,
        "observation": proposal.observation,
        "rationale": proposal.rationale,
        "change": proposal.change,
        "principle_ids": list(proposal.principle_ids),
        "impact": proposal.impact.value,
        "effort": proposal.effort.value,
        "section_refs": [
            {
                "url": ref.url,
                "section_label": ref.section_label,
                "box": dict(ref.box),
                "summary": ref.summary,
            }
            for ref in proposal.section_refs
        ],
        "also_affects": list(proposal.also_affects),
        "deliberate_choice_check": (
            None
            if proposal.deliberate_choice_check is None
            else {
                "pattern": proposal.deliberate_choice_check.pattern,
                "rationale": proposal.deliberate_choice_check.rationale,
            }
        ),
    }


def attempt_to_payload(
    attempt: RedesignAttempt, *, captures_digest: str = ""
) -> dict[str, object]:
    """Serialize an attempt into the persisted payload document."""

    return {
        "schema": REDESIGN_PAYLOAD_SCHEMA,
        "attempt_id": attempt.attempt_id,
        "status": attempt.status.value,
        "created_at": attempt.created_at or "",
        "audience": attempt.audience,
        "pack_version": attempt.pack_version,
        "captures_digest": captures_digest,
        "unavailable_reason": attempt.unavailable_reason,
        "rejection_reasons": list(attempt.rejection_reasons),
        "consistency_notes": list(attempt.consistency_notes),
        "page_understanding": [
            {
                "page_url": item.page_url,
                "intent": item.intent,
                "audience_inference": item.audience_inference,
                "section_relationships": item.section_relationships,
            }
            for item in attempt.page_understanding
        ],
        "proposals": [_proposal_to_dict(item) for item in attempt.proposals],
        "killed": [
            {"proposal_id": item.proposal_id, "reason": item.reason}
            for item in attempt.killed
        ],
    }


def attempt_from_payload(value: object) -> RedesignAttempt:
    """Rebuild an attempt from a payload document; invalid shapes raise."""

    if not isinstance(value, dict):
        raise RedesignArtifactError("payload must be a JSON object")
    payload = cast(dict[str, object], value)
    if payload.get("schema") != REDESIGN_PAYLOAD_SCHEMA:
        raise RedesignArtifactError("unsupported payload schema")
    status_text = str(payload.get("status", ""))
    try:
        status = RedesignAttemptStatus(status_text)
    except ValueError as error:
        raise RedesignArtifactError("unknown attempt status") from error
    proposals_value = payload.get("proposals", [])
    understanding_value = payload.get("page_understanding", [])
    killed_value = payload.get("killed", [])
    if not isinstance(proposals_value, list):
        raise RedesignArtifactError("proposals must be a list")
    if not isinstance(understanding_value, list):
        raise RedesignArtifactError("page_understanding must be a list")
    if not isinstance(killed_value, list):
        raise RedesignArtifactError("killed must be a list")
    rejection = payload.get("rejection_reasons", [])
    notes = payload.get("consistency_notes", [])
    if not isinstance(rejection, list) or not isinstance(notes, list):
        raise RedesignArtifactError("reason and note lists must be arrays")
    typed_rejection = cast(list[object], rejection)
    typed_notes = cast(list[object], notes)
    created_at = str(payload.get("created_at", "") or "")
    return RedesignAttempt(
        attempt_id=str(payload.get("attempt_id", "")),
        status=status,
        proposals=tuple(
            _proposal_from_dict(item) for item in cast(list[object], proposals_value)
        ),
        killed=tuple(
            _killed_from_dict(item) for item in cast(list[object], killed_value)
        ),
        page_understanding=tuple(
            _understanding_from_dict(item)
            for item in cast(list[object], understanding_value)
        ),
        consistency_notes=_string_tuple(typed_notes),
        pack_version=str(payload.get("pack_version", "")),
        audience=str(payload.get("audience", "")),
        created_at=created_at or None,
        unavailable_reason=str(payload.get("unavailable_reason", "")),
        rejection_reasons=_string_tuple(typed_rejection),
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RedesignArtifactError("expected a list of strings")
    return tuple(str(item) for item in cast(list[object], value))


def _killed_from_dict(value: object) -> KilledProposal:
    if not isinstance(value, Mapping):
        raise RedesignArtifactError("killed entry must be an object")
    mapping = cast(Mapping[str, object], value)
    return KilledProposal(
        proposal_id=str(mapping.get("proposal_id", "")),
        reason=str(mapping.get("reason", "")),
    )


def _understanding_from_dict(value: object) -> PageUnderstanding:
    if not isinstance(value, Mapping):
        raise RedesignArtifactError("page understanding entry must be an object")
    mapping = cast(Mapping[str, object], value)
    return PageUnderstanding(
        page_url=str(mapping.get("page_url", "")),
        intent=str(mapping.get("intent", "")),
        audience_inference=str(mapping.get("audience_inference", "")),
        section_relationships=str(mapping.get("section_relationships", "")),
    )


class RedesignAttemptStore:
    """Immutable store for redesign attempts under ``<root>/redesign``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.attempts_root = self.root / REDESIGN_ATTEMPT_DIRNAME

    def publish(
        self, attempt: RedesignAttempt, *, captures_digest: str = ""
    ) -> Path:
        """Persist one attempt; refuses to overwrite existing attempt ids."""

        payload = attempt_to_payload(attempt, captures_digest=captures_digest)
        payload_bytes = _canonical_bytes(payload)
        if len(payload_bytes) > MAX_REDESIGN_JSON_BYTES:
            raise RedesignArtifactError("redesign payload exceeds size budget")
        destination = self.attempts_root / attempt.attempt_id
        if secure_is_link_or_reparse(destination) or os.path.lexists(destination):
            raise RedesignArtifactError("attempt already exists; overwrite refused")
        destination.mkdir(parents=True, exist_ok=False)
        index = {
            "schema": REDESIGN_INDEX_SCHEMA,
            "attempt_id": attempt.attempt_id,
            "status": attempt.status.value,
            "created_at": attempt.created_at or "",
            "payload_sha256": _sha256(payload_bytes),
        }
        secure_write_bytes(destination / "payload.json", payload_bytes)
        secure_write_bytes(destination / "index.json", _canonical_bytes(index))
        return destination

    def _read_attempt(self, directory: Path) -> RedesignAttempt | None:
        """Read one attempt; ``None`` when anything is corrupt or invalid.

        Reads are size-bounded (``MAX_REDESIGN_JSON_BYTES`` for the payload,
        a small manifest cap for the index) so the offline report path can
        never be forced to buffer an unbounded untrusted file.
        """

        try:
            index = json.loads(
                secure_read_bytes(
                    directory / "index.json",
                    "redesign index manifest",
                    max_bytes=_REDESIGN_INDEX_MAX_BYTES,
                ).decode("utf-8")
            )
            payload_bytes = secure_read_bytes(
                directory / "payload.json",
                "redesign attempt payload",
                max_bytes=MAX_REDESIGN_JSON_BYTES,
            )
        except (OSError, RuntimeError, ValueError, UnicodeError):
            return None
        if not isinstance(index, dict):
            return None
        index_mapping = cast(Mapping[str, object], index)
        digest = index_mapping.get("payload_sha256")
        if not isinstance(digest, str) or digest != _sha256(payload_bytes):
            return None
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeError, ValueError):
            return None
        try:
            attempt = attempt_from_payload(payload)
        except (RedesignArtifactError, TypeError, ValueError):
            return None
        if attempt.attempt_id != directory.name:
            return None
        return attempt

    def newest_valid_attempt(self) -> RedesignAttemptSelection:
        """Select the newest valid attempt, noting skipped corrupt ones."""

        if not self.attempts_root.is_dir():
            return RedesignAttemptSelection(None)
        directories = [
            entry
            for entry in self.attempts_root.iterdir()
            if entry.is_dir() and _ATTEMPT_ID_PATTERN.fullmatch(entry.name)
        ]
        directories.sort(key=lambda entry: entry.name, reverse=True)
        skipped: list[str] = []
        for directory in directories:
            attempt = self._read_attempt(directory)
            if attempt is None:
                skipped.append(
                    f"attempt {directory.name} is corrupt or unreadable"
                )
                continue
            return RedesignAttemptSelection(attempt, tuple(skipped))
        return RedesignAttemptSelection(None, tuple(skipped))


__all__ = [
    "MAX_REDESIGN_JSON_BYTES",
    "REDESIGN_ATTEMPT_DIRNAME",
    "REDESIGN_INDEX_SCHEMA",
    "REDESIGN_PAYLOAD_SCHEMA",
    "RedesignArtifactError",
    "RedesignAttemptSelection",
    "RedesignAttemptStore",
    "attempt_from_payload",
    "attempt_to_payload",
    "new_attempt_id",
]
