"""Immutable experiment-scoped report-synthesis artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TypeVar, cast
from uuid import uuid4

from ux_analyzer.domain.synthesis import (
    CANONICAL_SYNTHESIS_ROLES,
    REPORT_ADJUDICATOR_ROLE,
    EvidenceRef,
    FindingKind,
    ObjectionSeverity,
    RejectedCandidateAudit,
    SynthesisAttempt,
    SynthesisFinding,
    SynthesisObjection,
    SynthesisRoleReceipt,
    SynthesisStatus,
    final_finding_preserves_candidate,
)
from ux_analyzer.ports.report_synthesis import SynthesisCorpusPort
from ux_analyzer.storage.run_bundle import (
    SecurePathIdentity,
    secure_assert_ancestors,
    secure_ensure_directory,
    secure_is_link_or_reparse,
    secure_make_temporary_directory,
    secure_open_file_descriptor,
    secure_path_identity,
    secure_read_bytes,
    secure_remove_tree,
    secure_replace,
    secure_unlink,
    secure_write_bytes,
)

_INDEX_SCHEMA_VERSION = "synthesis-index-v1"
_ARTIFACT_SCHEMA_VERSION = "synthesis-artifact-v3"
_PRIOR_ARTIFACT_SCHEMA_VERSION = "synthesis-artifact-v2"
_LEGACY_ARTIFACT_SCHEMA_VERSION = "synthesis-artifact-v1"
_SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = frozenset(
    {
        _LEGACY_ARTIFACT_SCHEMA_VERSION,
        _PRIOR_ARTIFACT_SCHEMA_VERSION,
        _ARTIFACT_SCHEMA_VERSION,
    }
)
MAX_SYNTHESIS_JSON_BYTES = 64 * 1024 * 1024
_OPTIMISTIC_READ_ATTEMPTS = 3
_DIGEST_LENGTH = 64
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_CREATED_PATTERN = re.compile(
    r"^(?:\d{8}T\d{6}(?:\d{6})?Z|\d{4}-\d{2}-\d{2}T\d{6}(?:\.\d{1,6})?Z)$"
)
_ACCEPTED_STATUSES = frozenset({SynthesisStatus.ACCEPTED, SynthesisStatus.NO_ISSUES})
_ATTEMPT_BUNDLE_FILES = ("synthesis.json", "corpus-manifest.json")
_PUBLICATION_LOCKS: dict[str, threading.RLock] = {}
_PUBLICATION_LOCKS_GUARD = threading.Lock()
_ReadResult = TypeVar("_ReadResult")


class SynthesisArtifactError(ValueError):
    """Raised when synthesis artifact state is invalid or cannot be trusted."""


def validate_publishable_synthesis_attempt(attempt: SynthesisAttempt) -> None:
    """Reject contradictory status and final-finding publication state."""

    _validate_publishable_synthesis_attempt(attempt, legacy=False)


def _validate_publishable_synthesis_attempt(
    attempt: SynthesisAttempt,
    *,
    legacy: bool,
) -> None:

    candidates = attempt.candidate_findings
    finals = attempt.findings
    rejected = attempt.rejected_findings
    candidate_ids = [finding.finding_id for finding in candidates]
    final_ids = [finding.finding_id for finding in finals]
    rejected_ids = [finding.finding_id for finding in rejected]
    objection_ids = [objection.objection_id for objection in attempt.objections]
    receipt_roles = [receipt.role for receipt in attempt.role_receipts]
    rejected_audit_ids = [
        audit.finding_id for audit in attempt.rejected_candidate_audits
    ]
    candidate_id_set = set(candidate_ids)

    if len(receipt_roles) != len(set(receipt_roles)):
        raise SynthesisArtifactError("synthesis role receipt roles must be unique")
    if not legacy and attempt.status in _ACCEPTED_STATUSES and set(receipt_roles) != set(
        CANONICAL_SYNTHESIS_ROLES
    ):
        raise SynthesisArtifactError(
            "accepted or no-issues synthesis requires one completion receipt for every canonical role"
        )
    if len(rejected_audit_ids) != len(set(rejected_audit_ids)):
        raise SynthesisArtifactError(
            "rejected candidate audit finding IDs must be unique"
        )

    if attempt.status is SynthesisStatus.NO_ISSUES:
        if candidates or rejected or attempt.objections or finals:
            raise SynthesisArtifactError(
                "no-issues synthesis must not contain review or finding state"
            )
        return

    if len(objection_ids) != len(set(objection_ids)):
        raise SynthesisArtifactError("synthesis objection IDs must be unique")
    if any(
        objection.finding_id not in candidate_id_set for objection in attempt.objections
    ):
        raise SynthesisArtifactError(
            "synthesis objection must reference an analyst candidate"
        )
    for objection in attempt.objections:
        if not legacy and attempt.status is SynthesisStatus.ACCEPTED and (
            objection.resolution is None
            or objection.resolved_by_role != REPORT_ADJUDICATOR_ROLE
        ):
            raise SynthesisArtifactError(
                "accepted synthesis requires an explicit report-adjudicator disposition for every objection"
            )
        if (
            objection.severity is not ObjectionSeverity.BLOCKING
            or not objection.resolved
        ):
            continue
        if objection.resolved_by_role != REPORT_ADJUDICATOR_ROLE:
            raise SynthesisArtifactError(
                "resolved blocking objection requires report-adjudicator provenance"
            )
        if not objection.resolution_evidence_refs:
            raise SynthesisArtifactError(
                "resolved blocking objection requires resolution evidence"
            )

    if attempt.status is SynthesisStatus.ACCEPTED:
        if not finals:
            raise SynthesisArtifactError(
                "accepted synthesis requires at least one final finding"
            )
        if any(finding.reviewer_state != "accepted" for finding in finals):
            raise SynthesisArtifactError(
                "accepted synthesis findings require accepted reviewer state"
            )
        published_ids = set(final_ids)
        if any(
            objection.finding_id in published_ids
            and objection.severity is ObjectionSeverity.BLOCKING
            and not objection.resolved
            for objection in attempt.objections
        ):
            raise SynthesisArtifactError(
                "accepted synthesis finding has an unresolved blocking objection"
            )
    elif finals:
        raise SynthesisArtifactError(
            "rejected or unavailable synthesis must not publish final findings"
        )

    for label, finding_ids in (
        ("candidate", candidate_ids),
        ("final", final_ids),
        ("rejected", rejected_ids),
    ):
        if len(finding_ids) != len(set(finding_ids)):
            raise SynthesisArtifactError(
                f"synthesis {label} finding IDs must be unique"
            )

    final_id_set = set(final_ids)
    rejected_id_set = set(rejected_ids)
    if not final_id_set <= candidate_id_set:
        raise SynthesisArtifactError(
            "final synthesis finding must derive from an analyst candidate"
        )
    candidates_by_id = {finding.finding_id: finding for finding in candidates}
    if any(
        not (
            _legacy_final_finding_preserves_candidate(
                final,
                candidates_by_id[final.finding_id],
            )
            if legacy
            else final_finding_preserves_candidate(
                final,
                candidates_by_id[final.finding_id],
                objections=attempt.objections,
            )
        )
        for final in finals
        if final.finding_id in candidates_by_id
    ):
        raise SynthesisArtifactError(
            "final synthesis finding must preserve reviewed candidate claim and evidence"
        )
    if not rejected_id_set <= candidate_id_set:
        raise SynthesisArtifactError(
            "rejected synthesis finding must derive from an analyst candidate"
        )
    if final_id_set & rejected_id_set:
        raise SynthesisArtifactError(
            "candidate finding cannot have both final and rejected dispositions"
        )
    if final_id_set | rejected_id_set != candidate_id_set:
        raise SynthesisArtifactError(
            "every candidate finding requires exactly one final or rejected disposition"
        )
    if any(finding.reviewer_state != "not-established" for finding in rejected):
        raise SynthesisArtifactError(
            "rejected synthesis findings require not-established reviewer state"
        )


def _legacy_final_finding_preserves_candidate(
    final: SynthesisFinding,
    candidate: SynthesisFinding,
) -> bool:
    if final.finding_id != candidate.finding_id:
        return False
    if any(
        getattr(final, field_name) != getattr(candidate, field_name)
        for field_name in ("issue", "impact", "root_cause")
    ):
        return False
    candidate_evidence_ids = {ref.evidence_id for ref in candidate.evidence_refs}
    final_evidence_ids = {ref.evidence_id for ref in final.evidence_refs}
    return candidate_evidence_ids <= final_evidence_ids


def _publication_thread_lock(synthesis_root: Path) -> threading.RLock:
    key = os.path.normcase(os.path.abspath(os.fspath(synthesis_root)))
    with _PUBLICATION_LOCKS_GUARD:
        lock = _PUBLICATION_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PUBLICATION_LOCKS[key] = lock
        return lock


@contextmanager
def _publication_lock(synthesis_root: Path) -> Generator[None, None, None]:
    """Serialize attempt publication and index replacement across writers."""

    with _publication_thread_lock(synthesis_root):
        lock_path = synthesis_root / ".publication.lock"
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        with secure_open_file_descriptor(
            lock_path, flags, "synthesis publication lock"
        ) as opened:
            descriptor, lock_identity = opened
            if secure_is_link_or_reparse(lock_path):
                raise SynthesisArtifactError(
                    "synthesis publication lock must not be a link"
                )
            if os.name == "nt":
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"0")
                    os.fsync(descriptor)
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                if (
                    secure_path_identity(lock_path, "synthesis publication lock")
                    != lock_identity
                ):
                    raise SynthesisArtifactError(
                        "synthesis publication lock identity changed"
                    )
                yield
            finally:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def _existing_publication_read_lock(lock_path: Path) -> Generator[None, None, None]:
    """Share an existing publication lock without mutating its file."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with secure_open_file_descriptor(
        lock_path, flags, "synthesis publication lock"
    ) as opened:
        descriptor, lock_identity = opened
        if secure_is_link_or_reparse(lock_path):
            raise SynthesisArtifactError(
                "synthesis publication lock must not be a link"
            )
        if os.name == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_RLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_SH)
        try:
            if (
                secure_path_identity(lock_path, "synthesis publication lock")
                != lock_identity
            ):
                raise SynthesisArtifactError(
                    "synthesis publication lock identity changed"
                )
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)


def _json_object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SynthesisArtifactError("JSON contains duplicate fields")
        result[key] = value
    return result


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _json_value(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return [_json_value(item) for item in sequence]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(cast(object, getattr(value, item.name)))
            for item in fields(value)
        }
    return value


def _canonical_bytes(value: object, *, trailing_newline: bool = True) -> bytes:
    serialized = json.dumps(
        _json_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if trailing_newline:
        serialized += "\n"
    return serialized.encode("ascii")


def _validate_json_size(content: bytes | str, label: str) -> None:
    if len(content) > MAX_SYNTHESIS_JSON_BYTES:
        raise SynthesisArtifactError(f"{label} exceeds size limit")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _digest_identifier(identifier: str) -> str:
    return _sha256(identifier.encode("utf-8"))


def _created_at_from_attempt_token(value: str) -> datetime:
    formats = ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H%M%SZ")
    for date_format in formats:
        try:
            return datetime.strptime(value, date_format).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise SynthesisArtifactError("attempt ID contains an invalid UTC timestamp")


def _created_at_utc(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SynthesisArtifactError(f"{field_name} must be a UTC timestamp") from error
    if parsed.tzinfo is None:
        raise SynthesisArtifactError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def _created_at_matches_attempt_id(created_at: str, attempt_id: str) -> bool:
    created_token, _, _ = _validate_attempt_id(attempt_id)
    try:
        parsed = _created_at_utc(created_at, "created_at")
    except SynthesisArtifactError:
        return False
    expected = _created_at_from_attempt_token(created_token)
    return parsed.replace(microsecond=0) == expected


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SynthesisArtifactError(f"{field_name} must be a non-empty string")
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SynthesisArtifactError(f"{field_name} must be an object")
    mapping = cast(Mapping[object, object], value)
    return {str(key): item for key, item in mapping.items()}


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise SynthesisArtifactError(f"{field_name} must be an array")
    return cast(list[object], value)


def _enum_text(value: object, field_name: str) -> str:
    if isinstance(value, Enum):
        return _text(value.value, field_name)
    return _text(value, field_name)


def _float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SynthesisArtifactError(f"{field_name} must be numeric")
    return float(value)


def _digest(value: object, field_name: str) -> str:
    result = _text(value, field_name)
    if not _DIGEST_PATTERN.fullmatch(result):
        raise SynthesisArtifactError(f"{field_name} must be a lowercase SHA-256 digest")
    return result


def _validate_attempt_id(attempt_id: object) -> tuple[str, str, int]:
    value = _text(attempt_id, "attempt ID")
    created, digest_prefix, sequence_text = (
        value.rsplit("-", maxsplit=2) if value.count("-") >= 2 else ("", "", "")
    )
    if (
        not _ATTEMPT_CREATED_PATTERN.fullmatch(created)
        or not re.fullmatch(r"[0-9a-f]{12}", digest_prefix)
        or not re.fullmatch(r"[1-9][0-9]*", sequence_text)
        or Path(value).name != value
        or "\\" in value
        or "\x00" in value
    ):
        raise SynthesisArtifactError(
            "attempt ID must be <created-at-utc>-<first12(corpus_digest)>-<sequence>"
        )
    return created, digest_prefix, int(sequence_text)


def _attempt_order_key(attempt_id: str) -> tuple[datetime, int, str]:
    created, digest_prefix, sequence = _validate_attempt_id(attempt_id)
    return _created_at_from_attempt_token(created), sequence, digest_prefix


def synthesis_attempt_position(attempt_id: str) -> tuple[datetime, int]:
    """Return normalized UTC creation second and global sequence."""

    created, _, sequence = _validate_attempt_id(attempt_id)
    return _created_at_from_attempt_token(created), sequence


def _index_record_order_key(
    record: Mapping[str, object],
) -> tuple[datetime, int, str]:
    attempt_id = _text(record.get("attempt_id"), "index attempt ID")
    _, sequence = synthesis_attempt_position(attempt_id)
    created_at = _created_at_utc(
        _text(record.get("created_at"), "index created_at"),
        "index created_at",
    )
    return created_at, sequence, attempt_id


def _evidence_ref_to_dict(reference: EvidenceRef) -> dict[str, object]:
    return {
        "evidence_id": reference.evidence_id,
        "kind": reference.kind,
        "run_id": reference.run_id,
        "viewport_id": reference.viewport_id,
        "element_id": reference.element_id,
        "event_id": reference.event_id,
        "metric_id": reference.metric_id,
        "artifact_path": reference.artifact_path,
        "replay_sequence": reference.replay_sequence,
        "sha256": reference.sha256,
    }


def _evidence_ref_from_dict(value: object) -> EvidenceRef:
    mapping = _mapping(value, "evidence reference")
    artifact_path = mapping.get("artifact_path")
    if artifact_path is not None and not isinstance(artifact_path, str):
        raise SynthesisArtifactError(
            "evidence reference artifact_path must be a string"
        )
    replay_sequence = mapping.get("replay_sequence")
    if replay_sequence is not None and type(replay_sequence) is not int:
        raise SynthesisArtifactError(
            "evidence reference replay_sequence must be an exact integer"
        )
    return EvidenceRef(
        evidence_id=_text(mapping.get("evidence_id"), "evidence ID"),
        kind=_text(mapping.get("kind"), "evidence kind"),
        run_id=_text(mapping.get("run_id"), "run ID"),
        viewport_id=cast(str | None, mapping.get("viewport_id")),
        element_id=cast(str | None, mapping.get("element_id")),
        event_id=cast(str | None, mapping.get("event_id")),
        metric_id=cast(str | None, mapping.get("metric_id")),
        artifact_path=artifact_path,
        replay_sequence=replay_sequence,
        sha256=cast(str | None, mapping.get("sha256")),
    )


def _finding_to_dict(finding: SynthesisFinding) -> dict[str, object]:
    return {
        "finding_id": finding.finding_id,
        "title": finding.title,
        "issue": finding.issue,
        "impact": finding.impact,
        "root_cause": finding.root_cause,
        "fixes": list(finding.fixes),
        "severity": _enum_text(finding.severity, "finding severity"),
        "confidence": finding.confidence,
        "evidence_refs": [
            _evidence_ref_to_dict(reference) for reference in finding.evidence_refs
        ],
        "affected_surfaces": list(finding.affected_surfaces),
        "principles": list(finding.principles),
        "counterevidence": [
            _evidence_ref_to_dict(item) if isinstance(item, EvidenceRef) else item
            for item in finding.counterevidence
        ],
        "limitations": list(finding.limitations),
        "reviewer_state": finding.reviewer_state,
        "evidence_class": _enum_text(finding.evidence_class, "evidence class"),
        "reproducibility": _enum_text(finding.reproducibility, "reproducibility"),
        "severity_justification": finding.severity_justification,
        "reviewer_notes": list(finding.reviewer_notes),
        "finding_kind": _enum_text(finding.finding_kind, "finding kind"),
    }


def _finding_from_dict(value: object) -> SynthesisFinding:
    mapping = _mapping(value, "finding")
    counterevidence: list[str | EvidenceRef] = []
    for item in _list(mapping.get("counterevidence", []), "counterevidence"):
        if isinstance(item, Mapping):
            counterevidence.append(
                _evidence_ref_from_dict(_mapping(cast(object, item), "counterevidence"))
            )
        else:
            counterevidence.append(_text(item, "counterevidence"))
    return SynthesisFinding(
        finding_id=_text(mapping.get("finding_id"), "finding ID"),
        title=_text(mapping.get("title"), "finding title"),
        issue=_text(mapping.get("issue"), "finding issue"),
        impact=_text(mapping.get("impact"), "finding impact"),
        root_cause=_text(mapping.get("root_cause"), "finding root cause"),
        fixes=tuple(
            _text(item, "finding fix") for item in _list(mapping.get("fixes"), "fixes")
        ),
        severity=_text(mapping.get("severity"), "finding severity"),
        confidence=cast(float, mapping.get("confidence")),
        evidence_refs=tuple(
            _evidence_ref_from_dict(item)
            for item in _list(mapping.get("evidence_refs"), "evidence_refs")
        ),
        affected_surfaces=tuple(
            _text(item, "affected surface")
            for item in _list(mapping.get("affected_surfaces", []), "affected_surfaces")
        ),
        principles=tuple(
            _text(item, "principle")
            for item in _list(mapping.get("principles", []), "principles")
        ),
        counterevidence=tuple(counterevidence),
        limitations=tuple(
            _text(item, "finding limitation")
            for item in _list(mapping.get("limitations", []), "limitations")
        ),
        reviewer_state=_text(mapping.get("reviewer_state"), "reviewer state"),
        evidence_class=_text(mapping.get("evidence_class"), "evidence class"),
        reproducibility=_text(mapping.get("reproducibility"), "reproducibility"),
        severity_justification=cast(str, mapping.get("severity_justification", "")),
        reviewer_notes=tuple(
            _text(item, "reviewer note")
            for item in _list(mapping.get("reviewer_notes", []), "reviewer_notes")
        ),
        # Artifacts written before scenario-defect classification existed treat
        # every finding as a product issue.
        finding_kind=cast(
            str,
            mapping.get("finding_kind", FindingKind.UX_ISSUE.value),
        ),
    )


def _objection_to_dict(objection: SynthesisObjection) -> dict[str, object]:
    return {
        "objection_id": objection.objection_id,
        "finding_id": objection.finding_id,
        "objection_type": objection.objection_type,
        "severity": _enum_text(objection.severity, "objection severity"),
        "message": objection.message,
        "evidence_refs": [
            _evidence_ref_to_dict(reference) for reference in objection.evidence_refs
        ],
        "reviewer_role": objection.reviewer_role,
        "resolved": objection.resolved,
        "resolution": objection.resolution,
        "resolved_by_role": objection.resolved_by_role,
        "resolution_evidence_refs": [
            _evidence_ref_to_dict(reference)
            for reference in objection.resolution_evidence_refs
        ],
    }


def _objection_from_dict(
    value: object,
    *,
    legacy: bool = False,
) -> SynthesisObjection:
    mapping = _mapping(value, "objection")
    resolved = mapping.get("resolved", False)
    if not isinstance(resolved, bool):
        raise SynthesisArtifactError("objection resolved must be boolean")
    if "resolved_by_role" not in mapping:
        raise SynthesisArtifactError("objection resolved_by_role is required")
    resolved_by_role_value = mapping.get("resolved_by_role")
    resolved_by_role = (
        None
        if resolved_by_role_value is None
        else _text(resolved_by_role_value, "objection resolved_by_role")
    )
    if "resolution_evidence_refs" not in mapping:
        raise SynthesisArtifactError("objection resolution_evidence_refs is required")
    return SynthesisObjection(
        objection_id=_text(mapping.get("objection_id"), "objection ID"),
        finding_id=_text(mapping.get("finding_id"), "objection finding ID"),
        objection_type=(
            "other"
            if legacy and "objection_type" not in mapping
            else _text(mapping.get("objection_type"), "objection type")
        ),
        severity=_text(mapping.get("severity"), "objection severity"),
        message=_text(mapping.get("message"), "objection message"),
        evidence_refs=tuple(
            _evidence_ref_from_dict(item)
            for item in _list(
                mapping.get("evidence_refs", []), "objection evidence_refs"
            )
        ),
        reviewer_role=cast(str, mapping.get("reviewer_role", "")),
        resolved=resolved,
        resolution=cast(str | None, mapping.get("resolution")),
        resolved_by_role=resolved_by_role,
        resolution_evidence_refs=tuple(
            _evidence_ref_from_dict(item)
            for item in _list(
                mapping.get("resolution_evidence_refs"),
                "objection resolution_evidence_refs",
            )
        ),
    )


def _role_receipt_to_dict(receipt: SynthesisRoleReceipt) -> dict[str, object]:
    return {
        "role": receipt.role,
        "provider_id": receipt.provider_id,
        "model_id": receipt.model_id,
        "prompt_digest": receipt.prompt_digest,
        "schema_digest": receipt.schema_digest,
        "output_digest": receipt.output_digest,
    }


def _role_receipt_from_dict(value: object) -> SynthesisRoleReceipt:
    mapping = _mapping(value, "role receipt")
    return SynthesisRoleReceipt(
        role=_text(mapping.get("role"), "receipt role"),
        provider_id=_text(mapping.get("provider_id"), "receipt provider_id"),
        model_id=_text(mapping.get("model_id"), "receipt model_id"),
        prompt_digest=_digest(mapping.get("prompt_digest"), "receipt prompt_digest"),
        schema_digest=_digest(mapping.get("schema_digest"), "receipt schema_digest"),
        output_digest=_digest(mapping.get("output_digest"), "receipt output_digest"),
    )


def _rejected_candidate_audit_to_dict(
    audit: RejectedCandidateAudit,
) -> dict[str, object]:
    return {
        "finding_id": audit.finding_id,
        "source_role": audit.source_role,
        "reason_code": audit.reason_code,
        "output_digest": audit.output_digest,
    }


def _rejected_candidate_audit_from_dict(value: object) -> RejectedCandidateAudit:
    mapping = _mapping(value, "rejected candidate audit")
    return RejectedCandidateAudit(
        finding_id=_text(mapping.get("finding_id"), "audit finding_id"),
        source_role=_text(mapping.get("source_role"), "audit source_role"),
        reason_code=_text(mapping.get("reason_code"), "audit reason_code"),
        output_digest=_digest(mapping.get("output_digest"), "audit output_digest"),
    )


def _attempt_to_dict(attempt: SynthesisAttempt) -> dict[str, object]:
    prompt_digest = _digest_identifier(attempt.prompt_version)
    schema_digest = _digest_identifier(attempt.schema_version)
    return {
        "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION,
        "attempt_id": attempt.attempt_id,
        "created_at": attempt.created_at,
        "corpus_digest": attempt.corpus_digest,
        "expectation_digest": attempt.expectation_digest,
        "principle_pack_digest": attempt.principle_pack_digest,
        "prompt_version": attempt.prompt_version,
        "schema_version": attempt.schema_version,
        "prompt_digest": prompt_digest,
        "schema_digest": schema_digest,
        "digests": {
            "corpus": attempt.corpus_digest,
            "expectation": attempt.expectation_digest,
            "principle_pack": attempt.principle_pack_digest,
            "prompt": prompt_digest,
            "schema": schema_digest,
        },
        "model_manifest": attempt.model_manifest,
        "role_manifest": attempt.role_manifest,
        "retrieval_log": attempt.retrieval_log,
        "usage": attempt.usage,
        "role_receipts": [
            _role_receipt_to_dict(item) for item in attempt.role_receipts
        ],
        "rejected_candidate_audits": [
            _rejected_candidate_audit_to_dict(item)
            for item in attempt.rejected_candidate_audits
        ],
        "candidates": [_finding_to_dict(item) for item in attempt.candidates],
        "objections": [_objection_to_dict(item) for item in attempt.objections],
        "rejected_findings": [
            _finding_to_dict(item) for item in attempt.rejected_findings
        ],
        "final_findings": [_finding_to_dict(item) for item in attempt.final_findings],
        "status": _enum_text(attempt.status, "status"),
        "limitations": list(attempt.limitations),
        "fallback_available": attempt.fallback_available,
    }


def _attempt_from_dict(value: Mapping[str, object]) -> SynthesisAttempt:
    artifact_schema_version = value.get("artifact_schema_version")
    if artifact_schema_version not in _SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise SynthesisArtifactError("unsupported artifact schema version")
    legacy = artifact_schema_version == _LEGACY_ARTIFACT_SCHEMA_VERSION
    digests = _mapping(value.get("digests"), "digests")
    attempt_id = _text(value.get("attempt_id"), "attempt ID")
    prompt_version = _text(value.get("prompt_version"), "prompt_version")
    schema_version = _text(value.get("schema_version"), "schema_version")
    if _digest(digests.get("prompt"), "prompt digest") != _digest_identifier(
        prompt_version
    ) or _digest(value.get("prompt_digest"), "prompt_digest") != _digest_identifier(
        prompt_version
    ):
        raise SynthesisArtifactError("prompt digest mismatch")
    if _digest(digests.get("schema"), "schema digest") != _digest_identifier(
        schema_version
    ) or _digest(value.get("schema_digest"), "schema_digest") != _digest_identifier(
        schema_version
    ):
        raise SynthesisArtifactError("schema digest mismatch")

    fallback_available = value.get("fallback_available", True)
    if not isinstance(fallback_available, bool):
        raise SynthesisArtifactError("fallback_available must be boolean")
    created_at_value = value.get("created_at")
    if created_at_value is None:
        raise SynthesisArtifactError("created_at is required for persisted attempts")
    created_at = _text(created_at_value, "created_at")
    _validate_attempt_id(attempt_id)
    if not _created_at_matches_attempt_id(created_at, attempt_id):
        raise SynthesisArtifactError("attempt timestamp does not match attempt ID")
    return SynthesisAttempt(
        attempt_id=attempt_id,
        status=_text(value.get("status"), "status"),
        corpus_digest=_digest(value.get("corpus_digest"), "corpus_digest"),
        expectation_digest=_digest(
            value.get("expectation_digest"), "expectation_digest"
        ),
        principle_pack_digest=_digest(
            value.get("principle_pack_digest"), "principle_pack_digest"
        ),
        model_manifest=_mapping(value.get("model_manifest"), "model_manifest"),
        role_manifest=_mapping(value.get("role_manifest"), "role_manifest"),
        prompt_version=prompt_version,
        schema_version=schema_version,
        retrieval_log=tuple(
            _mapping(item, "retrieval log entry")
            for item in _list(value.get("retrieval_log"), "retrieval_log")
        ),
        usage={
            key: _float(item, f"usage.{key}")
            for key, item in _mapping(value.get("usage"), "usage").items()
        },
        role_receipts=tuple(
            _role_receipt_from_dict(item)
            for item in _list(
                value.get("role_receipts", [] if legacy else None),
                "role_receipts",
            )
        ),
        rejected_candidate_audits=tuple(
            _rejected_candidate_audit_from_dict(item)
            for item in _list(
                value.get(
                    "rejected_candidate_audits",
                    [] if legacy else None,
                ),
                "rejected_candidate_audits",
            )
        ),
        candidate_findings=tuple(
            _finding_from_dict(item)
            for item in _list(value.get("candidates"), "candidates")
        ),
        objections=tuple(
            _objection_from_dict(item, legacy=legacy)
            for item in _list(value.get("objections"), "objections")
        ),
        rejected_findings=tuple(
            _finding_from_dict(item)
            for item in _list(value.get("rejected_findings"), "rejected_findings")
        ),
        findings=tuple(
            _finding_from_dict(item)
            for item in _list(value.get("final_findings"), "final_findings")
        ),
        limitations=tuple(
            _text(item, "limitation")
            for item in _list(value.get("limitations"), "limitations")
        ),
        fallback_available=fallback_available,
        created_at=created_at,
    )


def _expectation_digest_from_manifest(value: Mapping[str, object]) -> str:
    entries = _list(value.get("entries"), "corpus entries")
    payloads: list[object] = []
    for item in entries:
        entry = _mapping(item, "corpus entry")
        if entry.get("kind") == "expectation":
            payloads.append(entry.get("payload"))
    return _sha256(_canonical_bytes(payloads, trailing_newline=False))


def _validate_objection_evidence_refs(
    attempt: SynthesisAttempt,
    corpus_manifest: Mapping[str, object],
) -> None:
    corpus_refs: dict[str, EvidenceRef] = {}
    for item in _list(corpus_manifest.get("entries"), "corpus entries"):
        reference = _evidence_ref_from_dict(_mapping(item, "corpus entry"))
        if reference.evidence_id in corpus_refs:
            raise SynthesisArtifactError(
                "synthesis corpus contains duplicate evidence ID"
            )
        corpus_refs[reference.evidence_id] = reference

    for objection in attempt.objections:
        for reference in (
            *objection.evidence_refs,
            *objection.resolution_evidence_refs,
        ):
            corpus_reference = corpus_refs.get(reference.evidence_id)
            if corpus_reference is None:
                raise SynthesisArtifactError(
                    "objection evidence reference is absent from synthesis corpus"
                )
            if (
                type(reference.replay_sequence)
                is not type(corpus_reference.replay_sequence)
                or reference != corpus_reference
            ):
                raise SynthesisArtifactError(
                    "objection evidence reference does not match synthesis corpus"
                )


class SynthesisArtifactStore:
    """Persist and select immutable experiment-level synthesis attempts."""

    def __init__(self, experiment_output: Path) -> None:
        self.output = Path(experiment_output)
        self.synthesis_root = self.output / "synthesis"
        self.attempts_root = self.synthesis_root / "attempts"
        self.index_path = self.synthesis_root / "index.json"
        self._validate_existing_roots()

    @property
    def attempts(self) -> tuple[SynthesisAttempt, ...]:
        """Return every complete published attempt, including rejected attempts."""

        self._validate_existing_roots()
        if not os.path.lexists(self.synthesis_root):
            return ()
        return self._read_transaction(self._read_attempts)

    @property
    def accepted_attempt(self) -> SynthesisAttempt | None:
        """Return the attempt selected by the durable accepted pointer."""

        self._validate_existing_roots()
        if not os.path.lexists(self.synthesis_root):
            return None
        return self._read_transaction(self._read_accepted_attempt)

    @property
    def report_attempt(self) -> SynthesisAttempt | None:
        """Return only the trusted accepted or latest attempt needed by a report."""

        self._validate_existing_roots()
        if not os.path.lexists(self.synthesis_root):
            return None
        return self._read_transaction(self._read_report_attempt)

    def write_attempt(
        self, attempt: SynthesisAttempt, corpus: SynthesisCorpusPort
    ) -> Path:
        """Publish one complete attempt and update the accepted pointer when eligible."""

        validate_publishable_synthesis_attempt(attempt)
        self._ensure_layout()
        lock_acquired = False
        try:
            with _publication_lock(self.synthesis_root):
                lock_acquired = True
                return self._write_attempt_locked(attempt, corpus)
        except SynthesisArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            if lock_acquired:
                raise
            raise SynthesisArtifactError(str(error)) from error

    def _write_attempt_locked(
        self, attempt: SynthesisAttempt, corpus: SynthesisCorpusPort
    ) -> Path:
        attempt_ids = self._attempt_ids()
        self._read_index(attempt_ids)
        _, digest_prefix, sequence = _validate_attempt_id(attempt.attempt_id)
        creation_second, _ = synthesis_attempt_position(attempt.attempt_id)
        corpus_json = corpus.to_json()
        _validate_json_size(corpus_json, "synthesis corpus manifest")
        try:
            corpus_bytes = corpus_json.encode("ascii")
        except UnicodeEncodeError as error:
            raise SynthesisArtifactError(
                "synthesis corpus manifest must be ASCII JSON"
            ) from error
        corpus_value = _mapping(json.loads(corpus_json), "synthesis corpus manifest")
        corpus_digest = corpus.digest
        if not _DIGEST_PATTERN.fullmatch(attempt.corpus_digest):
            raise SynthesisArtifactError(
                "corpus_digest must be a lowercase SHA-256 digest"
            )
        _digest(attempt.expectation_digest, "expectation_digest")
        _digest(attempt.principle_pack_digest, "principle_pack_digest")
        _digest(corpus.principle_pack_digest, "corpus principle_pack_digest")
        if attempt.corpus_digest != corpus_digest:
            raise SynthesisArtifactError("corpus digest mismatch")
        if digest_prefix != corpus_digest[:12]:
            raise SynthesisArtifactError("attempt ID corpus digest prefix mismatch")
        if attempt.expectation_digest != _expectation_digest_from_manifest(
            corpus_value
        ):
            raise SynthesisArtifactError("expectation digest mismatch")
        if attempt.principle_pack_digest != corpus.principle_pack_digest:
            raise SynthesisArtifactError("principle-pack digest mismatch")
        _validate_objection_evidence_refs(attempt, corpus_value)
        if attempt.created_at is None:
            raise SynthesisArtifactError(
                "created_at is required for persisted attempts"
            )

        destination = self.attempts_root / attempt.attempt_id
        if secure_is_link_or_reparse(destination):
            raise SynthesisArtifactError(
                "attempt destination must not be a symlink or reparse point"
            )
        if os.path.lexists(destination):
            raise SynthesisArtifactError("attempt already exists; overwrite refused")
        if any(
            existing_creation_second == creation_second
            and existing_sequence == sequence
            for existing_id in attempt_ids
            for existing_creation_second, existing_sequence in (
                synthesis_attempt_position(existing_id),
            )
        ):
            raise SynthesisArtifactError(
                "attempt sequence already exists for creation token"
            )

        synthesis_bytes = _canonical_bytes(_attempt_to_dict(attempt))
        _validate_json_size(synthesis_bytes, "synthesis artifact")
        staging = self._temporary_attempt_directory(attempt.attempt_id)
        staging_identity = secure_path_identity(staging, "synthesis attempt staging")
        try:
            secure_write_bytes(staging / "synthesis.json", synthesis_bytes)
            secure_write_bytes(staging / "corpus-manifest.json", corpus_bytes)
            persisted, synthesis_bytes, corpus_bytes = (
                self._read_attempt_bundle_from_directory(
                    staging,
                    attempt.attempt_id,
                )
            )
            validated_identity = self._attempt_bundle_identity(staging)
            self._validate_index_record(
                self._index_record(persisted, synthesis_bytes, corpus_bytes),
                persisted,
                synthesis_bytes,
                corpus_bytes,
            )
            published_identity = self._publish_attempt(
                staging, destination, validated_identity[0]
            )
            self._verify_published_attempt(
                destination,
                published_identity,
                validated_identity,
                synthesis_bytes,
                corpus_bytes,
            )
            selected = (
                persisted.attempt_id if persisted.status in _ACCEPTED_STATUSES else None
            )
            self._write_index(selected)
        except BaseException:
            self._remove_temporary_directory(staging, staging_identity)
            raise
        return destination

    def select_accepted(self, attempt_id: str | None = None) -> SynthesisAttempt | None:
        """Select a persisted accepted/no-issues attempt, or read the current selection."""

        if attempt_id is None:
            return self.accepted_attempt
        self._ensure_layout()
        with _publication_lock(self.synthesis_root):
            _validate_attempt_id(attempt_id)
            attempt_ids = self._attempt_ids()
            index = self._read_index(attempt_ids)
            if attempt_id not in attempt_ids:
                raise SynthesisArtifactError("cannot select missing synthesis attempt")
            attempt, synthesis_bytes, corpus_bytes = self._read_attempt_bundle(
                attempt_id
            )
            record = self._index_records(index).get(attempt_id)
            if record is not None:
                self._validate_index_record(
                    record, attempt, synthesis_bytes, corpus_bytes
                )
            if attempt.status not in _ACCEPTED_STATUSES:
                raise SynthesisArtifactError(
                    "only accepted or no-issues attempts can be selected"
                )
            self._write_index(attempt_id)
            return attempt

    def _validate_existing_roots(self) -> None:
        try:
            secure_assert_ancestors(self.output, "experiment output root")
            if secure_is_link_or_reparse(self.output):
                raise SynthesisArtifactError(
                    "experiment output root must not be a link"
                )
            if os.path.lexists(self.synthesis_root):
                secure_assert_ancestors(self.synthesis_root, "synthesis root")
                if (
                    secure_is_link_or_reparse(self.synthesis_root)
                    or not self.synthesis_root.is_dir()
                ):
                    raise SynthesisArtifactError(
                        "synthesis root must be a real directory"
                    )
            if os.path.lexists(self.attempts_root):
                secure_assert_ancestors(self.attempts_root, "synthesis attempts root")
                if (
                    secure_is_link_or_reparse(self.attempts_root)
                    or not self.attempts_root.is_dir()
                ):
                    raise SynthesisArtifactError(
                        "synthesis attempts root must be a real directory"
                    )
            if os.path.lexists(self.index_path) and secure_is_link_or_reparse(
                self.index_path
            ):
                raise SynthesisArtifactError("synthesis index must not be a link")
        except SynthesisArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise SynthesisArtifactError(str(error)) from error

    def _ensure_layout(self) -> None:
        self._validate_existing_roots()
        try:
            secure_ensure_directory(self.output, "experiment output root")
            secure_ensure_directory(self.synthesis_root, "synthesis root")
            secure_ensure_directory(self.attempts_root, "synthesis attempts root")
        except SynthesisArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise SynthesisArtifactError(str(error)) from error

    def _attempt_ids(self) -> tuple[str, ...]:
        if not os.path.lexists(self.attempts_root):
            return ()
        result: list[str] = []
        try:
            children = tuple(self.attempts_root.iterdir())
        except OSError as error:
            raise SynthesisArtifactError("cannot list synthesis attempts") from error
        for child in children:
            if secure_is_link_or_reparse(child):
                raise SynthesisArtifactError(
                    "synthesis attempts must not contain symlinks or reparse points"
                )
            if child.name.startswith("."):
                continue
            if not child.is_dir():
                raise SynthesisArtifactError(
                    "synthesis attempts contain a non-directory"
                )
            _validate_attempt_id(child.name)
            result.append(child.name)
        return tuple(sorted(result, key=_attempt_order_key))

    def _read_transaction(self, reader: Callable[[], _ReadResult]) -> _ReadResult:
        lock_path = self.synthesis_root / ".publication.lock"
        with _publication_thread_lock(self.synthesis_root):
            if os.path.lexists(lock_path):
                try:
                    with _existing_publication_read_lock(lock_path):
                        return reader()
                except FileNotFoundError:
                    pass

            last_error: SynthesisArtifactError | None = None
            for _ in range(_OPTIMISTIC_READ_ATTEMPTS):
                try:
                    before = self._optimistic_read_snapshot(lock_path)
                except SynthesisArtifactError as error:
                    last_error = error
                    continue
                if before[0]:
                    try:
                        with _existing_publication_read_lock(lock_path):
                            return reader()
                    except FileNotFoundError:
                        continue
                try:
                    result = reader()
                    read_error: SynthesisArtifactError | None = None
                except SynthesisArtifactError as error:
                    result = None
                    read_error = error
                try:
                    after = self._optimistic_read_snapshot(lock_path)
                except SynthesisArtifactError as error:
                    last_error = error
                    continue
                if before == after:
                    if read_error is not None:
                        raise read_error
                    return cast(_ReadResult, result)
                last_error = read_error
                if after[0]:
                    try:
                        with _existing_publication_read_lock(lock_path):
                            return reader()
                    except FileNotFoundError:
                        continue
            if last_error is not None:
                raise last_error
            raise SynthesisArtifactError(
                "synthesis artifacts changed during bounded read"
            )

    def _optimistic_read_snapshot(
        self, lock_path: Path
    ) -> tuple[bool, tuple[str, ...], str | None]:
        lock_exists = os.path.lexists(lock_path)
        attempt_ids = self._attempt_ids()
        if not os.path.lexists(self.index_path):
            return lock_exists, attempt_ids, None
        try:
            index_bytes = secure_read_bytes(
                self.index_path,
                "synthesis index snapshot",
                max_bytes=MAX_SYNTHESIS_JSON_BYTES,
            )
        except SynthesisArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise SynthesisArtifactError(
                "cannot read synthesis index snapshot"
            ) from error
        return lock_exists, attempt_ids, _sha256(index_bytes)

    def _read_attempts(self) -> tuple[SynthesisAttempt, ...]:
        self._validate_existing_roots()
        attempt_ids = self._attempt_ids()
        index = self._read_index(attempt_ids)
        records = self._index_records(index)
        attempts: list[SynthesisAttempt] = []
        for attempt_id in attempt_ids:
            attempt, synthesis_bytes, corpus_bytes = self._read_attempt_bundle(
                attempt_id
            )
            record = records.get(attempt_id)
            if record is not None:
                self._validate_index_record(
                    record, attempt, synthesis_bytes, corpus_bytes
                )
            attempts.append(attempt)
        return tuple(attempts)

    def _read_accepted_attempt(self) -> SynthesisAttempt | None:
        self._validate_existing_roots()
        attempt_ids = self._attempt_ids()
        index = self._read_index(attempt_ids)
        if index is None or index.get("accepted_attempt_id") is None:
            return None
        attempt_id = _text(index.get("accepted_attempt_id"), "accepted_attempt_id")
        attempt, synthesis_bytes, corpus_bytes = self._read_attempt_bundle(attempt_id)
        self._validate_index_record(
            self._index_records(index)[attempt_id],
            attempt,
            synthesis_bytes,
            corpus_bytes,
        )
        if attempt.status not in _ACCEPTED_STATUSES:
            raise SynthesisArtifactError(
                "accepted index points at a non-accepted attempt"
            )
        return attempt

    def _read_report_attempt(self) -> SynthesisAttempt | None:
        self._validate_existing_roots()
        attempt_ids = self._attempt_ids()
        index = self._read_index(attempt_ids)
        if index is None:
            return None
        records = self._index_records(index)
        accepted_id = cast(str | None, index.get("accepted_attempt_id"))
        latest_record = max(records.values(), key=_index_record_order_key, default=None)
        attempt_id = accepted_id or (
            _text(latest_record.get("attempt_id"), "index attempt ID")
            if latest_record is not None
            else None
        )
        if attempt_id is None:
            return None
        attempt, synthesis_bytes, corpus_bytes = self._read_attempt_bundle(attempt_id)
        self._validate_index_record(
            records[attempt_id], attempt, synthesis_bytes, corpus_bytes
        )
        if accepted_id is None and attempt.status in _ACCEPTED_STATUSES:
            return None
        if accepted_id is not None and attempt.status not in _ACCEPTED_STATUSES:
            raise SynthesisArtifactError(
                "accepted index points at a non-accepted attempt"
            )
        return attempt

    def _read_index(
        self, attempt_ids: tuple[str, ...] | None = None
    ) -> Mapping[str, object] | None:
        if not os.path.lexists(self.index_path):
            return None
        valid_attempt_ids = frozenset(
            self._attempt_ids() if attempt_ids is None else attempt_ids
        )
        value, _ = self._read_json_object(self.index_path, "synthesis index")
        if value.get("schema_version") != _INDEX_SCHEMA_VERSION:
            raise SynthesisArtifactError("unsupported synthesis index schema")
        records = _list(value.get("attempts"), "synthesis index attempts")
        seen: set[str] = set()
        for item in records:
            record = _mapping(item, "synthesis index record")
            attempt_id = _text(record.get("attempt_id"), "index attempt ID")
            _validate_attempt_id(attempt_id)
            if attempt_id in seen:
                raise SynthesisArtifactError(
                    "synthesis index contains duplicate attempt"
                )
            seen.add(attempt_id)
            status = _text(record.get("status"), "index status")
            try:
                SynthesisStatus(status)
            except ValueError as error:
                raise SynthesisArtifactError(
                    "synthesis index contains unknown status"
                ) from error
            _digest(record.get("synthesis_digest"), "index synthesis_digest")
            _digest(
                record.get("corpus_manifest_digest"), "index corpus_manifest_digest"
            )
            _digest(record.get("corpus_digest"), "index corpus_digest")
            created_at = _text(record.get("created_at"), "index created_at")
            if not _created_at_matches_attempt_id(created_at, attempt_id):
                raise SynthesisArtifactError(
                    "synthesis index created_at does not match attempt ID"
                )
            if attempt_id not in valid_attempt_ids:
                raise SynthesisArtifactError(
                    "synthesis index references missing attempt"
                )

        accepted = value.get("accepted_attempt_id")
        if accepted is not None:
            accepted_id = _text(accepted, "accepted_attempt_id")
            _validate_attempt_id(accepted_id)
            if accepted_id not in seen:
                raise SynthesisArtifactError(
                    "accepted index points at an unlisted attempt"
                )
            accepted_record = self._index_records(value)[accepted_id]
            if (
                SynthesisStatus(_text(accepted_record.get("status"), "index status"))
                not in _ACCEPTED_STATUSES
            ):
                raise SynthesisArtifactError(
                    "accepted index points at a non-accepted attempt"
                )
        return value

    def _index_records(
        self, index: Mapping[str, object] | None
    ) -> dict[str, Mapping[str, object]]:
        if index is None:
            return {}
        return {
            _text(record.get("attempt_id"), "index attempt ID"): record
            for item in _list(index.get("attempts"), "synthesis index attempts")
            for record in (_mapping(item, "synthesis index record"),)
        }

    def _validate_index_record(
        self,
        record: Mapping[str, object],
        attempt: SynthesisAttempt,
        synthesis_bytes: bytes,
        corpus_bytes: bytes,
    ) -> None:
        if record.get("attempt_id") != attempt.attempt_id:
            raise SynthesisArtifactError("synthesis index attempt ID mismatch")
        if record.get("status") != _enum_text(attempt.status, "status"):
            raise SynthesisArtifactError("synthesis index status mismatch")
        if record.get("created_at") != attempt.created_at:
            raise SynthesisArtifactError("synthesis index created_at mismatch")
        if record.get("corpus_digest") != attempt.corpus_digest:
            raise SynthesisArtifactError("synthesis index corpus digest mismatch")
        if record.get("synthesis_digest") != _sha256(synthesis_bytes):
            raise SynthesisArtifactError("synthesis index synthesis digest mismatch")
        if record.get("corpus_manifest_digest") != _sha256(corpus_bytes):
            raise SynthesisArtifactError(
                "synthesis index corpus manifest digest mismatch"
            )

    def _read_attempt_bundle(
        self, attempt_id: str
    ) -> tuple[SynthesisAttempt, bytes, bytes]:
        return self._read_attempt_bundle_from_directory(
            self.attempts_root / attempt_id,
            attempt_id,
        )

    def _read_attempt_bundle_from_directory(
        self,
        directory: Path,
        attempt_id: str,
    ) -> tuple[SynthesisAttempt, bytes, bytes]:
        _, digest_prefix, _ = _validate_attempt_id(attempt_id)
        if secure_is_link_or_reparse(directory) or not directory.is_dir():
            raise SynthesisArtifactError("synthesis attempt is not a real directory")
        synthesis_value, synthesis_bytes = self._read_json_object(
            directory / "synthesis.json", "synthesis artifact"
        )
        corpus_value, corpus_bytes = self._read_json_object(
            directory / "corpus-manifest.json", "synthesis corpus manifest"
        )
        try:
            attempt = _attempt_from_dict(synthesis_value)
        except SynthesisArtifactError:
            raise
        except (TypeError, ValueError, RuntimeError) as error:
            raise SynthesisArtifactError("invalid synthesis artifact record") from error
        if attempt.attempt_id != attempt_id:
            raise SynthesisArtifactError("synthesis artifact attempt ID mismatch")
        if attempt.corpus_digest[:12] != digest_prefix:
            raise SynthesisArtifactError(
                "synthesis artifact attempt ID digest prefix mismatch"
            )
        if attempt.corpus_digest != _sha256(corpus_bytes):
            raise SynthesisArtifactError("synthesis corpus digest mismatch")
        if attempt.expectation_digest != _expectation_digest_from_manifest(
            corpus_value
        ):
            raise SynthesisArtifactError("synthesis expectation digest mismatch")
        if corpus_value.get("principle_pack_digest") != attempt.principle_pack_digest:
            raise SynthesisArtifactError("synthesis principle-pack digest mismatch")
        digests = _mapping(synthesis_value.get("digests"), "digests")
        if _digest(digests.get("corpus"), "corpus digest") != attempt.corpus_digest:
            raise SynthesisArtifactError("synthesis corpus digest field mismatch")
        if (
            _digest(digests.get("expectation"), "expectation digest")
            != attempt.expectation_digest
        ):
            raise SynthesisArtifactError("synthesis expectation digest field mismatch")
        if (
            _digest(digests.get("principle_pack"), "principle-pack digest")
            != attempt.principle_pack_digest
        ):
            raise SynthesisArtifactError(
                "synthesis principle-pack digest field mismatch"
            )
        _validate_publishable_synthesis_attempt(
            attempt,
            legacy=(
                synthesis_value.get("artifact_schema_version")
                == _LEGACY_ARTIFACT_SCHEMA_VERSION
            ),
        )
        _validate_objection_evidence_refs(attempt, corpus_value)
        return attempt, synthesis_bytes, corpus_bytes

    def _attempt_bundle_identity(
        self, directory: Path
    ) -> tuple[SecurePathIdentity, SecurePathIdentity, SecurePathIdentity]:
        """Snapshot allowlisted directory/file identities around publication."""

        if secure_is_link_or_reparse(directory) or not directory.is_dir():
            raise SynthesisArtifactError("synthesis attempt is not a real directory")
        children = tuple(directory.iterdir())
        if {child.name for child in children} != set(_ATTEMPT_BUNDLE_FILES) or len(
            children
        ) != len(_ATTEMPT_BUNDLE_FILES):
            raise SynthesisArtifactError(
                "synthesis attempt contains unexpected bundle entries"
            )
        paths = tuple(directory / name for name in _ATTEMPT_BUNDLE_FILES)
        if any(secure_is_link_or_reparse(path) or not path.is_file() for path in paths):
            raise SynthesisArtifactError(
                "synthesis attempt files must be real regular files"
            )
        return (
            secure_path_identity(directory, "synthesis attempt directory"),
            secure_path_identity(paths[0], "synthesis artifact"),
            secure_path_identity(paths[1], "synthesis corpus manifest"),
        )

    def _verify_published_attempt(
        self,
        destination: Path,
        published_root_identity: SecurePathIdentity,
        validated_identity: tuple[
            SecurePathIdentity,
            SecurePathIdentity,
            SecurePathIdentity,
        ],
        synthesis_bytes: bytes,
        corpus_bytes: bytes,
    ) -> None:
        """Bind published identities and bytes before any index mutation."""

        try:
            published_identity = self._attempt_bundle_identity(destination)
        except SynthesisArtifactError:
            self._quarantine_invalid_published_attempt(
                destination, published_root_identity
            )
            raise
        except (OSError, RuntimeError, ValueError) as error:
            self._quarantine_invalid_published_attempt(
                destination, published_root_identity
            )
            raise SynthesisArtifactError(
                "cannot verify published synthesis attempt identity"
            ) from error
        if validated_identity != published_identity:
            self._quarantine_invalid_published_attempt(
                destination, published_root_identity
            )
            raise SynthesisArtifactError(
                "published synthesis attempt differs from validated staging identity"
            )

        try:
            published_bytes = tuple(
                secure_read_bytes(
                    destination / name,
                    f"published synthesis attempt {name}",
                    max_bytes=MAX_SYNTHESIS_JSON_BYTES,
                )
                for name in _ATTEMPT_BUNDLE_FILES
            )
            identity_after_read = self._attempt_bundle_identity(destination)
        except (OSError, RuntimeError, ValueError) as error:
            self._quarantine_invalid_published_attempt(
                destination, published_root_identity
            )
            raise SynthesisArtifactError(
                "cannot verify published synthesis attempt bytes"
            ) from error

        expected_bytes = (synthesis_bytes, corpus_bytes)
        identities_match = published_identity == identity_after_read
        digests_match = all(
            hmac.compare_digest(
                hashlib.sha256(actual).digest(), hashlib.sha256(expected).digest()
            )
            and actual == expected
            for actual, expected in zip(published_bytes, expected_bytes, strict=True)
        )
        if identities_match and digests_match:
            return
        self._quarantine_invalid_published_attempt(destination, published_root_identity)
        raise SynthesisArtifactError(
            "published synthesis attempt differs from validated staged bytes"
        )

    def _index_record(
        self,
        attempt: SynthesisAttempt,
        synthesis_bytes: bytes,
        corpus_bytes: bytes,
    ) -> dict[str, object]:
        return {
            "attempt_id": attempt.attempt_id,
            "created_at": attempt.created_at,
            "status": _enum_text(attempt.status, "status"),
            "corpus_digest": attempt.corpus_digest,
            "synthesis_digest": _sha256(synthesis_bytes),
            "corpus_manifest_digest": _sha256(corpus_bytes),
        }

    def _read_json_object(
        self, path: Path, label: str
    ) -> tuple[dict[str, object], bytes]:
        try:
            raw = secure_read_bytes(
                path,
                label,
                max_bytes=MAX_SYNTHESIS_JSON_BYTES,
            )
            value = json.loads(
                raw.decode("ascii"),
                object_pairs_hook=_json_object_without_duplicates,
            )
        except SynthesisArtifactError:
            raise
        except (
            OSError,
            RuntimeError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
        ) as error:
            raise SynthesisArtifactError(f"invalid {label}: {error}") from error
        if not isinstance(value, dict):
            raise SynthesisArtifactError(f"invalid {label}: expected an object")
        typed_value = cast(dict[str, object], value)
        try:
            canonical = _canonical_bytes(typed_value)
        except (TypeError, ValueError) as error:
            raise SynthesisArtifactError(f"invalid {label}: {error}") from error
        if raw != canonical:
            raise SynthesisArtifactError(f"{label} is not canonical JSON")
        return typed_value, raw

    def _temporary_attempt_directory(self, attempt_id: str) -> Path:
        try:
            return secure_make_temporary_directory(
                self.attempts_root,
                f".{attempt_id}.",
                "synthesis attempt staging",
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise SynthesisArtifactError(str(error)) from error

    def _publish_attempt(
        self,
        staging: Path,
        destination: Path,
        validated_identity: SecurePathIdentity,
    ) -> SecurePathIdentity:
        if os.path.lexists(destination) or secure_is_link_or_reparse(destination):
            raise SynthesisArtifactError("attempt already exists; overwrite refused")
        try:
            return secure_replace(
                staging,
                destination,
                "synthesis attempt publication",
                replace_existing=False,
                expected_source_identity=validated_identity,
            )
        except SynthesisArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise SynthesisArtifactError(str(error)) from error

    def _quarantine_invalid_published_attempt(
        self,
        destination: Path,
        rejected_identity: SecurePathIdentity,
    ) -> None:
        """Remove a proven mismatch from the immutable attempt namespace."""

        if not os.path.lexists(destination):
            return
        try:
            if secure_is_link_or_reparse(destination):
                secure_unlink(
                    destination,
                    "invalid synthesis attempt publication",
                    expected_identity=rejected_identity,
                )
                return
            quarantine = destination.with_name(
                f".invalid-{destination.name}-{uuid4().hex}"
            )
            quarantine_identity = secure_replace(
                destination,
                quarantine,
                "invalid synthesis attempt quarantine",
                replace_existing=False,
                expected_source_identity=rejected_identity,
            )
            try:
                if quarantine.is_dir():
                    secure_remove_tree(
                        quarantine,
                        "invalid synthesis attempt quarantine cleanup",
                        expected_identity=quarantine_identity,
                    )
                else:
                    secure_unlink(
                        quarantine,
                        "invalid synthesis attempt quarantine cleanup",
                        expected_identity=quarantine_identity,
                    )
            except (OSError, RuntimeError, ValueError):
                pass
        except SynthesisArtifactError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise SynthesisArtifactError(
                "cannot quarantine invalid synthesis attempt publication"
            ) from error

    def _write_index(self, preferred_accepted_id: str | None) -> None:
        ids = self._attempt_ids()
        previous = self._read_index(ids)
        previous_records = self._index_records(previous)
        previous_accepted = (
            cast(str | None, previous.get("accepted_attempt_id"))
            if previous is not None
            else None
        )
        selected = preferred_accepted_id or previous_accepted
        records: list[dict[str, object]] = []
        attempts: dict[str, SynthesisAttempt] = {}
        for attempt_id in ids:
            attempt, synthesis_bytes, corpus_bytes = self._read_attempt_bundle(
                attempt_id
            )
            previous_record = previous_records.get(attempt_id)
            if previous_record is not None:
                self._validate_index_record(
                    previous_record, attempt, synthesis_bytes, corpus_bytes
                )
            attempts[attempt_id] = attempt
            records.append(self._index_record(attempt, synthesis_bytes, corpus_bytes))
        if selected is not None:
            if selected not in attempts:
                raise SynthesisArtifactError("accepted index points at missing attempt")
            if attempts[selected].status not in _ACCEPTED_STATUSES:
                raise SynthesisArtifactError(
                    "accepted index points at a non-accepted attempt"
                )
        index_value = {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "accepted_attempt_id": selected,
            "attempts": records,
        }
        temporary = self.synthesis_root / f".index.{uuid4().hex}.tmp"
        temporary_identity: SecurePathIdentity | None = None
        try:
            index_bytes = _canonical_bytes(index_value)
            _validate_json_size(index_bytes, "synthesis index")
            secure_write_bytes(temporary, index_bytes)
            temporary_identity = secure_path_identity(
                temporary, "synthesis index temporary file"
            )
            _replace_index(temporary, self.index_path)
        except BaseException:
            try:
                if temporary_identity is not None:
                    secure_unlink(
                        temporary,
                        "synthesis index temporary file",
                        missing_ok=True,
                        expected_identity=temporary_identity,
                    )
            except (OSError, RuntimeError, ValueError):
                pass
            raise

    def _remove_temporary_directory(
        self, path: Path, expected_identity: SecurePathIdentity
    ) -> None:
        try:
            secure_remove_tree(
                path,
                "synthesis attempt staging cleanup",
                missing_ok=True,
                expected_identity=expected_identity,
            )
        except (OSError, RuntimeError, ValueError):
            pass


def _replace_index(source: Path, destination: Path) -> None:
    """Atomically replace index.json while refusing link/reparse paths."""

    try:
        secure_assert_ancestors(source, "synthesis index")
        secure_assert_ancestors(destination, "synthesis index")
        if secure_is_link_or_reparse(source) or secure_is_link_or_reparse(destination):
            raise SynthesisArtifactError("synthesis index must not contain links")
        secure_replace(
            source,
            destination,
            "synthesis index publication",
            replace_existing=True,
        )
        secure_assert_ancestors(destination, "synthesis index")
        if secure_is_link_or_reparse(destination):
            raise SynthesisArtifactError("synthesis index must not contain links")
    except SynthesisArtifactError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise SynthesisArtifactError(str(error)) from error


__all__ = [
    "MAX_SYNTHESIS_JSON_BYTES",
    "SynthesisArtifactError",
    "SynthesisArtifactStore",
    "synthesis_attempt_position",
    "validate_publishable_synthesis_attempt",
]
