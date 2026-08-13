"""Fresh-context structured providers for evidence-grounded report synthesis."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import ClassVar, Literal, cast

from pydantic import BaseModel, ValidationError

from ux_analyzer.application.evidence_corpus import (
    EvidenceCorpus,
    EvidenceEntry,
    ResolvedEvidence,
)
from ux_analyzer.domain.findings import EvidenceClass, FindingSeverity, Reproducibility
from ux_analyzer.domain.synthesis import (
    EvidenceRef,
    ObjectionSeverity,
    SynthesisFinding,
    SynthesisObjection,
)
from ux_analyzer.ports.model_transport import (
    MODEL_REQUEST_MAX_BYTES,
    require_finite_float,
)
from ux_analyzer.ports.models import (
    ChatMessage,
    ModelAttachment,
    ModelManifest,
    ModelResponseValidationError,
    ModelRole,
    StructuredModelClient,
)
from ux_analyzer.ports.report_synthesis import (
    FORBIDDEN_NARRATIVE_MARKERS,
    REPORT_SYNTHESIS_SCHEMA_VERSION,
    AdjudicationResponse,
    AnalystResponse,
    AuditorResponse,
    CandidateFinding,
    EvidenceAuditResponse,
    EvidenceReference,
    EvidenceRefSchema,
    FinalFinding,
    FindingSchema,
    InvestigativeResponse,
    ObjectionResolution,
    ObjectionSchema,
    PatternResponse,
    PatternReviewResponse,
    ReportAdjudicatorResponse,
    ReportAnalystResponse,
    ReportEvidenceAuditorResponse,
    ReportPatternReviewerResponse,
    TypedObjection,
    UxPrinciple,
    contains_forbidden_narrative,
    is_sensitive_key,
    redact_forbidden_narrative,
)
from ux_analyzer.providers.ux_principles import ux_principles

_SAFE_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?::[A-Za-z0-9._-]+)+$")
_EVIDENCE_REFERENCE_FIELDS = (
    "evidence_id",
    "kind",
    "run_id",
    "viewport_id",
    "element_id",
    "event_id",
    "metric_id",
    "artifact_path",
    "replay_sequence",
    "sha256",
)
_MANIFEST_REFERENCE_FIELDS = tuple(
    field for field in _EVIDENCE_REFERENCE_FIELDS if field != "artifact_path"
)
_INITIAL_MANIFEST_SUMMARY_MAX_CHARS = 512
# Reserve measured room for role prompts, principles, schemas, and transport data.
_REPORT_REQUEST_RESERVED_OVERHEAD_BYTES = 50_000
_REPORT_REQUEST_MAX_BYTES = MODEL_REQUEST_MAX_BYTES
_INITIAL_MANIFEST_MAX_BYTES = (
    _REPORT_REQUEST_MAX_BYTES - _REPORT_REQUEST_RESERVED_OVERHEAD_BYTES
)
_MANIFEST_VALUE_MAX_CHARS = 512
_MANIFEST_SEQUENCE_MAX_ITEMS = 128
_COMPACT_MANIFEST_FORMAT = "compact-parallel-v1"
_COMPACT_MANIFEST_ENTRY_FIELDS = (
    "kind_index",
    "run_index",
    "viewport_index",
    "evidence_class_index",
    "element_id",
    "event_id",
    "metric_id",
    "replay_sequence",
    "sha256",
)

ManifestInput = EvidenceCorpus | Mapping[str, object]

_OMIT_MANIFEST_VALUE = object()
_MISSING_MANIFEST_FIELD = object()

_MANIFEST_PATH_FRAGMENT = re.compile(
    r"""(?ix)
    (?:
        (?<![a-z0-9])(?:[a-z]:[\\/]|//|\\\\)[^\s\"'<>]+  # drive or UNC path
        |
        (?<![a-z0-9_.:-])[a-z]:[^\\/\s\"'<>]+  # drive-relative path
        |
        (?<![a-z0-9])/(?:[^\s\"'<>/]+/)*[^\s\"'<>/]+  # absolute POSIX path
        |
        (?<![a-z0-9])(?:\.\.?[\\/]|[a-z0-9_.-]+[\\/])[^\s\"'<>]+  # relative path
    )
    """
)
_MANIFEST_URL = re.compile(r"(?ix)\bhttps?://[^\s\"'<>]+")


def _is_manifest_path_key(key: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
    if normalized in {"reference_path", "reference_paths"}:
        return False
    if normalized.endswith(("_path", "_paths")):
        return True
    return normalized in {
        "path",
        "paths",
        "artifact_path",
        "artifact_paths",
        "attachment_path",
        "attachment_paths",
        "file_path",
        "file_paths",
        "filepath",
        "filepaths",
        "filesystem_path",
        "filesystem_paths",
        "absolute_path",
        "absolute_paths",
        "relative_path",
        "relative_paths",
        "output_path",
        "output_paths",
        "output_root",
        "root_path",
        "root_paths",
        "directory",
        "directory_path",
        "directory_paths",
        "dir_path",
        "file",
        "filename",
        "file_name",
    }


def _is_manifest_forbidden_key(key: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
    return (
        normalized in {"payload", "payloads"}
        or normalized.endswith(("_payload", "_payloads"))
        or normalized.startswith("payload_")
        or _is_manifest_path_key(key)
    )


def _manifest_sort_key(value: object, *, depth: int = 0) -> str:
    if depth > 8:
        return "truncated"
    if isinstance(value, os.PathLike):
        path_text = os.fspath(cast(os.PathLike[str], value))
        value_type = type(cast(object, value))
        return f"pathlike:{value_type.__module__}.{value_type.__qualname__}:{path_text}"
    if isinstance(value, BaseModel):
        return _manifest_sort_key(
            value.model_dump(mode="python"),
            depth=depth + 1,
        )
    if is_dataclass(value) and not isinstance(value, type):
        return _manifest_sort_key(asdict(value), depth=depth + 1)
    if isinstance(value, Mapping):
        items = sorted(
            (
                str(key),
                _manifest_sort_key(item, depth=depth + 1),
            )
            for key, item in cast(Mapping[object, object], value).items()
        )
        return f"mapping:{items!r}"
    if isinstance(value, (list, tuple)):
        items = cast(Sequence[object], value)
        return "sequence:" + repr(
            tuple(_manifest_sort_key(item, depth=depth + 1) for item in items)
        )
    if isinstance(value, (set, frozenset)):
        items = cast(Iterable[object], value)
        return "set:" + repr(
            tuple(sorted(_manifest_sort_key(item, depth=depth + 1) for item in items))
        )
    if isinstance(value, str):
        return f"str:{value}"
    if value is None:
        return "none:"
    if isinstance(value, bool):
        return f"bool:{value!s}"
    if isinstance(value, (int, float)):
        return f"number:{type(value).__name__}:{value!r}"
    return f"object:{type(value).__module__}.{type(value).__qualname__}:{value!s}"


def _is_manifest_path_like_string(value: str) -> bool:
    return _MANIFEST_PATH_FRAGMENT.search(_MANIFEST_URL.sub("", value)) is not None


def _bounded_manifest_value(value: object, *, depth: int = 0) -> object:
    if depth > 8:
        return "[truncated]"
    if isinstance(value, float):
        require_finite_float(value, context="bounded manifest")
        return value
    if isinstance(value, os.PathLike):
        return _OMIT_MANIFEST_VALUE
    if isinstance(value, BaseModel):
        return _bounded_manifest_value(value.model_dump(mode="python"), depth=depth + 1)
    if is_dataclass(value) and not isinstance(value, type):
        return _bounded_manifest_value(asdict(value), depth=depth + 1)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        result: dict[str, object] = {}
        for key, item in mapping.items():
            if isinstance(key, float):
                require_finite_float(key, context="bounded manifest key")
            if (
                isinstance(key, os.PathLike)
                or (isinstance(key, str) and _is_manifest_path_like_string(key))
                or _is_sensitive_key(key)
                or _is_manifest_forbidden_key(key)
            ):
                continue
            bounded = _bounded_manifest_value(item, depth=depth + 1)
            if bounded is not _OMIT_MANIFEST_VALUE:
                result[str(key)] = bounded
        return result
    values: list[object] | None
    if isinstance(value, (set, frozenset)):
        values = sorted(
            cast(Iterable[object], value),
            key=_manifest_sort_key,
        )
    elif isinstance(value, (list, tuple)):
        values = list(cast(Iterable[object], value))
    else:
        values = None
    if values is not None:
        bounded_values = [
            bounded
            for item in values[:_MANIFEST_SEQUENCE_MAX_ITEMS]
            if (bounded := _bounded_manifest_value(item, depth=depth + 1))
            is not _OMIT_MANIFEST_VALUE
        ]
        if len(values) > _MANIFEST_SEQUENCE_MAX_ITEMS:
            bounded_values.append("[truncated]")
        return bounded_values
    if isinstance(value, str):
        if _is_manifest_path_like_string(value):
            return _OMIT_MANIFEST_VALUE
        safe_value = _safe_string(value)
        if len(safe_value) <= _MANIFEST_VALUE_MAX_CHARS:
            return safe_value
        return safe_value[: _MANIFEST_VALUE_MAX_CHARS - 3] + "..."
    if value is None or isinstance(value, (bool, int)):
        return value
    return _bounded_manifest_value(str(cast(object, value)), depth=depth + 1)


def _manifest_entries(raw_entries: object) -> tuple[object, ...]:
    if raw_entries is None:
        return ()
    if isinstance(raw_entries, Mapping):
        return (cast(Mapping[object, object], raw_entries),)
    if isinstance(raw_entries, Sequence) and not isinstance(raw_entries, (str, bytes)):
        return tuple(cast(Sequence[object], raw_entries))
    raise TypeError("corpus manifest entries must be a sequence or mapping")


def _manifest_evidence_ids(value: object) -> list[str]:
    if isinstance(value, str):
        if not value.strip():
            raise TypeError("corpus manifest evidence_ids must contain strings")
        return [_canonical_manifest_evidence_id(value)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("corpus manifest evidence_ids must be a sequence of strings")
    result: list[str] = []
    for item in cast(Sequence[object], value):
        if not isinstance(item, str) or not item.strip():
            raise TypeError("corpus manifest evidence_ids must contain strings")
        result.append(_canonical_manifest_evidence_id(item))
    if len(result) != len(set(result)):
        raise ValueError("corpus manifest contains duplicate evidence ID")
    return result


def _canonical_manifest_evidence_id(value: str) -> str:
    normalized = value.strip()
    parts = normalized.split(":")
    if len(parts) < 2:
        raise ValueError("corpus manifest evidence ID namespace is invalid")
    reference = EvidenceRef(normalized, parts[0], parts[1])
    _validate_manifest_reference(reference)
    return normalized


def _validate_manifest_reference(reference: EvidenceRef) -> None:
    # EvidenceEntry owns canonical namespace and field validation for corpus refs.
    EvidenceEntry(
        ref=reference,
        evidence_class=EvidenceClass.DETERMINISTIC_FACT,
        summary="validated manifest reference",
        payload={},
    )


def _mapping_manifest_field(
    entry_mapping: Mapping[object, object],
    reference_mapping: Mapping[object, object],
    name: str,
) -> object:
    entry_value = entry_mapping.get(name, _MISSING_MANIFEST_FIELD)
    reference_value = reference_mapping.get(name, _MISSING_MANIFEST_FIELD)
    if (
        entry_value is not _MISSING_MANIFEST_FIELD
        and reference_value is not _MISSING_MANIFEST_FIELD
        and entry_value != reference_value
    ):
        raise ValueError(
            f"corpus manifest entry has conflicting reference field: {name}"
        )
    if entry_value is not _MISSING_MANIFEST_FIELD:
        return entry_value
    if reference_value is not _MISSING_MANIFEST_FIELD:
        return reference_value
    return None


def _mapping_manifest_entry_reference(
    entry: Mapping[object, object],
) -> EvidenceRef:
    raw_reference = entry.get("reference")
    if raw_reference is None:
        reference_mapping: Mapping[object, object] = {}
    elif isinstance(raw_reference, Mapping):
        reference_mapping = cast(Mapping[object, object], raw_reference)
    else:
        raise TypeError("corpus manifest entry reference must be a mapping")

    raw_id = _mapping_manifest_field(entry, reference_mapping, "evidence_id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError("corpus manifest entry requires an evidence ID")
    evidence_id = _canonical_manifest_evidence_id(raw_id)

    raw_kind = _mapping_manifest_field(entry, reference_mapping, "kind")
    if not isinstance(raw_kind, str) or not raw_kind.strip():
        raise ValueError("corpus manifest entry requires an evidence kind")
    raw_run_id = _mapping_manifest_field(entry, reference_mapping, "run_id")
    if not isinstance(raw_run_id, str) or not raw_run_id.strip():
        raise ValueError("corpus manifest entry requires a run ID")

    def optional_text(name: str) -> str | None:
        value = _mapping_manifest_field(entry, reference_mapping, name)
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"corpus manifest entry reference field must be a string: {name}"
            )
        return value

    viewport_id = optional_text("viewport_id")
    element_id = optional_text("element_id")
    event_id = optional_text("event_id")
    metric_id = optional_text("metric_id")
    sha256 = optional_text("sha256")
    raw_replay_sequence = _mapping_manifest_field(
        entry,
        reference_mapping,
        "replay_sequence",
    )
    if raw_replay_sequence is None:
        replay_sequence = None
    elif isinstance(raw_replay_sequence, bool) or not isinstance(
        raw_replay_sequence, int
    ):
        raise ValueError(
            "corpus manifest entry reference field must be an integer: replay_sequence"
        )
    else:
        replay_sequence = raw_replay_sequence

    reference = EvidenceRef(
        evidence_id,
        raw_kind.strip(),
        raw_run_id.strip(),
        viewport_id=viewport_id,
        element_id=element_id,
        event_id=event_id,
        metric_id=metric_id,
        replay_sequence=replay_sequence,
        sha256=sha256,
    )
    _validate_manifest_reference(reference)
    return reference


def _mapping_manifest_entry_id(entry: object) -> str:
    if isinstance(entry, EvidenceEntry):
        return entry.ref.evidence_id
    if not isinstance(entry, Mapping):
        raise TypeError("corpus manifest entries must be evidence entries or mappings")
    return _mapping_manifest_entry_reference(
        cast(Mapping[object, object], entry)
    ).evidence_id


def _mapping_manifest_entry_ids(
    source: Mapping[object, object],
    entries: Sequence[object],
) -> list[str]:
    entry_ids = [_mapping_manifest_entry_id(entry) for entry in entries]
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("corpus manifest contains duplicate evidence ID")
    if "evidence_ids" in source:
        declared_ids = _manifest_evidence_ids(source["evidence_ids"])
        if declared_ids != entry_ids:
            raise ValueError(
                "corpus manifest evidence_ids must match entry IDs in row order"
            )
    return entry_ids


def _is_sensitive_key(key: object) -> bool:
    return is_sensitive_key(key)


def _safe_string(value: str) -> str:
    return redact_forbidden_narrative(value)


def _safe_prompt_value(value: object, *, depth: int = 0) -> object:
    if depth > 8:
        return "[truncated]"
    if isinstance(value, float):
        require_finite_float(value, context="canonical JSON")
        return value
    if isinstance(value, EvidenceEntry):
        return _safe_entry(value, depth=depth + 1)
    if isinstance(value, EvidenceRef):
        return _safe_evidence_ref(value)
    if isinstance(value, BaseModel):
        return _safe_prompt_value(value.model_dump(mode="python"), depth=depth + 1)
    if isinstance(value, UxPrinciple):
        return _safe_prompt_value(asdict(value), depth=depth + 1)
    if isinstance(value, SynthesisFinding):
        return _safe_prompt_value(_domain_finding_payload(value), depth=depth + 1)
    if isinstance(value, SynthesisObjection):
        return _safe_prompt_value(_domain_objection_payload(value), depth=depth + 1)
    if is_dataclass(value) and not isinstance(value, type):
        return _safe_prompt_value(asdict(value), depth=depth + 1)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        result: dict[str, object] = {}
        for key, item in mapping.items():
            if isinstance(key, float):
                require_finite_float(key, context="canonical JSON key")
            if _is_sensitive_key(key):
                continue
            safe_item = _safe_prompt_value(item, depth=depth + 1)
            result[str(key)] = safe_item
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _safe_prompt_value(item, depth=depth + 1)
            for item in cast(Iterable[object], value)
        ]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, str):
        return _safe_string(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    return str(value)


def _safe_evidence_ref(
    reference: EvidenceRef,
    *,
    fields: Sequence[str] = _EVIDENCE_REFERENCE_FIELDS,
) -> dict[str, object]:
    return {
        name: value
        for name in fields
        if (value := getattr(reference, name)) is not None
    }


def _safe_entry(entry: EvidenceEntry, *, depth: int = 0) -> dict[str, object]:
    return {
        "evidence_id": entry.ref.evidence_id,
        "kind": entry.ref.kind,
        "run_id": entry.ref.run_id,
        "reference": _safe_evidence_ref(entry.ref),
        "evidence_class": entry.evidence_class.value,
        "summary": _safe_prompt_value(entry.summary, depth=depth + 1),
        "payload": _safe_prompt_value(entry.payload, depth=depth + 1),
    }


def _bounded_manifest_summary(value: object) -> str:
    summary = _bounded_manifest_value(value)
    if summary is _OMIT_MANIFEST_VALUE:
        return ""
    if not isinstance(summary, str):
        summary = str(summary)
    if len(summary) <= _INITIAL_MANIFEST_SUMMARY_MAX_CHARS:
        return summary
    return summary[: _INITIAL_MANIFEST_SUMMARY_MAX_CHARS - 3] + "..."


def _bounded_manifest_entry(entry: object) -> dict[str, object]:
    if isinstance(entry, EvidenceEntry):
        return {
            "evidence_id": entry.ref.evidence_id,
            "kind": entry.ref.kind,
            "run_id": entry.ref.run_id,
            "reference": _safe_evidence_ref(
                entry.ref,
                fields=_MANIFEST_REFERENCE_FIELDS,
            ),
            "evidence_class": entry.evidence_class.value,
            "summary": _bounded_manifest_summary(entry.summary),
        }
    if not isinstance(entry, Mapping):
        raise TypeError("corpus manifest entries must be evidence entries or mappings")

    entry_mapping = cast(Mapping[object, object], entry)
    reference_value = _mapping_manifest_entry_reference(entry_mapping)

    def field(name: str) -> object:
        if name == "evidence_id":
            return reference_value.evidence_id
        if name in _EVIDENCE_REFERENCE_FIELDS:
            return getattr(reference_value, name)
        return entry_mapping.get(name)

    reference: dict[str, object] = {}
    for name in _MANIFEST_REFERENCE_FIELDS:
        if name == "evidence_id":
            reference[name] = reference_value.evidence_id
            continue
        value = getattr(reference_value, name)
        if value is not None:
            bounded = _bounded_manifest_value(value)
            if bounded is not _OMIT_MANIFEST_VALUE:
                reference[name] = bounded

    def bounded_field(name: str) -> object:
        value = _bounded_manifest_value(field(name))
        return None if value is _OMIT_MANIFEST_VALUE else value

    return {
        "evidence_id": reference_value.evidence_id,
        "kind": bounded_field("kind"),
        "run_id": bounded_field("run_id"),
        "reference": reference,
        "evidence_class": bounded_field("evidence_class"),
        "summary": _bounded_manifest_summary(field("summary")),
    }


def _bounded_manifest_entry_with_summary_limit(
    entry: object,
    summary_max_chars: int,
) -> dict[str, object]:
    bounded = _bounded_manifest_entry(entry)
    if summary_max_chars < _INITIAL_MANIFEST_SUMMARY_MAX_CHARS:
        summary = bounded.get("summary")
        if isinstance(summary, str):
            if summary_max_chars <= 0:
                bounded.pop("summary", None)
            elif len(summary) > summary_max_chars:
                bounded["summary"] = summary[:summary_max_chars]
    return bounded


def _compact_manifest_candidate(
    payload: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compact repeated selection metadata without changing evidence IDs."""
    kind_values: list[str] = []
    kind_indexes: dict[str, int] = {}
    run_ids: list[str] = []
    run_indexes: dict[str, int] = {}
    viewport_values: list[str] = []
    viewport_indexes: dict[str, int] = {}
    evidence_class_values: list[str] = []
    evidence_class_indexes: dict[str, int] = {}

    def index_value(
        value: object,
        values: list[str],
        indexes: dict[str, int],
    ) -> int | None:
        if not isinstance(value, str) or not value:
            return None
        value_index = indexes.get(value)
        if value_index is None:
            value_index = len(values)
            indexes[value] = value_index
            values.append(value)
        return value_index

    evidence_ids: list[str] = []
    seen_evidence_ids: set[str] = set()
    compact_entries: list[list[object]] = []
    for entry in entries:
        evidence_id = entry.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("compact manifest entry requires an evidence ID")
        evidence_id = evidence_id.strip()
        if evidence_id in seen_evidence_ids:
            raise ValueError("compact manifest contains duplicate evidence ID")
        evidence_ids.append(evidence_id)
        seen_evidence_ids.add(evidence_id)
        kind_index = index_value(entry.get("kind"), kind_values, kind_indexes)
        run_index = index_value(entry.get("run_id"), run_ids, run_indexes)
        reference_value = entry.get("reference")
        reference: Mapping[object, object] | None = (
            cast(Mapping[object, object], reference_value)
            if isinstance(reference_value, Mapping)
            else None
        )
        viewport_index = index_value(
            reference.get("viewport_id") if reference is not None else None,
            viewport_values,
            viewport_indexes,
        )
        evidence_class_index = index_value(
            entry.get("evidence_class"),
            evidence_class_values,
            evidence_class_indexes,
        )
        compact: list[object] = [
            kind_index,
            run_index,
            viewport_index,
            evidence_class_index,
        ]
        compact.extend(
            reference.get(field) if reference is not None else None
            for field in _COMPACT_MANIFEST_ENTRY_FIELDS[4:]
        )
        while compact and compact[-1] is None:
            compact.pop()
        compact_entries.append(compact)

    declared_ids = payload.get("evidence_ids")
    if declared_ids is not None:
        normalized_declared_ids = _manifest_evidence_ids(declared_ids)
        if normalized_declared_ids != evidence_ids:
            raise ValueError(
                "compact manifest evidence_ids must match entry IDs in row order"
            )

    result = dict(payload)
    result["manifest_format"] = _COMPACT_MANIFEST_FORMAT
    result["evidence_ids"] = evidence_ids
    result["entry_fields"] = list(_COMPACT_MANIFEST_ENTRY_FIELDS)
    result["kind_values"] = kind_values
    result["run_ids"] = run_ids
    result["viewport_values"] = viewport_values
    result["evidence_class_values"] = evidence_class_values
    result["entries"] = compact_entries
    return result


_MANIFEST_CONTEXT_FIELDS = (
    "schema_version",
    "principle_pack_version",
    "principle_pack_digest",
    "metadata",
    "expectation",
    "scenario",
    "persona",
    "goal",
    "scope",
    "tested_scope",
    "evidence_ids",
)


def _domain_finding_payload(finding: SynthesisFinding) -> dict[str, object]:
    return {
        "finding_id": finding.finding_id,
        "title": finding.title,
        "issue": finding.issue,
        "impact": finding.impact,
        "root_cause": finding.root_cause,
        "fixes": finding.fixes,
        "severity": cast(FindingSeverity, finding.severity).value,
        "confidence": finding.confidence,
        "evidence_refs": [_safe_evidence_ref(ref) for ref in finding.evidence_refs],
        "affected_surfaces": finding.affected_surfaces,
        "principles": finding.principles,
        "counterevidence": [
            _safe_evidence_ref(item) if isinstance(item, EvidenceRef) else item
            for item in finding.counterevidence
        ],
        "limitations": finding.limitations,
        "reviewer_state": finding.reviewer_state,
        "evidence_class": cast(EvidenceClass, finding.evidence_class).value,
        "reproducibility": cast(Reproducibility, finding.reproducibility).value,
        "severity_justification": finding.severity_justification,
        "reviewer_notes": finding.reviewer_notes,
    }


def _domain_objection_payload(objection: SynthesisObjection) -> dict[str, object]:
    return {
        "objection_id": objection.objection_id,
        "finding_id": objection.finding_id,
        "severity": cast(ObjectionSeverity, objection.severity).value,
        "message": objection.message,
        "evidence_refs": [_safe_evidence_ref(ref) for ref in objection.evidence_refs],
        "reviewer_role": objection.reviewer_role,
        "resolved": objection.resolved,
        "resolution": objection.resolution,
    }


def _manifest_payload(manifest: object) -> dict[str, object]:
    if isinstance(manifest, EvidenceCorpus):
        payload = {
            "schema_version": "evidence-corpus-v1",
            "principle_pack_version": manifest.principle_pack_version,
            "principle_pack_digest": manifest.principle_pack_digest,
            "metadata": _bounded_manifest_value(manifest.metadata),
        }
        entries: Sequence[object] = manifest.entries
    elif isinstance(manifest, Mapping):
        source = cast(Mapping[object, object], manifest)
        payload = {}
        for field_name in _MANIFEST_CONTEXT_FIELDS:
            if field_name not in source:
                continue
            raw_value = source[field_name]
            if field_name == "evidence_ids":
                payload[field_name] = _manifest_evidence_ids(raw_value)
                continue
            bounded = _bounded_manifest_value(raw_value)
            if bounded is not _OMIT_MANIFEST_VALUE:
                payload[field_name] = bounded
        entries = _manifest_entries(source.get("entries"))
        payload["evidence_ids"] = _mapping_manifest_entry_ids(source, entries)
    else:
        raise TypeError("corpus manifest must be an EvidenceCorpus or mapping")

    def candidate(summary_max_chars: int) -> dict[str, object]:
        result = dict(payload)
        result["entries"] = [
            _bounded_manifest_entry_with_summary_limit(entry, summary_max_chars)
            for entry in entries
        ]
        return result

    def size(value: Mapping[str, object]) -> int:
        return len(_canonical_json(value).encode("utf-8"))

    full = candidate(_INITIAL_MANIFEST_SUMMARY_MAX_CHARS)
    if size(full) <= _INITIAL_MANIFEST_MAX_BYTES:
        return full

    compact = _compact_manifest_candidate(
        payload,
        [
            _bounded_manifest_entry_with_summary_limit(
                entry,
                _INITIAL_MANIFEST_SUMMARY_MAX_CHARS,
            )
            for entry in entries
        ],
    )
    if size(compact) > _INITIAL_MANIFEST_MAX_BYTES:
        raise ValueError(
            "initial evidence manifest identifiers exceed hard byte budget"
        )
    return compact


def _initial_manifest_payload(manifest: ManifestInput) -> dict[str, object]:
    """Return the compact, index-only context sent at role invocation time."""

    bounded = _manifest_payload(manifest)
    if bounded.get("manifest_format") == _COMPACT_MANIFEST_FORMAT:
        return bounded
    entries = bounded.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("bounded corpus manifest entries must be a sequence")
    bounded_entries = cast(Sequence[object], entries)
    mapping_entries = [
        cast(Mapping[str, object], entry)
        for entry in bounded_entries
        if isinstance(entry, Mapping)
    ]
    if len(mapping_entries) != len(bounded_entries):
        raise TypeError("bounded corpus manifest entries must be mappings")
    return _compact_manifest_candidate(bounded, mapping_entries)


def _known_evidence_ids(manifest: ManifestInput) -> frozenset[str]:
    if isinstance(manifest, EvidenceCorpus):
        return frozenset(entry.ref.evidence_id for entry in manifest.entries)
    manifest_mapping = manifest
    values: set[str] = set()
    entries = _manifest_entries(manifest_mapping.get("entries"))
    entry_ids: list[str] = []
    for entry in entries:
        if isinstance(entry, EvidenceEntry):
            entry_ids.append(entry.ref.evidence_id)
        elif isinstance(entry, Mapping):
            entry_ids.append(
                _mapping_manifest_entry_id(cast(Mapping[object, object], entry))
            )
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("corpus manifest contains duplicate evidence ID")
    if "evidence_ids" in manifest_mapping:
        declared_ids = _manifest_evidence_ids(manifest_mapping["evidence_ids"])
        if declared_ids != entry_ids:
            raise ValueError(
                "corpus manifest evidence_ids must match entry IDs in row order"
            )
    values.update(entry_ids)
    return frozenset(values)


def _resolved_payload(resolved: ResolvedEvidence | None) -> list[object]:
    if resolved is None:
        return []
    return [_safe_entry(entry) for entry in resolved.entries]


def _principle_payload(
    principles: object | None,
) -> tuple[UxPrinciple, ...]:
    if principles is None:
        return ux_principles()
    if isinstance(principles, (str, bytes)):
        raise TypeError("principles must be a sequence of UxPrinciple values")
    if not isinstance(principles, Sequence):
        raise TypeError("principles must be a sequence of UxPrinciple values")
    normalized_values = tuple(cast(Sequence[object], principles))
    if any(not isinstance(item, UxPrinciple) for item in normalized_values):
        raise TypeError("principles must contain UxPrinciple values")
    return cast(tuple[UxPrinciple, ...], normalized_values)


def _attachment_values(
    manifest: ManifestInput,
    resolved: ResolvedEvidence | None,
) -> tuple[ModelAttachment, ...]:
    if resolved is None or not isinstance(manifest, EvidenceCorpus):
        return ()
    attachments: list[ModelAttachment] = []
    for entry in resolved.entries:
        if entry.attachment_path is None or entry.ref.sha256 is None:
            continue
        media_type = entry.payload.get("media_type")
        if media_type not in {"image/png", "image/jpeg"}:
            continue
        typed_media_type = cast(Literal["image/png", "image/jpeg"], media_type)
        attachments.append(
            ModelAttachment(
                evidence_id=entry.ref.evidence_id,
                path=manifest.output_root / entry.attachment_path,
                media_type=typed_media_type,
                sha256=entry.ref.sha256,
            )
        )
    return tuple(attachments)


def _canonical_json(value: object) -> str:
    return json.dumps(
        _safe_prompt_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _raw_canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _request_context_size(
    schema: type[BaseModel],
    messages: Sequence[ChatMessage],
    *,
    model: str,
    role: ModelRole,
    resolved_evidence: ResolvedEvidence | None,
    client: StructuredModelClient | None = None,
) -> int:
    request_size = getattr(client, "request_size", None)
    if callable(request_size):
        transport_request_size = cast(Callable[..., int], request_size)
        return int(
            transport_request_size(
                schema,
                messages,
                model=model,
                role=role,
            )
        )
    request: dict[str, object] = {
        "model": model,
        "role": role.value,
        "messages": [
            {
                **message.model_dump(),
                "attachments": [
                    {
                        "evidence_id": attachment.evidence_id,
                        "media_type": attachment.media_type,
                        "sha256": attachment.sha256,
                    }
                    for attachment in message.attachments
                ],
            }
            for message in messages
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": str(getattr(schema, "schema_version", schema.__name__)).replace(
                    "-", "_"
                ),
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        },
    }
    serialized_size = len(_raw_canonical_json(request).encode("utf-8"))
    if resolved_evidence is not None and resolved_evidence.attachment_bytes:
        encoded_bytes = ((resolved_evidence.attachment_bytes + 2) // 3) * 4
        serialized_size += encoded_bytes
    return serialized_size


class _ReportRole:
    response_schema: ClassVar[type[InvestigativeResponse]]
    role: ClassVar[ModelRole]
    prompt_version: ClassVar[str]

    def __init__(self, client: StructuredModelClient, *, model: object) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("report model must not be empty")
        self.client = client
        self.model: str = model

    @property
    def manifest(self) -> ModelManifest:
        return ModelManifest(
            provider_id=str(
                getattr(
                    client := self.client, "provider_id", "openai-compatible-structured"
                )
            ),
            role=self.role,
            model_id=self.model,
            endpoint_origin=str(getattr(client, "endpoint_origin", "unknown")),
            prompt_version=self.prompt_version,
            schema_version=self.response_schema.schema_version,
            provider_version=str(
                getattr(client, "provider_version", "openai-compatible-v1")
            ),
        )

    @property
    def prompt(self) -> str:
        return _COMMON_PROMPT + "\n\n" + self._role_prompt

    @property
    def _role_prompt(self) -> str:
        raise NotImplementedError

    async def _complete(
        self,
        manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None,
        *,
        role_input: Mapping[str, object] | None = None,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: InvestigativeResponse | None = None,
    ) -> InvestigativeResponse:
        normalized_principles = _principle_payload(principles)
        if previous_output is not None and not isinstance(
            previous_output, self.response_schema
        ):
            raise ValueError("previous output must belong to this role")
        message_payload: dict[str, object] = {
            "corpus_manifest": _initial_manifest_payload(manifest),
            "resolved_evidence": _resolved_payload(resolved_evidence),
            "ux_principle_pack": [asdict(item) for item in normalized_principles],
            "role_input": dict(role_input or {}),
        }
        if previous_output is not None:
            message_payload["prior_structured_output"] = previous_output.model_dump(
                mode="json"
            )
        messages = (
            ChatMessage(role="system", content=self.prompt),
            ChatMessage(
                role="user",
                content=_canonical_json(message_payload),
                attachments=_attachment_values(manifest, resolved_evidence),
            ),
        )
        request_size = _request_context_size(
            self.response_schema,
            messages,
            model=self.model,
            role=self.role,
            resolved_evidence=resolved_evidence,
            client=self.client,
        )
        has_transport_measurement = callable(getattr(self.client, "request_size", None))
        measured_size = (
            request_size
            if has_transport_measurement
            else (request_size + _REPORT_REQUEST_RESERVED_OVERHEAD_BYTES)
        )
        if measured_size > _REPORT_REQUEST_MAX_BYTES:
            raise ValueError(
                "report synthesis request context exceeds transport-safe byte budget"
            )
        try:
            response = await self.client.complete(
                self.response_schema,
                messages,
                model=self.model,
                role=self.role,
            )
        except ModelResponseValidationError:
            raise
        except ValidationError as error:
            raise ModelResponseValidationError(
                self.role,
                "response schema validation failed",
                response_summary={"schema": self.response_schema.__name__},
            ) from error
        except (TypeError, ValueError) as error:
            raise ModelResponseValidationError(
                self.role,
                "response schema validation failed",
                response_summary={"schema": self.response_schema.__name__},
            ) from error

        if not isinstance(response, self.response_schema):
            try:
                response = self.response_schema.model_validate(response)
            except ValidationError as error:
                raise ModelResponseValidationError(
                    self.role,
                    "response schema validation failed",
                    response_summary={"schema": self.response_schema.__name__},
                ) from error
        self._validate_response(response, manifest, normalized_principles)
        return response

    def _validate_response(
        self,
        response: InvestigativeResponse,
        manifest: ManifestInput,
        principles: Sequence[UxPrinciple],
    ) -> None:
        known_ids = _known_evidence_ids(manifest)
        if not response.complete and not response.evidence_requests:
            self._invalid("incomplete response needs evidence requests")
        for evidence_id in response.evidence_requests:
            self._validate_requested_id(evidence_id, known_ids)

        principle_ids = {principle.principle_id for principle in principles}
        for finding in self._findings(response):
            self._validate_finding(finding, known_ids, principle_ids)
        for objection in self._objections(response):
            self._validate_refs(objection.evidence_refs, known_ids)
        for resolution in self._resolutions(response):
            self._validate_refs(resolution.evidence_refs, known_ids)

    def _validate_requested_id(
        self,
        evidence_id: str,
        known_ids: frozenset[str],
    ) -> None:
        if _SAFE_EVIDENCE_ID.fullmatch(evidence_id) is None:
            self._invalid("invalid evidence request ID")
        if evidence_id.startswith("principle:"):
            self._invalid("principles are not evidence")
        if not known_ids:
            self._invalid("evidence request cannot be validated without corpus IDs")
        if evidence_id not in known_ids:
            self._invalid(f"unknown evidence ID: {evidence_id}")

    def _validate_refs(
        self,
        refs: Sequence[EvidenceReference],
        known_ids: frozenset[str],
    ) -> None:
        evidence_ids = [reference.evidence_id for reference in refs]
        if len(evidence_ids) != len(set(evidence_ids)):
            self._invalid("duplicate evidence ID")
        for reference in refs:
            if reference.evidence_id.startswith("principle:"):
                self._invalid("principles are not evidence")
            if reference.evidence_id not in known_ids:
                self._invalid(f"unknown evidence ID: {reference.evidence_id}")

    def _validate_finding(
        self,
        finding: CandidateFinding,
        known_ids: frozenset[str],
        principle_ids: set[str],
    ) -> None:
        refs = tuple(finding.evidence_refs) + tuple(
            item
            for item in finding.counterevidence
            if isinstance(item, EvidenceReference)
        )
        self._validate_refs(refs, known_ids)
        unknown_principles = set(finding.principles) - principle_ids
        if unknown_principles:
            self._invalid("finding references an unknown UX principle")
        if not finding.evidence_refs:
            self._invalid("principles are not evidence")

    @staticmethod
    def _findings(response: InvestigativeResponse) -> tuple[CandidateFinding, ...]:
        if isinstance(response, AnalystResponse):
            return tuple(response.candidate_findings)
        if isinstance(response, AdjudicationResponse):
            return tuple(response.final_findings)
        return ()

    @staticmethod
    def _objections(response: InvestigativeResponse) -> tuple[TypedObjection, ...]:
        if isinstance(response, (EvidenceAuditResponse, PatternReviewResponse)):
            return tuple(response.objections)
        return ()

    @staticmethod
    def _resolutions(
        response: InvestigativeResponse,
    ) -> tuple[ObjectionResolution, ...]:
        if isinstance(response, AdjudicationResponse):
            return tuple(response.objection_resolutions)
        return ()

    def _invalid(self, reason: str) -> None:
        raise ModelResponseValidationError(
            self.role,
            reason,
            response_summary={"schema": self.response_schema.__name__},
        )


_COMMON_PROMPT = """You are one isolated report-synthesis role.

Use only the allowlisted structured evidence in corpus_manifest and resolved_evidence. Do not use prior prompts, raw model responses, private reasoning, chat history, cognitive prose, or existing finding prose. Frozen Expectations describe outcomes, invariants, acceptable alternatives, and warning signals. Treat reference paths as examples, not the only correct path. Judge path deviations only through observed user impact and supported outcomes.

The initial corpus_manifest is always an index-only manifest_format compact-parallel-v1. Each entries row position corresponds to the same position in evidence_ids and follows entry_fields; trailing null fields may be omitted. kind_index, run_index, viewport_index, and evidence_class_index index kind_values, run_ids, viewport_values, and evidence_class_values. Use row metadata to choose evidence IDs, then request full evidence only by those IDs. The initial manifest has no full evidence payload or summary; resolved_evidence is the exact on-demand retrieval channel.

The UX principle pack is optional interpretive guidance. Principles are not evidence and cannot determine severity. A principle may help name or explain an issue only when observed evidence supports it. Never request or cite a principle as an evidence ID.

Every factual claim and finding must use resolvable evidence IDs from corpus_manifest. If more evidence is needed, set complete to false and request only valid evidence IDs listed in corpus_manifest. Return exactly one valid JSON object matching the requested structured response schema. Do not include private reasoning or extra fields."""


class ReportAnalyst(_ReportRole):
    """Discover evidence-backed UX issues and plausible root causes."""

    role = ModelRole.REPORT_ANALYST
    prompt_version = "report-analyst-v1"
    response_schema = AnalystResponse

    @property
    def _role_prompt(self) -> str:
        return "Discover material UX issues and their likely root causes. Emit candidate findings with plain language, concrete fixes, evidence references, limitations, and justified severity."

    async def analyze(
        self,
        corpus_manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None = None,
        *,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: AnalystResponse | None = None,
    ) -> AnalystResponse:
        response = await self._complete(
            corpus_manifest,
            principles,
            resolved_evidence=resolved_evidence,
            previous_output=previous_output,
            role_input={"task": "discover issues and root causes"},
        )
        return cast(AnalystResponse, response)


class EvidenceAuditor(_ReportRole):
    """Challenge factual and visual support for analyst candidates."""

    role = ModelRole.REPORT_EVIDENCE_AUDITOR
    prompt_version = "report-evidence-auditor-v1"
    response_schema = EvidenceAuditResponse

    @property
    def _role_prompt(self) -> str:
        return "Challenge factual support, visual interpretation, citation accuracy, and contradictions in candidate findings. Emit typed objections tied to the evidence needed to resolve each challenge."

    async def audit(
        self,
        corpus_manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None = None,
        candidate_findings: Sequence[
            CandidateFinding | SynthesisFinding | Mapping[str, object]
        ] = (),
        *,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: EvidenceAuditResponse | None = None,
    ) -> EvidenceAuditResponse:
        response = await self._complete(
            corpus_manifest,
            principles,
            resolved_evidence=resolved_evidence,
            previous_output=previous_output,
            role_input={
                "task": "audit factual and visual support",
                "candidate_findings": list(candidate_findings),
            },
        )
        return cast(EvidenceAuditResponse, response)

    async def review(
        self,
        corpus_manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None = None,
        candidate_findings: Sequence[
            CandidateFinding | SynthesisFinding | Mapping[str, object]
        ] = (),
        *,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: EvidenceAuditResponse | None = None,
    ) -> EvidenceAuditResponse:
        return await self.audit(
            corpus_manifest,
            principles,
            candidate_findings,
            resolved_evidence=resolved_evidence,
            previous_output=previous_output,
        )


class PatternReviewer(_ReportRole):
    """Review recurrence, cross-surface impact, severity, and fix leverage."""

    role = ModelRole.REPORT_PATTERN_REVIEWER
    prompt_version = "report-pattern-reviewer-v1"
    response_schema = PatternReviewResponse

    @property
    def _role_prompt(self) -> str:
        return "Check recurrence, affected surfaces, counterexamples, shared causes, severity, and fix leverage. Emit typed objections when a pattern or priority claim is not established by evidence."

    async def review(
        self,
        corpus_manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None = None,
        candidate_findings: Sequence[
            CandidateFinding | SynthesisFinding | Mapping[str, object]
        ] = (),
        *,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: PatternReviewResponse | None = None,
    ) -> PatternReviewResponse:
        response = await self._complete(
            corpus_manifest,
            principles,
            resolved_evidence=resolved_evidence,
            previous_output=previous_output,
            role_input={
                "task": "review recurrence, scope, severity, and fix leverage",
                "candidate_findings": list(candidate_findings),
            },
        )
        return cast(PatternReviewResponse, response)


class ReportAdjudicator(_ReportRole):
    """Resolve reviewer objections and write plain-language final findings."""

    role = ModelRole.REPORT_ADJUDICATOR
    prompt_version = "report-adjudicator-v1"
    response_schema = AdjudicationResponse

    @property
    def _role_prompt(self) -> str:
        return "This role resolves objections and writes plain-language final findings. Publish only findings with supported evidence, concrete fixes, justified severity, and explicit resolutions for every objection."

    async def adjudicate(
        self,
        corpus_manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None = None,
        candidate_findings: Sequence[
            CandidateFinding | SynthesisFinding | Mapping[str, object]
        ] = (),
        objections: Sequence[
            TypedObjection | SynthesisObjection | Mapping[str, object]
        ] = (),
        *,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: AdjudicationResponse | None = None,
    ) -> AdjudicationResponse:
        response = await self._complete(
            corpus_manifest,
            principles,
            resolved_evidence=resolved_evidence,
            previous_output=previous_output,
            role_input={
                "task": "resolve objections and write final findings",
                "candidate_findings": list(candidate_findings),
                "objections": list(objections),
            },
        )
        return cast(AdjudicationResponse, response)

    async def resolve(
        self,
        corpus_manifest: ManifestInput,
        principles: Sequence[UxPrinciple] | None = None,
        candidate_findings: Sequence[
            CandidateFinding | SynthesisFinding | Mapping[str, object]
        ] = (),
        objections: Sequence[
            TypedObjection | SynthesisObjection | Mapping[str, object]
        ] = (),
        *,
        resolved_evidence: ResolvedEvidence | None = None,
        previous_output: AdjudicationResponse | None = None,
    ) -> AdjudicationResponse:
        return await self.adjudicate(
            corpus_manifest,
            principles,
            candidate_findings,
            objections,
            resolved_evidence=resolved_evidence,
            previous_output=previous_output,
        )


__all__ = [
    "AdjudicationResponse",
    "AnalystResponse",
    "AuditorResponse",
    "CandidateFinding",
    "EvidenceAuditResponse",
    "EvidenceAuditor",
    "EvidenceReference",
    "EvidenceRefSchema",
    "FinalFinding",
    "FindingSchema",
    "InvestigativeResponse",
    "ManifestInput",
    "ObjectionResolution",
    "ObjectionSchema",
    "PatternResponse",
    "PatternReviewResponse",
    "PatternReviewer",
    "ReportAdjudicator",
    "ReportAdjudicatorResponse",
    "ReportAnalyst",
    "ReportAnalystResponse",
    "ReportEvidenceAuditorResponse",
    "ReportPatternReviewerResponse",
    "REPORT_SYNTHESIS_SCHEMA_VERSION",
    "TypedObjection",
    "FORBIDDEN_NARRATIVE_MARKERS",
    "contains_forbidden_narrative",
    "redact_forbidden_narrative",
]
