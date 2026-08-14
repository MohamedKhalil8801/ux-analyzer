from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from ux_analyzer.application.evidence_corpus import (
    EvidenceCorpus,
    EvidenceEntry,
    EvidenceResolver,
)
from ux_analyzer.domain.findings import EvidenceClass, FindingSeverity
from ux_analyzer.domain.synthesis import EvidenceRef, ObjectionSeverity
from ux_analyzer.ports.model_transport import TransportBudgetError
from ux_analyzer.ports.models import ModelResponseValidationError, ModelRole
from ux_analyzer.providers.report_synthesis import (
    _INITIAL_MANIFEST_MAX_BYTES,
    _REPORT_MODEL_CONTEXT_MAX_ENTRIES,
    _REPORT_REQUEST_MAX_BYTES,
    _REPORT_REQUEST_RESERVED_OVERHEAD_BYTES,
    AdjudicationResponse,
    AnalystResponse,
    CandidateFinding,
    EvidenceAuditor,
    EvidenceAuditResponse,
    ObjectionResolution,
    PatternReviewer,
    PatternReviewResponse,
    ReportAdjudicator,
    ReportAnalyst,
    ReportAnalystResponse,
    ReportEvidenceAuditorResponse,
    ReportPatternReviewerResponse,
    TypedObjection,
    _bounded_manifest_value,
    _canonical_json,
    _initial_manifest_payload,
    _known_evidence_ids,
    _manifest_payload,
)
from ux_analyzer.providers.ux_principles import ux_principles

EVIDENCE_ID = "event:run-a:1"


@dataclass(frozen=True)
class _TypedManifestMetadata:
    safe_scalar: str
    posix_path_value: object
    pure_path_value: object
    nested: dict[str, object]
    sequence: tuple[object, ...]


class RecordingClient:
    endpoint_origin = "https://llm.example.test/v1"
    provider_id = "recording-client"
    provider_version = "recording-v1"

    def __init__(
        self,
        response_factory: Callable[[type[Any], ModelRole], object] | None = None,
    ) -> None:
        self.calls: list[tuple[type[Any], tuple[Any, ...], str, ModelRole]] = []
        self.messages: list[Any] = []
        self.response_factory = response_factory or self._default_response

    async def complete(
        self,
        schema: type[Any],
        messages: Sequence[Any],
        model: str,
        role: ModelRole,
    ) -> object:
        message_tuple = tuple(messages)
        self.calls.append((schema, message_tuple, model, role))
        self.messages.extend(message_tuple)
        return self.response_factory(schema, role)

    @staticmethod
    def _default_response(schema: type[Any], role: ModelRole) -> object:
        del role
        payload: dict[str, object] = {
            "complete": True,
            "evidence_requests": [],
        }
        if schema is AnalystResponse:
            payload["candidate_findings"] = []
        elif schema in {EvidenceAuditResponse, PatternReviewResponse}:
            payload["objections"] = []
        elif schema is AdjudicationResponse:
            payload["final_findings"] = []
            payload["objection_resolutions"] = []
        return schema.model_validate(payload)


class MeasuringRecordingClient(RecordingClient):
    def request_size(
        self,
        schema: type[Any],
        messages: Sequence[Any],
        *,
        model: str,
        role: ModelRole,
    ) -> int:
        del schema, messages, model, role
        return _REPORT_REQUEST_MAX_BYTES + 1


class AttachmentAwareRecordingClient(RecordingClient):
    def __init__(self) -> None:
        super().__init__()
        self.measured_sizes: list[int] = []

    def request_size(
        self,
        schema: type[Any],
        messages: Sequence[Any],
        *,
        model: str,
        role: ModelRole,
    ) -> int:
        del schema, model, role
        has_attachments = any(message.attachments for message in messages)
        resolved_evidence = json.loads(messages[1].content)["resolved_evidence"]
        measured = (
            _REPORT_REQUEST_MAX_BYTES + 1
            if has_attachments or len(resolved_evidence) > 1
            else _REPORT_REQUEST_MAX_BYTES - 1
        )
        self.measured_sizes.append(measured)
        return measured


def _manifest(*, include_sentinels: bool = False) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "evidence-corpus-v1",
        "principle_pack_version": "ux-principles-v1",
        "entries": [
            {
                "evidence_id": EVIDENCE_ID,
                "kind": "event",
                "run_id": "run-a",
                "summary": "User opened the invite control.",
                "payload": {"sequence": 1, "action": "interact"},
            }
        ],
        "expectation": {
            "reference_paths": [["open-team", "invite", "confirm"]],
            "acceptable_alternatives": ["Use the team page."],
        },
    }
    if include_sentinels:
        manifest.update(
            {
                "prior_agent_private_reasoning": "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL",
                "prior_finding_prose": "PRIOR_FINDING_PROSE_SENTINEL",
                "private_reasoning": "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL",
                "existing_finding_title": "PRIOR_FINDING_PROSE_SENTINEL",
            }
        )
        manifest["entries"] = [
            {
                **manifest["entries"][0],  # type: ignore[index]
                "payload": {
                    "sequence": 1,
                    "private_reasoning": "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL",
                    "finding_prose": "PRIOR_FINDING_PROSE_SENTINEL",
                },
            }
        ]
    return manifest


def _corpus(tmp_path: Path) -> EvidenceCorpus:
    return EvidenceCorpus(
        output_root=tmp_path,
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(EVIDENCE_ID, "event", "run-a", replay_sequence=1),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="User opened the invite control.",
                payload={"sequence": 1, "action": "interact"},
            ),
        ),
    )


def _finding_payload(*, principle_only: bool = False) -> dict[str, object]:
    return {
        "finding_id": "invite-control",
        "title": "Invite control is hard to find",
        "issue": "People look in unrelated areas before finding the invite control.",
        "impact": "Important collaboration tasks take longer.",
        "root_cause": "The entry point is labeled around internal structure.",
        "fixes": ["Label the entry point around the user's goal."],
        "severity": FindingSeverity.MEDIUM,
        "confidence": 0.8,
        "evidence_refs": []
        if principle_only
        else [
            {
                "evidence_id": EVIDENCE_ID,
                "kind": "event",
                "run_id": "run-a",
                "replay_sequence": 1,
            }
        ],
        "principles": ["mental-models"],
        "severity_justification": "The task is important and the evidence shows extra navigation.",
    }


@pytest.mark.asyncio
async def test_analyst_prompt_has_boundary_and_excludes_prior_agent_context() -> None:
    client = RecordingClient()
    analyst = ReportAnalyst(client, model="gpt-report")

    await analyst.analyze(_manifest(include_sentinels=True), ux_principles())

    prompt = client.messages[0].content
    serialized_messages = json.dumps(
        [message.model_dump() for message in client.messages],
        ensure_ascii=True,
    )
    assert "Treat reference paths as examples, not the only correct path" in prompt
    assert "manifest_format compact-parallel-v2" in prompt
    assert (
        "Return exactly one valid JSON object matching the requested structured response schema"
        in prompt
    )
    assert "element_index" in prompt
    assert "element_values" in prompt
    assert "run_ids, viewport_values" in prompt
    assert "PRIOR_AGENT_PRIVATE_REASONING_SENTINEL" not in serialized_messages
    assert "PRIOR_FINDING_PROSE_SENTINEL" not in serialized_messages


@pytest.mark.asyncio
async def test_plain_report_request_delivers_role_schema_contract() -> None:
    client = RecordingClient()
    analyst = ReportAnalyst(client, model="gpt-report")

    await analyst.analyze(_manifest(), ux_principles())

    payload = json.loads(client.messages[1].content)
    contract = payload["response_schema"]
    assert contract["role"] == ModelRole.REPORT_ANALYST.value
    assert contract["schema_version"] == AnalystResponse.schema_version
    schema = contract["schema"]
    expected_schema = AnalystResponse.model_json_schema()
    assert schema["type"] == expected_schema["type"]
    assert schema["required"] == expected_schema["required"]
    assert schema["properties"] == expected_schema["properties"]
    assert schema["$defs"].keys() == expected_schema["$defs"].keys()


@pytest.mark.asyncio
async def test_initial_manifest_is_bounded_but_resolved_evidence_keeps_payload() -> (
    None
):
    client = RecordingClient()
    analyst = ReportAnalyst(client, model="gpt-report")
    second_evidence_id = "event:run-a:2"
    large_payload = {"large": "payload-sentinel" * 1000}
    corpus = EvidenceCorpus(
        output_root=Path.cwd(),
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(
                    EVIDENCE_ID,
                    "event",
                    "run-a",
                    event_id="event-1",
                    replay_sequence=1,
                    viewport_id="viewport-1",
                ),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="summary-sentinel" * 1000,
                payload=large_payload,
            ),
            EvidenceEntry(
                ref=EvidenceRef(
                    second_evidence_id,
                    "event",
                    "run-a",
                    event_id="event-2",
                    replay_sequence=2,
                    viewport_id="viewport-1",
                ),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="Second event.",
                payload={"sequence": 2},
            ),
        ),
    )
    resolved = EvidenceResolver().resolve(
        corpus,
        [EVIDENCE_ID],
        max_entries=2,
        max_attachment_bytes=1024,
    )

    await analyst.analyze(corpus, resolved_evidence=resolved)

    message = json.loads(client.messages[1].content)
    manifest = message["corpus_manifest"]
    manifest_entries = manifest["entries"]
    manifest_entry = manifest_entries[0]
    resolved_entry = message["resolved_evidence"][0]
    serialized_manifest = json.dumps(manifest, ensure_ascii=True, separators=(",", ":"))
    assert manifest["manifest_format"] == "compact-parallel-v2"
    assert "evidence_ids" not in manifest
    assert manifest["entry_fields"] == [
        "handle_index",
        "kind_index",
        "run_index",
        "viewport_index",
        "evidence_class_index",
        "event_index",
        "metric_index",
        "replay_sequence",
        "element_index",
    ]
    assert manifest["kind_values"] == ["event"]
    assert manifest["run_ids"] == ["run-a"]
    assert manifest["viewport_values"] == ["viewport-1"]
    assert manifest["event_values"] == ["event-1", "event-2"]
    assert manifest["metric_values"] == []
    assert manifest_entry == [0, 0, 0, 0, 0, 0, None, 1]
    assert manifest_entries[1] == [1, 0, 0, 0, 0, 1, None, 2]
    assert len(serialized_manifest.encode("utf-8")) < 1_000
    assert "summary" not in manifest["entry_fields"]
    assert "payload-sentinel" not in serialized_manifest
    assert "summary-sentinel" not in serialized_manifest
    assert resolved_entry["payload"] == large_payload


def test_initial_manifest_index_preserves_context_and_reference_metadata() -> None:
    corpus = EvidenceCorpus(
        output_root=Path.cwd(),
        metadata={
            "scenario": {"id": "invite", "name": "Invite"},
            "persona": {"id": "admin", "name": "Workspace administrator"},
            "goal": "Invite a teammate to the workspace",
        },
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(
                    EVIDENCE_ID,
                    "event",
                    "run-a",
                    event_id="event-1",
                    replay_sequence=1,
                    viewport_id="viewport-1",
                    element_id="target",
                ),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="Do not send this summary in the initial prompt.",
                payload={"secret": "full evidence payload"},
            ),
        ),
    )

    manifest = _initial_manifest_payload(corpus)
    fields = manifest["entry_fields"]
    row = manifest["entries"][0]

    assert manifest["metadata"] == corpus.metadata
    assert "evidence_ids" not in manifest
    assert manifest["run_ids"] == ["run-a"]
    assert manifest["viewport_values"] == ["viewport-1"]
    assert manifest["element_values"] == ["target"]
    assert row[fields.index("handle_index")] == 0
    assert row[fields.index("element_index")] == 0
    assert row[fields.index("replay_sequence")] == 1
    serialized = _canonical_json(manifest)
    assert "full evidence payload" not in serialized
    assert "Do not send this summary" not in serialized


def test_initial_manifest_uses_compact_handles_without_exposing_full_ids() -> None:
    corpus = EvidenceCorpus(
        output_root=Path.cwd(),
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(
                    EVIDENCE_ID,
                    "event",
                    "run-a",
                    event_id="event-1",
                    replay_sequence=1,
                    viewport_id="viewport-1",
                ),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="Event summary stays out of the initial manifest.",
                payload={"secret": "resolved evidence only"},
            ),
        ),
    )

    manifest = _initial_manifest_payload(corpus)
    serialized = _canonical_json(manifest)

    assert manifest["manifest_format"] == "compact-parallel-v2"
    assert "evidence_ids" not in manifest
    assert manifest["entry_fields"] == [
        "handle_index",
        "kind_index",
        "run_index",
        "viewport_index",
        "evidence_class_index",
        "event_index",
        "metric_index",
        "replay_sequence",
        "element_index",
    ]
    assert manifest["entries"][0][0] == 0
    assert manifest["run_ids"] == ["run-a"]
    assert manifest["viewport_values"] == ["viewport-1"]
    assert manifest["event_values"] == ["event-1"]
    assert manifest["metric_values"] == []
    assert EVIDENCE_ID not in serialized
    assert "resolved evidence only" not in serialized


@pytest.mark.asyncio
async def test_report_role_expands_provider_handles_before_delivery_validation() -> (
    None
):
    class HandleClient(RecordingClient):
        @staticmethod
        def _handle_response(schema: type[Any], role: ModelRole) -> object:
            del role
            if schema is AnalystResponse:
                candidate = _finding_payload()
                candidate["evidence_refs"] = [
                    {
                        **candidate["evidence_refs"][0],  # type: ignore[index]
                        "evidence_id": "e0",
                    }
                ]
                return schema.model_validate(
                    {
                        "complete": False,
                        "evidence_requests": ["e0"],
                        "candidate_findings": [candidate],
                    }
                )
            return RecordingClient._default_response(schema, ModelRole.REPORT_ANALYST)

        def __init__(self) -> None:
            super().__init__(self._handle_response)

    with pytest.raises(ModelResponseValidationError, match="undelivered evidence ID"):
        await ReportAnalyst(HandleClient(), model="gpt-report").analyze(
            _manifest(), ux_principles()
        )


def test_initial_manifest_omits_attachment_and_filesystem_path_fields() -> None:
    corpus = EvidenceCorpus(
        output_root=Path.cwd(),
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(
                    f"screenshot:run-a:{'a' * 64}",
                    "screenshot",
                    "run-a",
                    viewport_id="viewport-1",
                    artifact_path="runs/run-a/artifacts/screenshot.png",
                    sha256="a" * 64,
                ),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="A screenshot summary.",
                payload={"media_type": "image/png"},
                attachment_path=Path("runs/run-a/artifacts/screenshot.png"),
            ),
        ),
    )

    manifest = _manifest_payload(corpus)
    serialized = json.dumps(manifest, ensure_ascii=True, sort_keys=True)

    for manifest_entry in manifest["entries"]:
        assert not {"artifact_path", "attachment_path", "path"}.intersection(
            manifest_entry
        )
        assert not {"artifact_path", "attachment_path", "path"}.intersection(
            manifest_entry["reference"]
        )
    assert "runs/run-a/artifacts/screenshot.png" not in serialized
    assert "media_type" not in serialized


def test_strict_manifest_serializer_omits_pathlike_values_and_path_strings() -> None:
    typed_metadata = _bounded_manifest_value(
        _TypedManifestMetadata(
            safe_scalar="safe typed metadata",
            posix_path_value=Path("/tmp/typed-path-secret.json"),
            pure_path_value=PurePath("C:/typed-pure-path-secret.json"),
            nested={
                "safe_nested": "safe nested metadata",
                "path_string": "workspace/typed-nested-secret.json",
                "windows_path": PureWindowsPath(
                    "C:/typed-nested-pure-path-secret.json"
                ),
            },
            sequence=(
                "safe sequence metadata",
                "/tmp/typed-sequence-secret.json",
                PurePosixPath("/tmp/typed-sequence-pure-path-secret.json"),
            ),
        )
    )
    mapping_manifest = _manifest_payload(
        {
            "metadata": {
                "safe_scalar": "safe mapping metadata",
                "posix_path_value": "/tmp/mapping-path-secret.json",
                "pure_path_value": PurePath("C:/mapping-pure-path-secret.json"),
                "nested": {
                    "safe_nested": "safe nested mapping metadata",
                    "path_string": "workspace/mapping-nested-secret.json",
                    "/tmp/path-named-key.json": "path-key-secret",
                },
                "sequence": [
                    "safe mapping sequence metadata",
                    r"C:\mapping-sequence-secret.json",
                    "file:///tmp/file-uri-secret.json",
                    {"nested_path_string": "reports/mapping-report.json"},
                ],
            },
            "entries": [
                {
                    "evidence_id": EVIDENCE_ID,
                    "kind": "event",
                    "run_id": "run-a",
                    "summary": "Summary contains /tmp/mapping-summary-secret.json",
                }
            ],
        }
    )
    serialized = _canonical_json({"typed": typed_metadata, "mapping": mapping_manifest})

    assert typed_metadata == {
        "safe_scalar": "safe typed metadata",
        "nested": {"safe_nested": "safe nested metadata"},
        "sequence": ["safe sequence metadata"],
    }
    assert mapping_manifest["metadata"] == {
        "safe_scalar": "safe mapping metadata",
        "nested": {"safe_nested": "safe nested mapping metadata"},
        "sequence": ["safe mapping sequence metadata", {}],
    }
    assert mapping_manifest["entries"][0]["summary"] == ""
    for secret in (
        "typed-path-secret",
        "typed-pure-path-secret",
        "typed-nested-secret",
        "typed-nested-pure-path-secret",
        "typed-sequence-secret",
        "typed-sequence-pure-path-secret",
        "mapping-path-secret",
        "mapping-pure-path-secret",
        "mapping-nested-secret",
        "mapping-sequence-secret",
        "mapping-report.json",
        "mapping-summary-secret",
        "path-key-secret",
        "file:///tmp/file-uri-secret.json",
    ):
        assert secret not in serialized


def test_strict_manifest_serializer_omits_drive_relative_paths() -> None:
    typed_corpus = EvidenceCorpus(
        output_root=Path.cwd(),
        metadata={"drive_relative": "H:typed-corpus-drive-relative-secret.txt"},
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(EVIDENCE_ID, "event", "run-a"),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="I:typed-corpus-summary-drive-relative-secret.txt",
                payload={"drive_relative": "J:typed-corpus-payload-secret.txt"},
            ),
        ),
    )
    typed = _bounded_manifest_value(
        {
            "safe": "safe typed value",
            "drive_relative": "C:typed-drive-relative-secret.txt",
            "nested": {"drive_relative": "D:nested-drive-relative-secret.txt"},
            "sequence": ["E:sequence-drive-relative-secret.txt"],
        }
    )
    mapping = _manifest_payload(
        {
            "metadata": {
                "drive_relative": "F:mapping-drive-relative-secret.txt",
            },
            "entries": [
                {
                    "evidence_id": EVIDENCE_ID,
                    "kind": "event",
                    "run_id": "run-a",
                    "summary": "G:summary-drive-relative-secret.txt",
                }
            ],
        }
    )

    serialized = _canonical_json(
        {"corpus": _manifest_payload(typed_corpus), "typed": typed, "mapping": mapping}
    )

    assert typed == {"safe": "safe typed value", "nested": {}, "sequence": []}
    assert mapping["metadata"] == {}
    assert mapping["entries"][0]["summary"] == ""
    for secret in (
        "typed-drive-relative-secret",
        "nested-drive-relative-secret",
        "sequence-drive-relative-secret",
        "mapping-drive-relative-secret",
        "summary-drive-relative-secret",
        "typed-corpus-drive-relative-secret",
        "typed-corpus-summary-drive-relative-secret",
        "typed-corpus-payload-secret",
    ):
        assert secret not in serialized


def test_typed_corpus_summary_uses_strict_manifest_serializer() -> None:
    corpus = EvidenceCorpus(
        output_root=Path.cwd(),
        metadata={
            "safe": "safe corpus metadata",
            "nested": {"path_value": "/tmp/corpus-metadata-secret.json"},
        },
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(EVIDENCE_ID, "event", "run-a"),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="Observed at C:/corpus-summary-secret.json",
                payload={"safe": "payload stays resolved-only"},
            ),
        ),
    )

    serialized = _canonical_json(_manifest_payload(corpus))

    assert "safe corpus metadata" in serialized
    assert "corpus-metadata-secret" not in serialized
    assert "corpus-summary-secret" not in serialized


def test_mapping_manifest_rejects_duplicate_entry_ids_before_compaction() -> None:
    entry = {
        "evidence_id": EVIDENCE_ID,
        "kind": "event",
        "run_id": "run-a",
        "event_id": "event-1",
        "summary": "Duplicate mapping evidence.",
    }

    with pytest.raises(ValueError, match="duplicate evidence ID"):
        _manifest_payload({"entries": [entry, dict(entry)]})


def test_mapping_manifest_rejects_missing_entry_ids_before_compaction() -> None:
    with pytest.raises(ValueError, match="evidence ID"):
        _manifest_payload(
            {
                "entries": [
                    {
                        "kind": "event",
                        "run_id": "run-a",
                        "event_id": "event-without-id",
                        "summary": "Mapping evidence without an ID.",
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    ("evidence_id", "kind", "run_id", "message"),
    (
        ("forged:run-a:1", "forged", "run-a", "namespace"),
        (EVIDENCE_ID, "metric", "run-a", "kind"),
        (EVIDENCE_ID, "event", "run-b", "run"),
        ("event:run-a:event-1", "event", "run-a", "sequence"),
    ),
)
def test_mapping_manifest_rejects_noncanonical_or_mismatched_ids(
    evidence_id: str,
    kind: str,
    run_id: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _manifest_payload(
            {
                "entries": [
                    {
                        "evidence_id": evidence_id,
                        "kind": kind,
                        "run_id": run_id,
                        "summary": "Mapping evidence with an invalid reference.",
                    }
                ]
            }
        )


def test_mapping_manifest_rejects_conflicting_nested_reference_fields() -> None:
    with pytest.raises(ValueError, match="reference field"):
        _manifest_payload(
            {
                "entries": [
                    {
                        "evidence_id": EVIDENCE_ID,
                        "kind": "event",
                        "run_id": "run-a",
                        "summary": "Mapping evidence with conflicting fields.",
                        "reference": {
                            "evidence_id": EVIDENCE_ID,
                            "kind": "metric",
                            "run_id": "run-a",
                        },
                    }
                ]
            }
        )


def test_mapping_manifest_rejects_mismatched_nested_event_fields() -> None:
    with pytest.raises(ValueError, match="sequence"):
        _manifest_payload(
            {
                "entries": [
                    {
                        "evidence_id": EVIDENCE_ID,
                        "kind": "event",
                        "run_id": "run-a",
                        "event_id": "event-2",
                        "replay_sequence": 1,
                        "summary": "Mapping evidence with a mismatched event field.",
                    }
                ]
            }
        )


def test_mapping_manifest_rejects_invalid_declared_and_known_ids() -> None:
    with pytest.raises(ValueError, match="namespace"):
        _manifest_payload(
            {
                "evidence_ids": ["forged:run-a:1"],
                "entries": [
                    {
                        "evidence_id": EVIDENCE_ID,
                        "kind": "event",
                        "run_id": "run-a",
                        "summary": "Mapping evidence with a forged declaration.",
                    }
                ],
            }
        )

    with pytest.raises(ValueError, match="namespace"):
        _known_evidence_ids({"evidence_ids": ["forged:run-a:1"], "entries": []})

    with pytest.raises(ValueError, match="evidence_ids"):
        _known_evidence_ids(
            {
                "evidence_ids": ["event:run-a:2"],
                "entries": [
                    {
                        "evidence_id": EVIDENCE_ID,
                        "kind": "event",
                        "run_id": "run-a",
                        "summary": "Mapping evidence with a mismatched declaration.",
                    }
                ],
            }
        )


def test_mapping_manifest_accepts_valid_ids_at_all_reference_levels() -> None:
    manifest = {
        "evidence_ids": [EVIDENCE_ID],
        "entries": [
            {
                "evidence_id": EVIDENCE_ID,
                "kind": "event",
                "run_id": "run-a",
                "event_id": "event-1",
                "replay_sequence": 1,
                "summary": "Canonical mapping evidence.",
                "reference": {
                    "evidence_id": EVIDENCE_ID,
                    "kind": "event",
                    "run_id": "run-a",
                    "event_id": "event-1",
                    "replay_sequence": 1,
                },
            }
        ],
    }

    bounded = _manifest_payload(manifest)

    assert bounded["evidence_ids"] == [EVIDENCE_ID]
    assert bounded["entries"][0]["reference"] == {
        "evidence_id": EVIDENCE_ID,
        "kind": "event",
        "run_id": "run-a",
        "event_id": "event-1",
        "replay_sequence": 1,
    }
    assert _known_evidence_ids(manifest) == frozenset({EVIDENCE_ID})


def test_mapping_manifest_rejects_declared_ids_that_do_not_match_rows() -> None:
    with pytest.raises(ValueError, match="evidence_ids"):
        _manifest_payload(
            {
                "evidence_ids": [EVIDENCE_ID, "event:run-a:2"],
                "entries": [
                    {
                        "evidence_id": EVIDENCE_ID,
                        "kind": "event",
                        "run_id": "run-a",
                        "summary": "One mapping row.",
                    }
                ],
            }
        )


def test_mapping_manifest_uses_bounded_allowlisted_entries_and_context() -> None:
    evidence_id = "event:run-a:1"
    manifest = {
        "schema_version": "evidence-corpus-v1",
        "principle_pack_version": "ux-principles-v1",
        "payload": {"top_level": "mapping-payload-sentinel" * 1000},
        "artifact_path": "C:/private/top-level-artifact.json",
        "attachment_path": "C:/private/top-level-attachment.png",
        "path": "C:/private/top-level-path.txt",
        "metadata": {
            "payload": "nested-payload-sentinel",
            "path": "C:/private/nested-path.txt",
            "safe": "safe metadata" * 1000,
        },
        "expectation": {
            "reference_paths": [["open-team", "invite", "confirm"]],
            "payload": "expectation-payload-sentinel",
            "artifact_path": "C:/private/expectation.json",
        },
        "entries": [
            {
                "evidence_id": evidence_id,
                "kind": "event",
                "run_id": "run-a",
                "evidence_class": "deterministic-fact",
                "summary": "Observed event.",
                "payload": {"entry": "entry-payload-sentinel" * 1000},
                "artifact_path": "C:/private/entry-artifact.json",
                "attachment_path": "C:/private/entry-attachment.png",
                "path": "C:/private/entry-path.txt",
                "reference": {
                    "evidence_id": evidence_id,
                    "kind": "event",
                    "run_id": "run-a",
                    "event_id": "event-1",
                    "payload": "reference-payload-sentinel",
                    "artifact_path": "C:/private/reference-artifact.json",
                    "attachment_path": "C:/private/reference-attachment.png",
                    "path": "C:/private/reference-path.txt",
                },
            }
        ],
    }

    bounded = _manifest_payload(manifest)
    serialized = _canonical_json(bounded)
    entry = bounded["entries"][0]

    assert len(serialized.encode("utf-8")) <= _INITIAL_MANIFEST_MAX_BYTES
    assert bounded["expectation"]["reference_paths"] == [
        ["open-team", "invite", "confirm"]
    ]
    assert entry["evidence_id"] == evidence_id
    assert entry["reference"] == {
        "evidence_id": evidence_id,
        "kind": "event",
        "run_id": "run-a",
        "event_id": "event-1",
    }
    assert "payload" not in bounded
    assert "artifact_path" not in bounded
    assert "attachment_path" not in bounded
    assert "path" not in bounded
    assert "payload" not in bounded["metadata"]
    assert "path" not in bounded["metadata"]
    assert "payload" not in entry
    assert not {"artifact_path", "attachment_path", "path"}.intersection(entry)
    assert not {"artifact_path", "attachment_path", "path"}.intersection(
        entry["reference"]
    )
    assert all(
        sentinel not in serialized
        for sentinel in (
            "mapping-payload-sentinel",
            "nested-payload-sentinel",
            "expectation-payload-sentinel",
            "entry-payload-sentinel",
            "reference-payload-sentinel",
            "C:/private/",
        )
    )


def test_mapping_manifest_owns_compact_marker_fields() -> None:
    bounded = _manifest_payload(
        {
            "manifest_format": "compact-parallel-v1",
            "entry_fields": ["forged-entry-fields"],
            "kind_values": ["forged-kind"],
            "run_ids": ["forged-run"],
            "viewport_values": ["forged-viewport"],
            "entries": [
                {
                    "evidence_id": EVIDENCE_ID,
                    "kind": "event",
                    "run_id": "run-a",
                    "summary": "Small manifest entry.",
                }
            ],
        }
    )

    assert isinstance(bounded["entries"][0], dict)
    assert bounded["entries"][0]["evidence_id"] == EVIDENCE_ID
    assert not {
        "manifest_format",
        "entry_fields",
        "kind_values",
        "run_ids",
        "viewport_values",
    }.intersection(bounded)
    assert "forged" not in _canonical_json(bounded)


def test_mapping_manifest_normalizes_single_entry_and_rejects_scalar_entries() -> None:
    evidence_id = "event:run-a:1"
    bounded = _manifest_payload(
        {
            "goal": "Find the invite control.",
            "entries": {
                "evidence_id": evidence_id,
                "kind": "event",
                "run_id": "run-a",
                "summary": "Single mapping entry.",
                "payload": {"leak": "single-entry-payload-sentinel"},
                "path": "C:/private/single-entry.txt",
            },
        }
    )

    assert bounded["entries"] == [
        {
            "evidence_id": evidence_id,
            "kind": "event",
            "run_id": "run-a",
            "reference": {
                "evidence_id": evidence_id,
                "kind": "event",
                "run_id": "run-a",
            },
            "evidence_class": None,
            "summary": "Single mapping entry.",
        }
    ]
    assert "single-entry-payload-sentinel" not in _canonical_json(bounded)

    with pytest.raises(TypeError, match="corpus manifest entries"):
        _manifest_payload({"entries": "unbounded-entry-sentinel" * 10000})


def test_large_mapping_manifest_is_bounded_and_keeps_every_evidence_id() -> None:
    entry_count = 2_070
    entries = [
        {
            "evidence_id": f"event:run-{index % 6}:{index}",
            "kind": "event",
            "run_id": f"run-{index % 6}",
            "event_id": f"event-{index}",
            "summary": (f"Mapping evidence summary {index}. " * 40),
            "payload": "mapping-payload-sentinel" * 1000,
            "artifact_path": f"runs/run-{index % 6}/events/{index}.json",
            "attachment_path": f"runs/run-{index % 6}/events/{index}.png",
            "reference": {
                "evidence_id": f"event:run-{index % 6}:{index}",
                "kind": "event",
                "run_id": f"run-{index % 6}",
                "event_id": f"event-{index}",
                "path": f"C:/private/event-{index}.json",
            },
        }
        for index in range(1, entry_count + 1)
    ]

    manifest = _manifest_payload(
        {
            "schema_version": "evidence-corpus-v1",
            "payload": "top-level-mapping-payload-sentinel" * 1000,
            "artifact_path": "C:/private/top-level.json",
            "entries": entries,
        }
    )
    serialized = _canonical_json(manifest).encode("utf-8")
    expected_ids = [entry["evidence_id"] for entry in entries]

    assert len(serialized) <= _INITIAL_MANIFEST_MAX_BYTES
    assert manifest["manifest_format"] == "compact-parallel-v1"
    assert manifest["evidence_ids"] == expected_ids
    assert len(manifest["entries"]) == entry_count
    event_index = manifest["entry_fields"].index("event_id")
    assert [row[event_index] for row in manifest["entries"]] == [
        f"event-{index}" for index in range(1, entry_count + 1)
    ]
    assert "mapping-payload-sentinel" not in serialized.decode("utf-8")
    assert "C:/private/" not in serialized.decode("utf-8")


def test_mapping_manifest_serializes_set_metadata_deterministically() -> None:
    code = (
        "from ux_analyzer.providers.report_synthesis import "
        "_canonical_json, _manifest_payload; "
        "print(_canonical_json(_manifest_payload({"
        "'metadata': {'values': {'alpha', 'bravo', 'charlie', 'delta', "
        "'echo', 'foxtrot', 'golf', 'hotel', 'india', 'juliet'}}, "
        "'entries': []})))"
    )
    outputs: list[str] = []
    for seed in ("1", "2", "3"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.check_output(
                (sys.executable, "-c", code),
                cwd=Path.cwd(),
                env=environment,
                text=True,
            ).strip()
        )

    assert outputs[0] == outputs[1] == outputs[2]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_bounded_manifest_rejects_non_finite_float_nested_in_dataclass(
    value: float,
) -> None:
    metadata = _TypedManifestMetadata(
        safe_scalar="safe",
        posix_path_value="/private/omit.txt",
        pure_path_value=PurePath("private/omit.txt"),
        nested={"values": [value]},
        sequence=(value,),
    )

    with pytest.raises(ValueError, match="finite"):
        _bounded_manifest_value({"metadata": [metadata]})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_float_nested_in_mapping_and_list(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        _canonical_json({"outer": [{"value": value}]})


def test_full_size_initial_manifest_stays_under_hard_budget_and_keeps_ids() -> None:
    entry_count = 2_070
    entries = tuple(
        EvidenceEntry(
            ref=EvidenceRef(
                f"element:run-{index % 6}-{'a' * 64}:"
                f"viewport-{index % 10}-verification-1:"
                f"run-{index % 6}-{'a' * 64}-element-{index}",
                "element",
                run_id := f"run-{index % 6}-{'a' * 64}",
                f"viewport-{index % 10}-verification-1",
                f"{run_id}-element-{index}",
                artifact_path=f"runs/{run_id}/elements/{index}.json",
                replay_sequence=index,
            ),
            evidence_class=EvidenceClass.DETERMINISTIC_FACT,
            summary=(f"Evidence summary {index}. " * 40),
            payload={"sequence": index, "full_payload": "payload-sentinel"},
        )
        for index in range(1, entry_count + 1)
    )
    corpus = EvidenceCorpus(output_root=Path.cwd(), entries=entries)

    manifest = _manifest_payload(corpus)
    serialized = _canonical_json(manifest).encode("utf-8")
    manifest_ids = set(manifest["evidence_ids"])
    expected_ids = {entry.ref.evidence_id for entry in corpus.entries}
    resolved = EvidenceResolver().resolve(
        corpus,
        [entries[0].ref.evidence_id],
        max_entries=1,
        max_attachment_bytes=1024,
    )

    assert manifest_ids == expected_ids
    assert len(serialized) <= _INITIAL_MANIFEST_MAX_BYTES
    assert not {"artifact_path", "attachment_path", "path"}.intersection(
        manifest["entry_fields"]
    )
    assert "payload" not in manifest["entry_fields"]
    assert "payload-sentinel" not in serialized.decode("utf-8")
    assert "summary" not in manifest["entry_fields"]
    serialized_text = serialized.decode("utf-8")
    for forbidden_key in ("artifact_path", "attachment_path", "path"):
        assert forbidden_key not in serialized_text
    assert manifest["entry_fields"] == [
        "kind_index",
        "run_index",
        "viewport_index",
        "evidence_class_index",
        "element_id",
        "event_id",
        "metric_id",
        "replay_sequence",
        "sha256",
    ]
    assert not {"artifact_path", "attachment_path", "path"}.intersection(
        manifest["entry_fields"]
    )
    assert manifest["kind_values"]
    assert manifest["run_ids"]
    assert manifest["viewport_values"]
    assert manifest["manifest_format"] == "compact-parallel-v1"
    assert manifest["evidence_ids"] == [
        entry.ref.evidence_id for entry in corpus.entries
    ]
    first_row = manifest["entries"][0]
    assert manifest["kind_values"][first_row[0]] == corpus.entries[0].ref.kind
    assert manifest["run_ids"][first_row[1]] == corpus.entries[0].ref.run_id
    assert (
        manifest["viewport_values"][first_row[2]] == corpus.entries[0].ref.viewport_id
    )
    assert all(
        len(entry) <= len(manifest["entry_fields"]) for entry in manifest["entries"]
    )
    assert resolved.entries[0].payload["full_payload"] == "payload-sentinel"


@pytest.mark.asyncio
async def test_complete_near_limit_request_stays_below_transport_ceiling() -> None:
    entry_count = 2_070
    entries = tuple(
        EvidenceEntry(
            ref=EvidenceRef(
                f"element:run-{index % 6}-{'a' * 64}:"
                f"viewport-{index % 10}-verification-1:"
                f"run-{index % 6}-{'a' * 64}-element-{index}",
                "element",
                run_id := f"run-{index % 6}-{'a' * 64}",
                f"viewport-{index % 10}-verification-1",
                f"{run_id}-element-{index}",
                artifact_path=f"runs/{run_id}/elements/{index}.json",
                replay_sequence=index,
            ),
            evidence_class=EvidenceClass.DETERMINISTIC_FACT,
            summary=(f"Evidence summary {index}. " * 40),
            payload={"sequence": index, "full_payload": "payload-sentinel"},
        )
        for index in range(1, entry_count + 1)
    )
    corpus = EvidenceCorpus(output_root=Path.cwd(), entries=entries)
    client = RecordingClient()

    await ReportAnalyst(client, model="gpt-report").analyze(corpus)

    messages = client.calls[0][1]
    request = {
        "model": "gpt-report",
        "messages": [message.model_dump() for message in messages],
    }
    serialized_request = json.dumps(
        request,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    message_payload = json.loads(messages[1].content)

    assert len(serialized_request) <= _REPORT_REQUEST_MAX_BYTES
    assert (
        len(serialized_request) + _REPORT_REQUEST_RESERVED_OVERHEAD_BYTES
        <= _REPORT_REQUEST_MAX_BYTES
    )
    provider_manifest = message_payload["corpus_manifest"]
    assert provider_manifest["manifest_format"] == "compact-parallel-v2"
    assert "evidence_ids" not in provider_manifest
    assert len(provider_manifest["entries"]) == entry_count
    assert provider_manifest["viewport_values"] == [
        f"viewport-{index % 10}-verification-1" for index in range(1, 11)
    ]
    viewport_index = provider_manifest["entry_fields"].index("viewport_index")
    assert (
        provider_manifest["viewport_values"][
            provider_manifest["entries"][0][viewport_index]
        ]
        == "viewport-1-verification-1"
    )
    assert provider_manifest["event_values"] == []
    assert provider_manifest["metric_values"] == []


@pytest.mark.asyncio
async def test_oversized_resolved_context_is_rejected_before_model_client() -> None:
    corpus = EvidenceCorpus(
        output_root=Path.cwd(),
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(EVIDENCE_ID, "event", "run-a", replay_sequence=1),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="Large resolved evidence.",
                payload={"large": "resolved-payload-sentinel" * 40_000},
            ),
        ),
    )
    resolved = EvidenceResolver().resolve(
        corpus,
        [EVIDENCE_ID],
        max_entries=1,
        max_attachment_bytes=1024,
    )
    client = RecordingClient()

    with pytest.raises(
        TransportBudgetError, match="transport-safe byte budget"
    ) as failure:
        await ReportAnalyst(client, model="gpt-report").analyze(
            corpus,
            resolved_evidence=resolved,
        )

    assert client.calls == []
    assert failure.value.diagnostics["stage"] == "request_budget"
    assert failure.value.diagnostics["budget_bytes"] == _REPORT_REQUEST_MAX_BYTES


@pytest.mark.asyncio
async def test_report_preflight_uses_transport_owned_request_measurement() -> None:
    client = MeasuringRecordingClient()

    with pytest.raises(TransportBudgetError, match="transport-safe byte budget"):
        await ReportAnalyst(client, model="gpt-report").analyze(_manifest())

    assert client.calls == []


@pytest.mark.asyncio
async def test_over_budget_visual_attachment_is_deferred_before_model_client(
    tmp_path: Path,
) -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (1, 1), color=(220, 80, 80)).save(image_buffer, format="PNG")
    attachment_bytes = image_buffer.getvalue()
    attachment_path = Path("runs/run-a/screenshot.png")
    corpus_root = tmp_path
    target = corpus_root / attachment_path
    target.parent.mkdir(parents=True)
    target.write_bytes(attachment_bytes)
    attachment_digest = hashlib.sha256(attachment_bytes).hexdigest()
    corpus = EvidenceCorpus(
        output_root=corpus_root,
        entries=(
            EvidenceEntry(
                ref=EvidenceRef(
                    f"screenshot:run-a:{attachment_digest}",
                    "screenshot",
                    "run-a",
                    viewport_id="viewport-1",
                    artifact_path=attachment_path.as_posix(),
                    sha256=attachment_digest,
                ),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="A recorded screenshot.",
                payload={"media_type": "image/png"},
                attachment_path=attachment_path,
            ),
        ),
    )
    resolved = EvidenceResolver().resolve(
        corpus,
        [f"screenshot:run-a:{attachment_digest}"],
        max_entries=1,
        max_attachment_bytes=1024,
    )
    evidence_id = f"screenshot:run-a:{attachment_digest}"
    candidate = CandidateFinding.model_validate(
        {
            **_finding_payload(),
            "evidence_refs": [
                {
                    "evidence_id": evidence_id,
                    "kind": "screenshot",
                    "run_id": "run-a",
                    "viewport_id": "viewport-1",
                    "sha256": attachment_digest,
                }
            ],
        }
    )
    client = AttachmentAwareRecordingClient()
    client.response_factory = lambda schema, role: AnalystResponse(
            complete=True,
            candidate_findings=[candidate],
        )

    with pytest.raises(ModelResponseValidationError, match="undelivered evidence ID"):
        await ReportAnalyst(client, model="gpt-report").analyze(
            corpus,
            resolved_evidence=resolved,
        )

    assert client.measured_sizes == [
        _REPORT_REQUEST_MAX_BYTES + 1,
        _REPORT_REQUEST_MAX_BYTES - 1,
    ]
    assert client.messages[1].attachments == ()


@pytest.mark.asyncio
async def test_over_budget_resolved_entries_are_bounded_before_model_client() -> None:
    class EntryAwareRecordingClient(RecordingClient):
        def request_size(
            self,
            schema: type[Any],
            messages: Sequence[Any],
            *,
            model: str,
            role: ModelRole,
        ) -> int:
            del schema, model, role
            entry_count = len(json.loads(messages[1].content)["resolved_evidence"])
            return (
                _REPORT_REQUEST_MAX_BYTES - 1
                if entry_count <= 1
                else _REPORT_REQUEST_MAX_BYTES + entry_count
            )

    entries = tuple(
        EvidenceEntry(
            ref=EvidenceRef(
                f"event:run-a:{index}", "event", "run-a", replay_sequence=index
            ),
            evidence_class=EvidenceClass.DETERMINISTIC_FACT,
            summary=f"Recorded event {index}.",
            payload={"sequence": index},
        )
        for index in range(1, 4)
    )
    corpus = EvidenceCorpus(output_root=Path.cwd(), entries=entries)
    resolved = EvidenceResolver().resolve(
        corpus,
        [entry.ref.evidence_id for entry in entries],
        max_entries=3,
        max_attachment_bytes=1024,
    )
    candidate = CandidateFinding.model_validate(
        {
            **_finding_payload(),
            "evidence_refs": [
                {
                    "evidence_id": "event:run-a:3",
                    "kind": "event",
                    "run_id": "run-a",
                    "replay_sequence": 3,
                }
            ],
        }
    )
    client = EntryAwareRecordingClient(
        lambda schema, role: AnalystResponse(
            complete=True,
            candidate_findings=[candidate],
        )
    )

    with pytest.raises(ModelResponseValidationError, match="undelivered evidence ID"):
        await ReportAnalyst(client, model="gpt-report").analyze(
            corpus,
            resolved_evidence=resolved,
        )

    payload = json.loads(client.messages[1].content)
    assert len(payload["resolved_evidence"]) == 1
    assert payload["resolved_evidence_context"]["deferred_count"] == 2


@pytest.mark.asyncio
async def test_large_metric_continuation_uses_compact_bounded_context() -> None:
    core_kinds = (
        "scenario",
        "persona",
        "goal",
        "expectation",
        "verification",
    )
    core_entries = tuple(
        EvidenceEntry(
            ref=EvidenceRef(
                f"{kind}:run-a" if index < 5 else f"event:run-a:{index}",
                kind if index < 5 else "event",
                "run-a",
                replay_sequence=index if index >= 5 else None,
            ),
            evidence_class=EvidenceClass.DETERMINISTIC_FACT,
            summary=f"Recorded {kind}.",
            payload={"value": kind},
        )
        for index, kind in enumerate(core_kinds + ("event",) * 5)
    )
    metric_entries = tuple(
        EvidenceEntry(
            ref=EvidenceRef(
                f"metric:run-{run}:{name}",
                "metric",
                f"run-{run}",
                metric_id=name,
            ),
            evidence_class=EvidenceClass.MODEL_ESTIMATE,
            summary=f"Metric {name} recorded for run {run}.",
            payload={
                "name": name,
                "value": run / 10,
                "evidence_class": EvidenceClass.MODEL_ESTIMATE.value,
                "source_evidence_ids": (f"event:run-{run}:1",),
                "provenance": {
                    "provider_id": "heuristic",
                    "provider_version": "heuristic-v1",
                    "model_id": "heuristic-model",
                    "model_version": "v1",
                },
            },
        )
        for run in range(1, 7)
        for name in (
            "target-discovery-rank",
            "inspected-elements",
            "inspected-regions",
            "scrolls",
            "wrong-actions",
            "backtracks",
            "verified-completion",
            "claimed-completion",
            "false-success",
            "target-prominence",
            "target-below-fold",
            "unexpected-hierarchy",
            "ambiguous-target",
            "navigation-depth",
            "feedback-observed",
            "recovery-actions",
            "inspection-cost",
            "region-cost",
            "scroll-cost",
            "wrong-action-cost",
            "backtrack-cost",
            "uncertainty-cost",
            "abandonment-penalty",
            "discovery-cost",
            "outcome",
        )
    )
    corpus = EvidenceCorpus(
        output_root=Path.cwd(), entries=core_entries + metric_entries
    )
    resolved = EvidenceResolver().resolve(
        corpus,
        [entry.ref.evidence_id for entry in corpus.entries],
        max_entries=160,
        max_attachment_bytes=1024,
    )
    client = RecordingClient()

    await ReportAnalyst(client, model="gpt-report").analyze(
        corpus,
        resolved_evidence=resolved,
    )

    payload = json.loads(client.calls[0][1][1].content)
    context_entries = payload["resolved_evidence"]
    metric_groups = [
        item
        for item in context_entries
        if item.get("representation") == "metric-group-v1"
    ]
    assert len(context_entries) <= _REPORT_MODEL_CONTEXT_MAX_ENTRIES
    assert len(metric_groups) == 1
    assert len(metric_groups[0]["entries"]) == 150
    assert {item["evidence_id"] for item in metric_groups[0]["entries"]} == {
        entry.ref.evidence_id for entry in metric_entries
    }
    assert payload["resolved_evidence_context"] == {
        "requested_count": 160,
        "included_count": 160,
        "deferred_count": 0,
        "model_context_item_count": len(context_entries),
        "representation": "compact-metric-group-v1",
        "visual_attachments_deferred": False,
        "instruction": (
            "Context is transport-bounded; request deferred evidence "
            "again when it is required for a complete conclusion."
        ),
    }
    assert all("private" not in repr(item) for item in context_entries)
    serialized_size = len(client.calls[0][1][1].content.encode("utf-8"))
    assert (
        serialized_size + _REPORT_REQUEST_RESERVED_OVERHEAD_BYTES
        <= _REPORT_REQUEST_MAX_BYTES
    )


@pytest.mark.asyncio
async def test_role_prompts_state_distinct_review_responsibilities() -> None:
    client = RecordingClient()
    manifest = _manifest()
    principles = ux_principles()

    await ReportAnalyst(client, model="gpt-report").analyze(manifest, principles)
    await EvidenceAuditor(client, model="gpt-report").audit(manifest, principles)
    await PatternReviewer(client, model="gpt-report").review(manifest, principles)
    await ReportAdjudicator(client, model="gpt-report").adjudicate(manifest, principles)

    prompts = {call[3]: call[1][0].content.lower() for call in client.calls}
    assert "issues" in prompts[ModelRole.REPORT_ANALYST]
    assert "root causes" in prompts[ModelRole.REPORT_ANALYST]
    assert "factual support" in prompts[ModelRole.REPORT_EVIDENCE_AUDITOR]
    assert "visual interpretation" in prompts[ModelRole.REPORT_EVIDENCE_AUDITOR]
    assert "citation accuracy" in prompts[ModelRole.REPORT_EVIDENCE_AUDITOR]
    assert "contradictions" in prompts[ModelRole.REPORT_EVIDENCE_AUDITOR]
    assert "recurrence" in prompts[ModelRole.REPORT_PATTERN_REVIEWER]
    assert "affected surfaces" in prompts[ModelRole.REPORT_PATTERN_REVIEWER]
    assert "counterexamples" in prompts[ModelRole.REPORT_PATTERN_REVIEWER]
    assert "shared causes" in prompts[ModelRole.REPORT_PATTERN_REVIEWER]
    assert "fix leverage" in prompts[ModelRole.REPORT_PATTERN_REVIEWER]
    assert "resolves objections" in prompts[ModelRole.REPORT_ADJUDICATOR]
    assert "plain-language final findings" in prompts[ModelRole.REPORT_ADJUDICATOR]
    for prompt in prompts.values():
        assert "principles are not evidence" in prompt
        assert "cannot determine severity" in prompt


@pytest.mark.asyncio
async def test_role_calls_use_fresh_isolated_message_tuples() -> None:
    client = RecordingClient()
    manifest = _manifest()
    principles = ux_principles()
    corpus = _corpus(Path.cwd())
    resolved = EvidenceResolver().resolve(
        corpus,
        [EVIDENCE_ID],
        max_entries=2,
        max_attachment_bytes=1024,
    )

    analyst = ReportAnalyst(client, model="gpt-report")
    first = await analyst.analyze(manifest, principles)
    await analyst.analyze(
        manifest,
        principles,
        resolved_evidence=resolved,
        previous_output=first,
    )
    await EvidenceAuditor(client, model="gpt-report").audit(
        manifest,
        principles,
        candidate_findings=first.candidate_findings,
        resolved_evidence=resolved,
    )

    assert len(client.calls) == 3
    assert client.calls[0][1] is not client.calls[1][1]
    assert client.calls[1][1] is not client.calls[2][1]
    assert [call[3] for call in client.calls] == [
        ModelRole.REPORT_ANALYST,
        ModelRole.REPORT_ANALYST,
        ModelRole.REPORT_EVIDENCE_AUDITOR,
    ]
    assert all(
        tuple(message.role for message in call[1]) == ("system", "user")
        for call in client.calls
    )


def test_investigative_response_requires_retrieval_request_when_incomplete() -> None:
    with pytest.raises(ValidationError, match="evidence_requests"):
        AnalystResponse(complete=False, evidence_requests=[])

    response = AnalystResponse(
        complete=False,
        evidence_requests=[EVIDENCE_ID],
    )
    assert response.complete is False
    assert response.evidence_requests == [EVIDENCE_ID]


def test_role_response_schemas_require_role_outputs_and_validate_domains() -> None:
    candidate = CandidateFinding.model_validate(_finding_payload())
    objection = TypedObjection(
        objection_id="objection-1",
        finding_id=candidate.finding_id,
        objection_type="factual-support",
        severity=ObjectionSeverity.BLOCKING,
        message="The cited event does not establish the stated cause.",
        evidence_refs=[
            {
                "evidence_id": EVIDENCE_ID,
                "kind": "event",
                "run_id": "run-a",
                "replay_sequence": 1,
            }
        ],
    )
    resolution = ObjectionResolution(
        objection_id=objection.objection_id,
        finding_id=candidate.finding_id,
        resolved=True,
        resolution="The event and follow-up evidence support a narrower causal claim.",
        evidence_refs=objection.evidence_refs,
    )

    analyst = AnalystResponse(complete=True, candidate_findings=[candidate])
    auditor = EvidenceAuditResponse(complete=True, objections=[objection])
    pattern = PatternReviewResponse(complete=True, objections=[objection])
    adjudicator = AdjudicationResponse(
        complete=True,
        final_findings=[candidate],
        objection_resolutions=[resolution],
    )

    assert analyst.candidate_findings[0].severity is FindingSeverity.MEDIUM
    assert auditor.objections[0].objection_type == "factual-support"
    assert pattern.objections[0].finding_id == candidate.finding_id
    assert adjudicator.final_findings[0].finding_id == candidate.finding_id
    assert adjudicator.objection_resolutions[0].resolved is True


@pytest.mark.asyncio
async def test_path_deviation_is_tolerated_when_outcome_evidence_is_valid() -> None:
    class AlternatePathClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del role
            if schema is AnalystResponse:
                return schema.model_validate(
                    {
                        "complete": True,
                        "evidence_requests": [],
                        "candidate_findings": [_finding_payload()],
                    }
                )
            return RecordingClient._default_response(schema, ModelRole.REPORT_ANALYST)

    corpus = _corpus(Path.cwd())
    resolved = EvidenceResolver().resolve(
        corpus,
        [EVIDENCE_ID],
        max_entries=1,
        max_attachment_bytes=1024,
    )
    response = await ReportAnalyst(AlternatePathClient(), model="gpt-report").analyze(
        corpus,
        ux_principles(),
        resolved_evidence=resolved,
    )

    assert response.candidate_findings[0].evidence_refs[0].evidence_id == EVIDENCE_ID


@pytest.mark.asyncio
async def test_complete_finding_cannot_cite_manifest_only_evidence() -> None:
    class ManifestOnlyClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del role
            return schema.model_validate(
                {
                    "complete": True,
                    "candidate_findings": [_finding_payload()],
                }
            )

    with pytest.raises(ModelResponseValidationError, match="undelivered evidence ID"):
        await ReportAnalyst(ManifestOnlyClient(), model="gpt-report").analyze(
            _manifest(), ux_principles()
        )


@pytest.mark.asyncio
async def test_role_response_rejects_oversized_lists() -> None:
    class OversizedListClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del schema, role
            return AnalystResponse.model_construct(
                complete=False,
                evidence_requests=[EVIDENCE_ID] * 129,
                candidate_findings=[],
            )

    with pytest.raises(ModelResponseValidationError, match="bounded output limits"):
        await ReportAnalyst(OversizedListClient(), model="gpt-report").analyze(
            _manifest(), ux_principles()
        )


@pytest.mark.asyncio
async def test_role_response_rejects_oversized_strings() -> None:
    oversized = "x" * 8_193

    class OversizedStringClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del role
            payload = _finding_payload()
            payload["title"] = oversized
            return schema.model_validate(
                {
                    "complete": True,
                    "candidate_findings": [payload],
                }
            )

    corpus = _corpus(Path.cwd())
    resolved = EvidenceResolver().resolve(
        corpus,
        [EVIDENCE_ID],
        max_entries=1,
        max_attachment_bytes=1024,
    )
    with pytest.raises(ModelResponseValidationError, match="bounded output limits"):
        await ReportAnalyst(OversizedStringClient(), model="gpt-report").analyze(
            corpus,
            ux_principles(),
            resolved_evidence=resolved,
        )


@pytest.mark.asyncio
async def test_principles_cannot_be_used_as_evidence() -> None:
    class PrincipleOnlyClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del role
            if schema is AnalystResponse:
                candidate = CandidateFinding.model_construct(
                    **_finding_payload(principle_only=True)
                )
                return AnalystResponse.model_construct(
                    complete=True,
                    evidence_requests=[],
                    candidate_findings=[candidate],
                )
            return RecordingClient._default_response(schema, ModelRole.REPORT_ANALYST)

    with pytest.raises(
        ModelResponseValidationError, match="principles are not evidence"
    ):
        await ReportAnalyst(PrincipleOnlyClient(), model="gpt-report").analyze(
            _manifest(), ux_principles()
        )


@pytest.mark.asyncio
async def test_retrieval_requests_must_name_known_evidence_ids() -> None:
    class RetrievalClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del role
            if schema is AnalystResponse:
                return schema.model_validate(
                    {
                        "complete": False,
                        "evidence_requests": ["event:run-a:unknown"],
                    }
                )
            return RecordingClient._default_response(schema, ModelRole.REPORT_ANALYST)

    with pytest.raises(ModelResponseValidationError, match="unknown evidence ID"):
        await ReportAnalyst(RetrievalClient(), model="gpt-report").analyze(
            _manifest(), ux_principles()
        )


@pytest.mark.asyncio
async def test_empty_manifest_rejects_finding_evidence_reference() -> None:
    class EmptyManifestClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del role
            if schema is AnalystResponse:
                return schema.model_validate(
                    {
                        "complete": True,
                        "candidate_findings": [_finding_payload()],
                    }
                )
            return RecordingClient._default_response(schema, ModelRole.REPORT_ANALYST)

    manifest = _manifest()
    manifest["entries"] = []

    with pytest.raises(ModelResponseValidationError, match="unknown evidence ID"):
        await ReportAnalyst(EmptyManifestClient(), model="gpt-report").analyze(
            manifest, ux_principles()
        )


@pytest.mark.asyncio
async def test_empty_manifest_rejects_objection_and_resolution_evidence_references() -> (
    None
):
    class EmptyManifestClient(RecordingClient):
        @staticmethod
        def _default_response(schema: type[Any], role: ModelRole) -> object:
            del role
            reference = {
                "evidence_id": EVIDENCE_ID,
                "kind": "event",
                "run_id": "run-a",
                "replay_sequence": 1,
            }
            if schema is EvidenceAuditResponse:
                return schema.model_validate(
                    {
                        "complete": True,
                        "objections": [
                            {
                                "objection_id": "objection-1",
                                "finding_id": "invite-control",
                                "severity": "material",
                                "message": "The evidence does not establish the claim.",
                                "evidence_refs": [reference],
                            }
                        ],
                    }
                )
            if schema is AdjudicationResponse:
                return schema.model_validate(
                    {
                        "complete": True,
                        "objection_resolutions": [
                            {
                                "objection_id": "objection-1",
                                "finding_id": "invite-control",
                                "resolved": True,
                                "resolution": "The evidence supports the claim.",
                                "evidence_refs": [reference],
                            }
                        ],
                    }
                )
            return RecordingClient._default_response(schema, ModelRole.REPORT_ANALYST)

    manifest = _manifest()
    manifest["entries"] = []
    client = EmptyManifestClient()

    with pytest.raises(ModelResponseValidationError, match="unknown evidence ID"):
        await EvidenceAuditor(client, model="gpt-report").audit(
            manifest, ux_principles()
        )

    with pytest.raises(ModelResponseValidationError, match="unknown evidence ID"):
        await ReportAdjudicator(client, model="gpt-report").adjudicate(
            manifest, ux_principles()
        )


def test_manifest_and_role_manifests_use_report_role_metadata() -> None:
    client = RecordingClient()

    assert (
        ReportAnalyst(client, model="gpt-report").manifest.role
        is ModelRole.REPORT_ANALYST
    )
    assert ReportAnalyst(client, model="gpt-report").manifest.prompt_version == (
        "report-analyst-v3"
    )
    assert (
        EvidenceAuditor(client, model="gpt-report").manifest.role
        is ModelRole.REPORT_EVIDENCE_AUDITOR
    )
    assert (
        PatternReviewer(client, model="gpt-report").manifest.role
        is ModelRole.REPORT_PATTERN_REVIEWER
    )
    assert (
        ReportAdjudicator(client, model="gpt-report").manifest.role
        is ModelRole.REPORT_ADJUDICATOR
    )
    assert ReportAnalystResponse is AnalystResponse
    assert ReportEvidenceAuditorResponse is EvidenceAuditResponse
    assert ReportPatternReviewerResponse is PatternReviewResponse
