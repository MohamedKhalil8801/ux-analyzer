from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from ux_analyzer.domain.findings import (
    EvidenceClass,
    FindingSeverity,
    Reproducibility,
    UnsupportedHumanClaimError,
)
from ux_analyzer.domain.synthesis import (
    EvidenceRef,
    ObjectionSeverity,
    SynthesisAttempt,
    SynthesisFinding,
    SynthesisObjection,
    SynthesisStatus,
)


def _finding(**overrides: object) -> SynthesisFinding:
    values: dict[str, object] = {
        "finding_id": "finding-1",
        "title": "Navigation labels do not match user goals",
        "issue": "People look in several unrelated areas before finding settings.",
        "impact": "Important tasks take longer and are easy to abandon.",
        "root_cause": "Labels describe internal structure instead of the user's task.",
        "fixes": ("Rename the entry point around the user goal.",),
        "severity": FindingSeverity.HIGH,
        "confidence": 0.91,
        "evidence_refs": (EvidenceRef("event:run-a:18", "event", "run-a"),),
    }
    values.update(overrides)
    return SynthesisFinding(**values)  # type: ignore[arg-type]


def test_synthesis_finding_requires_resolvable_evidence_and_plain_language() -> None:
    finding = SynthesisFinding(
        finding_id="root-navigation-labels",
        title="Navigation labels do not match user goals",
        issue="People look in several unrelated areas before finding security settings.",
        impact="Important account protection tasks take longer and are easy to abandon.",
        root_cause="Labels describe internal product structure instead of the user's task.",
        fixes=(
            "Rename the entry point around the user goal and reuse it consistently.",
        ),
        severity="high",
        confidence=0.91,
        evidence_refs=(EvidenceRef("event:run-a:18", "event", "run-a"),),
    )

    assert finding.confidence == pytest.approx(0.91)
    assert finding.severity is FindingSeverity.HIGH
    assert finding.evidence_class is EvidenceClass.MODEL_ESTIMATE
    assert finding.reproducibility is Reproducibility.MODEL_DEPENDENT


def test_evidence_ref_preserves_all_optional_resolution_metadata() -> None:
    reference = EvidenceRef(
        evidence_id="screenshot:run-a:abc123",
        kind="screenshot",
        run_id="run-a",
        viewport_id="viewport-1",
        element_id="element-7",
        event_id="event-18",
        metric_id="task-seconds",
        artifact_path="runs/run-a/screenshots/abc123.png",
        replay_sequence=18,
        sha256="a" * 64,
    )

    assert reference.viewport_id == "viewport-1"
    assert reference.element_id == "element-7"
    assert reference.event_id == "event-18"
    assert reference.metric_id == "task-seconds"
    assert reference.artifact_path.endswith("abc123.png")
    assert reference.replay_sequence == 18
    assert reference.sha256 == "a" * 64


def test_synthesis_finding_normalizes_collections_and_existing_contracts() -> None:
    finding = _finding(
        fixes=["Fix the label."],
        affected_surfaces=["settings", "onboarding"],
        principles=["mental-models"],
        counterevidence=["One participant used the direct route."],
        limitations=["Fixture has one locale."],
        evidence_class="deterministic-fact",
        reproducibility="seeded",
        reviewer_state="accepted",
    )

    assert finding.fixes == ("Fix the label.",)
    assert finding.affected_surfaces == ("settings", "onboarding")
    assert finding.principles == ("mental-models",)
    assert finding.counterevidence == ("One participant used the direct route.",)
    assert finding.limitations == ("Fixture has one locale.",)
    assert finding.evidence_class is EvidenceClass.DETERMINISTIC_FACT
    assert finding.reproducibility is Reproducibility.SEEDED
    assert finding.reviewer_state == "accepted"


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_synthesis_finding_rejects_confidence_outside_unit_interval(
    confidence: float,
) -> None:
    with pytest.raises(ValueError, match="confidence"):
        _finding(confidence=confidence)


def test_synthesis_finding_rejects_empty_or_duplicate_evidence_ids() -> None:
    reference = EvidenceRef("event:run-a:18", "event", "run-a")
    duplicate_reference = EvidenceRef("event:run-a:18", "event", "run-a")
    with pytest.raises(ValueError, match="duplicate evidence"):
        _finding(evidence_refs=(reference, duplicate_reference))

    with pytest.raises(ValueError, match="evidence ID"):
        _finding(evidence_refs=())

    with pytest.raises(ValueError, match="evidence ID"):
        EvidenceRef("", "event", "run-a")


def test_synthesis_finding_rejects_unsupported_human_claim() -> None:
    with pytest.raises(UnsupportedHumanClaimError, match="unsupported human claim"):
        _finding(evidence_class=EvidenceClass.UNSUPPORTED_HUMAN_CLAIM)


def test_synthesis_finding_rejects_missing_fix_and_plain_language_fields() -> None:
    with pytest.raises(ValueError, match="fix"):
        _finding(fixes=())
    with pytest.raises(ValueError, match="issue"):
        _finding(issue=" ")


def test_synthesis_objection_normalizes_severity_and_reviewer_state() -> None:
    objection = SynthesisObjection(
        objection_id="objection-1",
        finding_id="finding-1",
        severity="blocking",
        message="The verifier outcome contradicts the proposed claim.",
        evidence_refs=[EvidenceRef("event:run-a:19", "event", "run-a")],
        reviewer_role="evidence-auditor",
        resolved=False,
    )

    assert objection.severity is ObjectionSeverity.BLOCKING
    assert objection.evidence_refs[0].evidence_id == "event:run-a:19"
    assert objection.evidence_refs == (EvidenceRef("event:run-a:19", "event", "run-a"),)
    assert objection.reviewer_role == "evidence-auditor"
    assert objection.resolved is False


def test_synthesis_attempt_supports_all_immutable_statuses_and_artifact_metadata() -> (
    None
):
    finding = _finding()
    objection = SynthesisObjection(
        objection_id="objection-1",
        finding_id=finding.finding_id,
        severity=ObjectionSeverity.EDITORIAL,
        message="Use shorter wording.",
        resolved=True,
        resolution="Wording updated.",
    )
    attempt = SynthesisAttempt(
        attempt_id="attempt-1",
        status="accepted",
        corpus_digest="corpus-digest",
        expectation_digest="expectation-digest",
        principle_pack_digest="principles-digest",
        model_manifest={"report-analyst": "gpt-report"},
        role_manifest={"report-analyst": "fresh-context"},
        prompt_version="prompt-v1",
        schema_version="synthesis-v1",
        retrieval_log=[{"role": "report-analyst", "round": 1}],
        usage={"input_tokens": 12, "output_tokens": 9},
        candidate_findings=[finding],
        objections=[objection],
        rejected_findings=[],
        findings=[finding],
        limitations=["Fixture evidence only."],
        fallback_available=True,
    )

    assert attempt.status is SynthesisStatus.ACCEPTED
    assert attempt.candidate_findings == (finding,)
    assert attempt.objections == (objection,)
    assert attempt.findings == (finding,)
    assert attempt.retrieval_log == ({"role": "report-analyst", "round": 1},)
    assert isinstance(attempt.model_manifest, MappingProxyType)
    assert isinstance(attempt.usage, MappingProxyType)
    with pytest.raises(TypeError):
        attempt.usage["input_tokens"] = 99  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        attempt.status = SynthesisStatus.REJECTED  # type: ignore[misc]

    assert {status.value for status in SynthesisStatus} == {
        "accepted",
        "rejected",
        "unavailable",
        "no-issues",
    }
    assert {severity.value for severity in ObjectionSeverity} == {
        "blocking",
        "material",
        "editorial",
    }


def test_synthesis_attempt_requires_non_empty_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        SynthesisAttempt(
            attempt_id="attempt-1",
            status=SynthesisStatus.UNAVAILABLE,
            schema_version="",
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_record"),
    [
        ("candidate_findings", {"finding_id": "mutable-record"}),
        ("objections", {"objection_id": "mutable-record"}),
        ("rejected_findings", {"finding_id": "mutable-record"}),
        ("findings", {"finding_id": "mutable-record"}),
    ],
)
def test_synthesis_attempt_rejects_non_domain_records(
    field_name: str, invalid_record: object
) -> None:
    values = {
        "attempt_id": "attempt-1",
        "status": SynthesisStatus.REJECTED,
        field_name: [invalid_record],
    }

    with pytest.raises(TypeError, match=field_name):
        SynthesisAttempt(**values)  # type: ignore[arg-type]


def test_synthesis_attempt_recursively_freezes_manifest_and_retrieval_data() -> None:
    model_manifest = {"report-analyst": {"models": ["gpt-report"]}}
    retrieval_entry = {"request": {"evidence_ids": ["event:run-a:18"]}}
    attempt = SynthesisAttempt(
        attempt_id="attempt-1",
        status=SynthesisStatus.UNAVAILABLE,
        model_manifest=model_manifest,
        retrieval_log=[retrieval_entry],
    )

    model_manifest["report-analyst"]["models"].append("changed")
    retrieval_entry["request"]["evidence_ids"].append("changed")

    frozen_model = attempt.model_manifest["report-analyst"]
    frozen_request = attempt.retrieval_log[0]["request"]
    assert isinstance(frozen_model, MappingProxyType)
    assert isinstance(frozen_request, MappingProxyType)
    assert frozen_model["models"] == ("gpt-report",)
    assert frozen_request["evidence_ids"] == ("event:run-a:18",)


def test_synthesis_attempt_rejects_unsupported_mutable_metadata_values() -> None:
    mutable_metadata = bytearray(b"mutable")

    with pytest.raises(TypeError, match="unsupported metadata value"):
        SynthesisAttempt(
            attempt_id="attempt-1",
            status=SynthesisStatus.UNAVAILABLE,
            model_manifest={"payload": mutable_metadata},
        )


@pytest.mark.parametrize("resolution", [None, " "])
def test_resolved_synthesis_objection_requires_non_empty_resolution(
    resolution: str | None,
) -> None:
    with pytest.raises(ValueError, match="resolution"):
        SynthesisObjection(
            objection_id="objection-1",
            finding_id="finding-1",
            severity=ObjectionSeverity.BLOCKING,
            message="The evidence does not support the proposed claim.",
            resolved=True,
            resolution=resolution,
        )


@pytest.mark.parametrize(
    "status",
    [
        SynthesisStatus.ACCEPTED,
        SynthesisStatus.NO_ISSUES,
        SynthesisStatus.REJECTED,
        SynthesisStatus.UNAVAILABLE,
    ],
)
def test_synthesis_attempt_converts_status_values(status: SynthesisStatus) -> None:
    attempt = SynthesisAttempt(attempt_id="attempt-1", status=status.value)

    assert attempt.status is status
