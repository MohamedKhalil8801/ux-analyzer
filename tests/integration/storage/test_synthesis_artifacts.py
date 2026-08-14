from __future__ import annotations

import ast
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from ux_analyzer.application.evidence_corpus import EvidenceCorpus, EvidenceEntry
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.synthesis import (
    EvidenceRef,
    ObjectionSeverity,
    SynthesisAttempt,
    SynthesisFinding,
    SynthesisObjection,
    SynthesisStatus,
)
from ux_analyzer.storage import synthesis_artifacts
from ux_analyzer.storage.synthesis_artifacts import (
    SynthesisArtifactError,
    SynthesisArtifactStore,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def _corpus(tmp_path: Path) -> EvidenceCorpus:
    return EvidenceCorpus(
        output_root=tmp_path,
        entries=(
            EvidenceEntry(
                ref=EvidenceRef("event:run-a:1", "event", "run-a", replay_sequence=1),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="The user opened the invite control.",
                payload={"sequence": 1, "succeeded": True},
            ),
            EvidenceEntry(
                ref=EvidenceRef("expectation:run-a", "expectation", "run-a"),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="A frozen expectation matched the run.",
                payload={
                    "matched": True,
                    "expectation_id": "invite-v1",
                    "desired_outcomes": ["A valid invitation is sent."],
                },
            ),
        ),
    )


def _expectation_digest(corpus: EvidenceCorpus) -> str:
    manifest = json.loads(corpus.to_json())
    payload = [
        entry["payload"]
        for entry in manifest["entries"]
        if entry["kind"] == "expectation"
    ]
    content = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _finding() -> SynthesisFinding:
    return SynthesisFinding(
        finding_id="invite-control",
        title="The invite control is hard to find",
        issue="The user searches outside the task area before finding the invite control.",
        impact="An important collaboration task takes longer to complete.",
        root_cause="The entry point is labeled around internal product structure.",
        fixes=("Label the entry point around the user's goal.",),
        severity="high",
        confidence=0.9,
        evidence_refs=(EvidenceRef("event:run-a:1", "event", "run-a"),),
        reviewer_state="accepted",
        severity_justification="The event records extra navigation on an important task.",
    )


def _attempt(
    corpus: EvidenceCorpus,
    *,
    sequence: int,
    status: SynthesisStatus = SynthesisStatus.ACCEPTED,
    findings: tuple[SynthesisFinding, ...] | None = None,
    candidate_findings: tuple[SynthesisFinding, ...] | None = None,
    rejected_findings: tuple[SynthesisFinding, ...] | None = None,
    objections: tuple[SynthesisObjection, ...] = (),
) -> SynthesisAttempt:
    if findings is None:
        findings = (_finding(),) if status is SynthesisStatus.ACCEPTED else ()
    if candidate_findings is None:
        candidate_findings = findings
    if rejected_findings is None:
        rejected_findings = ()
        if status is SynthesisStatus.REJECTED and candidate_findings:
            rejected_findings = tuple(
                replace(finding, reviewer_state="not-established")
                for finding in candidate_findings
            )
    return SynthesisAttempt(
        attempt_id=f"20260810T120000Z-{corpus.digest[:12]}-{sequence}",
        status=status,
        corpus_digest=corpus.digest,
        expectation_digest=_expectation_digest(corpus),
        principle_pack_digest=corpus.principle_pack_digest,
        model_manifest={"roles": {"report-analyst": {"model_id": "fixture"}}},
        role_manifest={
            "report-analyst": {
                "provider_id": "fixture",
                "model_id": "fixture",
                "prompt_version": "report-analyst-v1",
                "schema_version": "report-analyst-response-v1",
            }
        },
        prompt_version="report-synthesis-orchestrator-v1",
        schema_version="synthesis-v1",
        retrieval_log=(
            {
                "role": "report-analyst",
                "round": 1,
                "resolved_evidence_ids": ("event:run-a:1",),
            },
        ),
        usage={"role_calls": 1},
        candidate_findings=candidate_findings,
        objections=objections,
        rejected_findings=rejected_findings,
        findings=findings,
        limitations=("Fixture evidence only.",),
        fallback_available=True,
        created_at="2026-08-10T12:00:00+00:00",
    )


def _blocking_objection(finding_id: str) -> SynthesisObjection:
    evidence_ref = EvidenceRef(
        "event:run-a:1",
        "event",
        "run-a",
        replay_sequence=1,
    )
    return SynthesisObjection(
        objection_id="blocking-objection",
        finding_id=finding_id,
        severity=ObjectionSeverity.BLOCKING,
        message="The published claim remains contradicted by recorded evidence.",
        evidence_refs=(evidence_ref,),
        resolved=False,
    )


def _resolved_blocking_objection(finding_id: str) -> SynthesisObjection:
    objection = _blocking_objection(finding_id)
    return replace(
        objection,
        resolved=True,
        resolution="The adjudicator resolved the contradiction from recorded evidence.",
        resolved_by_role="report-adjudicator",
        resolution_evidence_refs=objection.evidence_refs,
    )


def test_write_attempt_publishes_canonical_layout_and_round_trips(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1, findings=(_finding(),))
    store = SynthesisArtifactStore(tmp_path)

    attempt_path = store.write_attempt(attempt, corpus)

    assert attempt_path == tmp_path / "synthesis" / "attempts" / attempt.attempt_id
    assert (attempt_path / "synthesis.json").is_file()
    assert (attempt_path / "corpus-manifest.json").is_file()
    assert (tmp_path / "synthesis" / "index.json").is_file()

    synthesis_bytes = (attempt_path / "synthesis.json").read_bytes()
    corpus_bytes = (attempt_path / "corpus-manifest.json").read_bytes()
    index_bytes = (tmp_path / "synthesis" / "index.json").read_bytes()
    assert synthesis_bytes == _canonical_bytes(json.loads(synthesis_bytes))
    assert corpus_bytes == corpus.to_json().encode("ascii")
    assert index_bytes == _canonical_bytes(json.loads(index_bytes))

    synthesis_value = json.loads(synthesis_bytes)
    assert {
        "corpus",
        "expectation",
        "principle_pack",
        "prompt",
        "schema",
    } <= set(synthesis_value["digests"])
    assert {
        "role_manifest",
        "retrieval_log",
        "usage",
        "candidates",
        "objections",
        "rejected_findings",
        "final_findings",
        "status",
        "limitations",
    } <= set(synthesis_value)
    assert synthesis_value["status"] == "accepted"
    assert synthesis_value["final_findings"][0]["finding_id"] == "invite-control"

    assert store.attempts == (attempt,)
    assert store.accepted_attempt == attempt


def test_write_attempt_rejects_accepted_unreviewed_finding(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    pending = replace(_finding(), reviewer_state="pending")
    attempt = _attempt(corpus, sequence=1, findings=(pending,))
    store = SynthesisArtifactStore(tmp_path)

    with pytest.raises(SynthesisArtifactError, match="reviewer|publish"):
        store.write_attempt(attempt, corpus)

    assert store.accepted_attempt is None


def test_write_attempt_rejects_accepted_without_final_findings(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(
        corpus,
        sequence=1,
        findings=(),
        candidate_findings=(),
    )

    with pytest.raises(SynthesisArtifactError, match="accepted|final"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


def test_write_attempt_rejects_accepted_finding_with_unresolved_blocker(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    finding = _finding()
    attempt = _attempt(
        corpus,
        sequence=1,
        findings=(finding,),
        objections=(_blocking_objection(finding.finding_id),),
    )

    with pytest.raises(SynthesisArtifactError, match="blocking|publish"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


def test_resolved_blocking_objection_round_trips_with_adjudicator_provenance(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    finding = _finding()
    objection = _resolved_blocking_objection(finding.finding_id)
    attempt = _attempt(
        corpus,
        sequence=1,
        findings=(finding,),
        objections=(objection,),
    )
    store = SynthesisArtifactStore(tmp_path)

    store.write_attempt(attempt, corpus)

    assert store.accepted_attempt == attempt
    persisted = json.loads(
        next((tmp_path / "synthesis" / "attempts").iterdir())
        .joinpath("synthesis.json")
        .read_text(encoding="ascii")
    )["objections"][0]
    assert persisted["resolved_by_role"] == "report-adjudicator"
    assert persisted["resolution_evidence_refs"]


@pytest.mark.parametrize("reference_field", ("reviewer", "resolution"))
@pytest.mark.parametrize("invalid_kind", ("nonexistent", "mismatched", "boolean"))
def test_write_attempt_rejects_objection_reference_outside_exact_corpus(
    tmp_path: Path,
    reference_field: str,
    invalid_kind: str,
) -> None:
    corpus = _corpus(tmp_path)
    finding = _finding()
    objection = _resolved_blocking_objection(finding.finding_id)
    valid_ref = objection.evidence_refs[0]
    with pytest.raises(
        (SynthesisArtifactError, TypeError),
        match="objection|evidence|corpus|replay_sequence",
    ):
        if invalid_kind == "nonexistent":
            invalid_ref = replace(
                valid_ref,
                evidence_id="event:run-a:999",
                replay_sequence=999,
            )
        elif invalid_kind == "boolean":
            invalid_ref = replace(valid_ref, replay_sequence=True)
        else:
            invalid_ref = replace(valid_ref, replay_sequence=2)
        objection = (
            replace(objection, evidence_refs=(invalid_ref,))
            if reference_field == "reviewer"
            else replace(objection, resolution_evidence_refs=(invalid_ref,))
        )
        attempt = _attempt(
            corpus,
            sequence=1,
            findings=(finding,),
            objections=(objection,),
        )
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


@pytest.mark.parametrize("invalid_kind", ("missing-role", "wrong-role", "no-evidence"))
def test_write_attempt_rejects_forged_blocking_resolution(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    corpus = _corpus(tmp_path)
    finding = _finding()
    objection = _resolved_blocking_objection(finding.finding_id)
    if invalid_kind == "missing-role":
        objection = replace(objection, resolved_by_role=None)
    elif invalid_kind == "wrong-role":
        objection = replace(objection, resolved_by_role="report-evidence-auditor")
    else:
        objection = replace(objection, resolution_evidence_refs=())
    attempt = _attempt(
        corpus,
        sequence=1,
        findings=(finding,),
        objections=(objection,),
    )

    with pytest.raises(SynthesisArtifactError, match="adjudicator|resolution evidence"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


@pytest.mark.parametrize("invalid_kind", ("duplicate", "orphan"))
def test_write_attempt_rejects_invalid_objection_identity(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    corpus = _corpus(tmp_path)
    finding = _finding()
    objection = replace(
        _blocking_objection(finding.finding_id),
        severity=ObjectionSeverity.MATERIAL,
    )
    objections = (
        (objection, objection)
        if invalid_kind == "duplicate"
        else (replace(objection, finding_id="unknown-finding"),)
    )
    attempt = _attempt(
        corpus,
        sequence=1,
        findings=(finding,),
        objections=objections,
    )

    with pytest.raises(SynthesisArtifactError, match="objection|candidate"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


@pytest.mark.parametrize("invalid_kind", ("changed-claim", "dropped-evidence"))
def test_write_attempt_rejects_final_that_does_not_preserve_candidate(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    corpus = _corpus(tmp_path)
    event_ref = EvidenceRef("event:run-a:1", "event", "run-a")
    expectation_ref = EvidenceRef("expectation:run-a", "expectation", "run-a")
    candidate = replace(
        _finding(),
        reviewer_state="candidate",
        evidence_refs=(event_ref, expectation_ref),
    )
    final = replace(candidate, reviewer_state="accepted")
    if invalid_kind == "changed-claim":
        final = replace(final, issue="A different issue replaced the reviewed claim.")
    else:
        final = replace(final, evidence_refs=(event_ref,))
    attempt = _attempt(
        corpus,
        sequence=1,
        findings=(final,),
        candidate_findings=(candidate,),
    )

    with pytest.raises(SynthesisArtifactError, match="reviewed|candidate|evidence"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


@pytest.mark.parametrize(
    "status",
    (SynthesisStatus.NO_ISSUES, SynthesisStatus.REJECTED, SynthesisStatus.UNAVAILABLE),
)
def test_write_attempt_rejects_status_with_promoted_final_finding(
    tmp_path: Path,
    status: SynthesisStatus,
) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1, status=status, findings=(_finding(),))

    with pytest.raises(SynthesisArtifactError, match="finding|publish"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


@pytest.mark.parametrize("invalid_field", ("candidates", "rejected", "objections"))
def test_write_attempt_rejects_no_issues_with_review_state(
    tmp_path: Path,
    invalid_field: str,
) -> None:
    corpus = _corpus(tmp_path)
    finding = _finding()
    values: dict[str, object] = {
        "candidate_findings": (),
        "rejected_findings": (),
        "objections": (),
    }
    if invalid_field == "candidates":
        values["candidate_findings"] = (finding,)
    elif invalid_field == "rejected":
        values["rejected_findings"] = (
            replace(finding, reviewer_state="not-established"),
        )
    else:
        values["objections"] = (_blocking_objection(finding.finding_id),)
    attempt = _attempt(
        corpus,
        sequence=1,
        status=SynthesisStatus.NO_ISSUES,
        **values,
    )

    with pytest.raises(SynthesisArtifactError, match="no-issues"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


def test_write_attempt_rejects_duplicate_final_finding_ids(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    finding = _finding()
    attempt = _attempt(
        corpus,
        sequence=1,
        findings=(finding, finding),
        candidate_findings=(finding,),
    )

    with pytest.raises(SynthesisArtifactError, match="duplicate|unique"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


def test_write_attempt_rejects_final_finding_absent_from_candidates(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(
        corpus,
        sequence=1,
        findings=(_finding(),),
        candidate_findings=(),
    )

    with pytest.raises(SynthesisArtifactError, match="candidate"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


def test_write_attempt_rejects_undisposed_candidate(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    published = _finding()
    omitted = replace(published, finding_id="omitted-candidate")
    attempt = _attempt(
        corpus,
        sequence=1,
        findings=(published,),
        candidate_findings=(published, omitted),
    )

    with pytest.raises(SynthesisArtifactError, match="disposed|disposition"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


@pytest.mark.parametrize(
    "status",
    (
        SynthesisStatus.ACCEPTED,
        SynthesisStatus.NO_ISSUES,
        SynthesisStatus.REJECTED,
        SynthesisStatus.UNAVAILABLE,
    ),
)
def test_valid_publication_states_round_trip(
    tmp_path: Path,
    status: SynthesisStatus,
) -> None:
    corpus = _corpus(tmp_path)
    candidate = replace(_finding(), reviewer_state="candidate")
    findings: tuple[SynthesisFinding, ...] = ()
    candidates: tuple[SynthesisFinding, ...] = ()
    rejected: tuple[SynthesisFinding, ...] = ()
    if status is SynthesisStatus.ACCEPTED:
        candidates = (candidate,)
        findings = (
            replace(
                candidate,
                title="Invite action is difficult to locate",
                fixes=("Expose an Invite teammate action in the team area.",),
                reviewer_state="accepted",
            ),
        )
    elif status is SynthesisStatus.REJECTED:
        candidates = (candidate,)
        rejected = (replace(candidate, reviewer_state="not-established"),)
    attempt = _attempt(
        corpus,
        sequence=1,
        status=status,
        findings=findings,
        candidate_findings=candidates,
        rejected_findings=rejected,
    )
    store = SynthesisArtifactStore(tmp_path)

    store.write_attempt(attempt, corpus)

    assert store.attempts == (attempt,)
    assert store.accepted_attempt == (
        attempt
        if status in {SynthesisStatus.ACCEPTED, SynthesisStatus.NO_ISSUES}
        else None
    )


def test_attempts_are_retained_and_rejected_attempts_remain_readable(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    accepted = _attempt(corpus, sequence=1)
    rejected = _attempt(corpus, sequence=2, status=SynthesisStatus.REJECTED)
    store = SynthesisArtifactStore(tmp_path)

    store.write_attempt(accepted, corpus)
    store.write_attempt(rejected, corpus)

    assert {item.attempt_id for item in store.attempts} == {
        accepted.attempt_id,
        rejected.attempt_id,
    }
    assert store.accepted_attempt == accepted
    assert (
        next(
            item for item in store.attempts if item.attempt_id == rejected.attempt_id
        ).status
        is SynthesisStatus.REJECTED
    )


def test_select_accepted_only_moves_pointer_to_accepted_status(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    first = _attempt(corpus, sequence=1)
    second = _attempt(corpus, sequence=2, status=SynthesisStatus.NO_ISSUES)
    rejected = _attempt(corpus, sequence=3, status=SynthesisStatus.REJECTED)
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(first, corpus)
    store.write_attempt(second, corpus)
    store.write_attempt(rejected, corpus)

    assert store.accepted_attempt == second
    assert store.select_accepted(first.attempt_id) == first
    assert store.accepted_attempt == first
    with pytest.raises(SynthesisArtifactError, match="accepted|no-issues"):
        store.select_accepted(rejected.attempt_id)


def test_incomplete_staging_and_index_temporary_files_are_ignored(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1)
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(attempt, corpus)

    staging = tmp_path / "synthesis" / "attempts" / ".staging-interrupted"
    staging.mkdir(parents=True)
    (staging / "synthesis.json").write_text("{", encoding="utf-8")
    (tmp_path / "synthesis" / ".index.json.tmp").write_text(
        "incomplete", encoding="utf-8"
    )

    reopened = SynthesisArtifactStore(tmp_path)
    assert reopened.attempts == (attempt,)
    assert reopened.accepted_attempt == attempt


def test_read_only_missing_root_does_not_create_synthesis_layout(
    tmp_path: Path,
) -> None:
    store = SynthesisArtifactStore(tmp_path)

    assert store.attempts == ()
    assert store.accepted_attempt is None
    assert store.report_attempt is None
    assert not store.synthesis_root.exists()


@pytest.mark.parametrize("read_boundary", ("attempts", "accepted", "report"))
def test_lockless_reader_does_not_create_publication_lock(
    tmp_path: Path,
    read_boundary: str,
) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1)
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(attempt, corpus)
    lock_path = store.synthesis_root / ".publication.lock"
    lock_path.unlink()

    if read_boundary == "attempts":
        assert store.attempts == (attempt,)
    elif read_boundary == "accepted":
        assert store.accepted_attempt == attempt
    else:
        assert store.report_attempt == attempt

    assert not lock_path.exists()


@pytest.mark.parametrize("read_boundary", ("attempts", "accepted", "report"))
def test_lockless_reader_retries_when_state_advances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    read_boundary: str,
) -> None:
    corpus = _corpus(tmp_path)
    first = _attempt(corpus, sequence=1)
    second = _attempt(corpus, sequence=2)
    writer = SynthesisArtifactStore(tmp_path)
    writer.write_attempt(first, corpus)
    lock_path = writer.synthesis_root / ".publication.lock"
    lock_path.unlink()
    reader = SynthesisArtifactStore(tmp_path)
    original_read_index = reader._read_index
    advanced = False

    def advance_before_index(
        attempt_ids: tuple[str, ...] | None = None,
    ) -> object:
        nonlocal advanced
        if not advanced:
            advanced = True
            writer._write_attempt_locked(second, corpus)
        return original_read_index(attempt_ids)

    monkeypatch.setattr(reader, "_read_index", advance_before_index)

    if read_boundary == "attempts":
        assert reader.attempts == (first, second)
    elif read_boundary == "accepted":
        assert reader.accepted_attempt == second
    else:
        assert reader.report_attempt == second
    assert advanced
    assert not lock_path.exists()


def test_lockless_reader_retries_transient_index_snapshot_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1)
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(attempt, corpus)
    lock_path = store.synthesis_root / ".publication.lock"
    lock_path.unlink()
    secure_read = synthesis_artifacts.secure_read_bytes
    snapshot_reads = 0

    def transient_snapshot_error(
        path: Path,
        label: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        nonlocal snapshot_reads
        if path == store.index_path and label == "synthesis index snapshot":
            snapshot_reads += 1
            if snapshot_reads == 1:
                raise OSError("index replaced during snapshot")
        return secure_read(path, label, max_bytes=max_bytes)

    monkeypatch.setattr(
        synthesis_artifacts,
        "secure_read_bytes",
        transient_snapshot_error,
    )

    assert store.report_attempt == attempt
    assert snapshot_reads >= 2
    assert not lock_path.exists()


@pytest.mark.parametrize("read_boundary", ("attempts", "accepted", "report"))
def test_reader_holds_publication_lock_through_bundle_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    read_boundary: str,
) -> None:
    corpus = _corpus(tmp_path)
    first = _attempt(corpus, sequence=1)
    second = _attempt(corpus, sequence=2)
    writer = SynthesisArtifactStore(tmp_path)
    writer.write_attempt(first, corpus)
    reader = SynthesisArtifactStore(tmp_path)
    original_read_index = reader._read_index
    reader_entered = Event()
    writer_started = Event()
    writer_finished = Event()

    def paused_read_index(
        attempt_ids: tuple[str, ...] | None = None,
    ) -> object:
        reader_entered.set()
        assert writer_started.wait(timeout=1)
        assert not writer_finished.wait(timeout=0.1)
        return original_read_index(attempt_ids)

    def publish() -> None:
        assert reader_entered.wait(timeout=1)
        writer_started.set()
        writer.write_attempt(second, corpus)
        writer_finished.set()

    monkeypatch.setattr(reader, "_read_index", paused_read_index)

    with ThreadPoolExecutor(max_workers=1) as executor:
        publication = executor.submit(publish)
        if read_boundary == "attempts":
            assert reader.attempts == (first,)
        elif read_boundary == "accepted":
            assert reader.accepted_attempt == first
        else:
            assert reader.report_attempt == first
        publication.result(timeout=2)

    assert writer_finished.is_set()
    assert SynthesisArtifactStore(tmp_path).accepted_attempt == second


def test_failed_index_replace_keeps_previous_accepted_pointer_and_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    first = _attempt(corpus, sequence=1)
    second = _attempt(corpus, sequence=2)
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(first, corpus)

    def fail_index_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(synthesis_artifacts, "_replace_index", fail_index_replace)
        with pytest.raises(OSError, match="simulated interruption"):
            store.write_attempt(second, corpus)

    assert store.accepted_attempt == first
    assert (tmp_path / "synthesis" / "attempts" / second.attempt_id).is_dir()

    third = _attempt(corpus, sequence=3, status=SynthesisStatus.REJECTED)
    store.write_attempt(third, corpus)
    assert {attempt.attempt_id for attempt in store.attempts} == {
        first.attempt_id,
        second.attempt_id,
        third.attempt_id,
    }
    assert store.accepted_attempt == first
    index = json.loads(store.index_path.read_text(encoding="ascii"))
    assert {record["attempt_id"] for record in index["attempts"]} == {
        first.attempt_id,
        second.attempt_id,
        third.attempt_id,
    }


def test_replace_index_windows_uses_secure_handle_relative_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / ".index.tmp"
    destination = tmp_path / "index.json"
    source.write_text("new", encoding="ascii")
    destination.write_text("old", encoding="ascii")
    calls: list[tuple[Path, Path, str, bool]] = []

    def fail_path_replace(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("path-based replacement must not be used on Windows")

    def reject_raced_parent(
        actual_source: Path,
        actual_destination: Path,
        label: str,
        *,
        replace_existing: bool = False,
    ) -> None:
        calls.append((actual_source, actual_destination, label, replace_existing))
        raise SynthesisArtifactError("destination parent changed during replacement")

    monkeypatch.setattr(synthesis_artifacts.os, "name", "nt")
    monkeypatch.setattr(synthesis_artifacts.os, "replace", fail_path_replace)
    monkeypatch.setattr(
        synthesis_artifacts, "secure_assert_ancestors", lambda path, label: None
    )
    monkeypatch.setattr(
        synthesis_artifacts, "secure_is_link_or_reparse", lambda path: False
    )
    monkeypatch.setattr(synthesis_artifacts, "secure_replace", reject_raced_parent)

    with pytest.raises(SynthesisArtifactError, match="parent changed"):
        synthesis_artifacts._replace_index(source, destination)

    assert calls == [
        (source, destination, "synthesis index publication", True),
    ]


@pytest.mark.parametrize("oversized_artifact", ("synthesis", "corpus"))
def test_writer_rejects_json_above_reader_size_limit_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    oversized_artifact: str,
) -> None:
    corpus = _corpus(tmp_path)
    attempt = replace(
        _attempt(corpus, sequence=1),
        limitations=("x" * 4096,),
    )
    corpus_bytes = corpus.to_json().encode("ascii")
    synthesis_bytes = synthesis_artifacts._canonical_bytes(
        synthesis_artifacts._attempt_to_dict(attempt)
    )
    target = synthesis_bytes if oversized_artifact == "synthesis" else corpus_bytes
    monkeypatch.setattr(
        synthesis_artifacts,
        "MAX_SYNTHESIS_JSON_BYTES",
        len(target) - 1,
    )
    publish_calls = 0
    original_publish = SynthesisArtifactStore._publish_attempt

    def counted_publish(
        store: SynthesisArtifactStore,
        staging: Path,
        destination: Path,
    ) -> None:
        nonlocal publish_calls
        publish_calls += 1
        original_publish(store, staging, destination)

    monkeypatch.setattr(SynthesisArtifactStore, "_publish_attempt", counted_publish)

    with pytest.raises(SynthesisArtifactError, match="exceeds size limit"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)

    assert publish_calls == 0
    attempts_root = tmp_path / "synthesis" / "attempts"
    assert not attempts_root.exists() or not tuple(attempts_root.iterdir())


def test_concurrent_writers_keep_each_attempt_and_index_record(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    attempts = tuple(_attempt(corpus, sequence=index) for index in (1, 2))

    def write(attempt: SynthesisAttempt) -> Path:
        return SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)

    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = tuple(executor.map(write, attempts))

    assert {path.name for path in paths} == {attempt.attempt_id for attempt in attempts}
    store = SynthesisArtifactStore(tmp_path)
    assert {attempt.attempt_id for attempt in store.attempts} == {
        attempt.attempt_id for attempt in attempts
    }
    index = json.loads(
        (tmp_path / "synthesis" / "index.json").read_text(encoding="ascii")
    )
    assert {record["attempt_id"] for record in index["attempts"]} == {
        attempt.attempt_id for attempt in attempts
    }


def test_concurrent_different_digest_writers_enforce_global_sequence(
    tmp_path: Path,
) -> None:
    first_corpus = _corpus(tmp_path)
    second_corpus = replace(first_corpus, metadata={"variant": "second"})
    attempts = (
        (_attempt(first_corpus, sequence=1), first_corpus),
        (_attempt(second_corpus, sequence=1), second_corpus),
    )

    def write(
        item: tuple[SynthesisAttempt, EvidenceCorpus],
    ) -> Path | SynthesisArtifactError:
        attempt, corpus = item
        try:
            return SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)
        except SynthesisArtifactError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(write, attempts))

    assert sum(isinstance(result, Path) for result in results) == 1
    collisions = [
        result for result in results if isinstance(result, SynthesisArtifactError)
    ]
    assert len(collisions) == 1
    assert "sequence already exists" in str(collisions[0])
    assert len(SynthesisArtifactStore(tmp_path).attempts) == 1


def test_mixed_creation_token_formats_share_global_sequence_namespace(
    tmp_path: Path,
) -> None:
    first_corpus = _corpus(tmp_path)
    second_corpus = replace(first_corpus, metadata={"variant": "second"})
    compact = _attempt(first_corpus, sequence=1)
    dashed = replace(
        _attempt(second_corpus, sequence=1),
        attempt_id=f"2026-08-10T120000Z-{second_corpus.digest[:12]}-1",
    )
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(compact, first_corpus)

    with pytest.raises(SynthesisArtifactError, match="sequence already exists"):
        store.write_attempt(dashed, second_corpus)


def test_latest_attempt_uses_full_created_at_before_global_sequence(
    tmp_path: Path,
) -> None:
    first_corpus = _corpus(tmp_path)
    second_corpus = replace(first_corpus, metadata={"variant": "second"})
    earlier = replace(
        _attempt(first_corpus, sequence=2, status=SynthesisStatus.UNAVAILABLE),
        created_at="2026-08-10T12:00:00.100000+00:00",
    )
    later = replace(
        _attempt(second_corpus, sequence=1, status=SynthesisStatus.REJECTED),
        created_at="2026-08-10T12:00:00.900000+00:00",
    )
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(earlier, first_corpus)
    store.write_attempt(later, second_corpus)

    assert store.report_attempt == later


@pytest.mark.parametrize(
    "attempt_id",
    [
        "attempt-1",
        "../20260810T120000Z-deadbeefdead-1",
        "20260810T120000Z-deadbeefdead-1",
        "20260810T120000Z-not-a-digest-1",
    ],
)
def test_write_attempt_rejects_malformed_or_mismatched_ids(
    tmp_path: Path,
    attempt_id: str,
) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1)
    attempt = replace(attempt, attempt_id=attempt_id)

    with pytest.raises(SynthesisArtifactError, match="attempt ID|digest"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


def test_write_attempt_rejects_digest_mismatch_and_overwrite(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1)
    mismatched = replace(attempt, corpus_digest="a" * 64)
    store = SynthesisArtifactStore(tmp_path)

    with pytest.raises(SynthesisArtifactError, match="corpus digest"):
        store.write_attempt(mismatched, corpus)
    store.write_attempt(attempt, corpus)
    with pytest.raises(SynthesisArtifactError, match="already exists|overwrite"):
        store.write_attempt(attempt, corpus)


def test_write_attempt_rejects_symlinked_attempt_target(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1)
    attempts_root = tmp_path / "synthesis" / "attempts"
    attempts_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = attempts_root / attempt.attempt_id
    try:
        target.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(SynthesisArtifactError, match="symlink|reparse"):
        SynthesisArtifactStore(tmp_path).write_attempt(attempt, corpus)


def test_accepted_index_cannot_point_to_rejected_attempt(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    accepted = _attempt(corpus, sequence=1)
    rejected = _attempt(corpus, sequence=2, status=SynthesisStatus.REJECTED)
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(accepted, corpus)
    store.write_attempt(rejected, corpus)
    index_path = tmp_path / "synthesis" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["accepted_attempt_id"] = rejected.attempt_id
    index_path.write_bytes(_canonical_bytes(index))

    with pytest.raises(SynthesisArtifactError, match="non-accepted"):
        _ = store.accepted_attempt


def _rewrite_synthesis_and_index(tmp_path: Path, value: dict[str, object]) -> None:
    synthesis_path = next((tmp_path / "synthesis" / "attempts").iterdir()) / (
        "synthesis.json"
    )
    synthesis_bytes = _canonical_bytes(value)
    synthesis_path.write_bytes(synthesis_bytes)
    index_path = tmp_path / "synthesis" / "index.json"
    index = json.loads(index_path.read_text(encoding="ascii"))
    record = index["attempts"][0]
    record["attempt_id"] = value["attempt_id"]
    record["created_at"] = value["created_at"]
    record["corpus_digest"] = value["corpus_digest"]
    record["status"] = value["status"]
    record["synthesis_digest"] = hashlib.sha256(synthesis_bytes).hexdigest()
    index["accepted_attempt_id"] = value["attempt_id"]
    index_path.write_bytes(_canonical_bytes(index))


def test_reader_rejects_selected_artifact_with_unreviewed_final_finding(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1, findings=(_finding(),))
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(attempt, corpus)
    synthesis_path = next((tmp_path / "synthesis" / "attempts").iterdir()) / (
        "synthesis.json"
    )
    value = json.loads(synthesis_path.read_text(encoding="ascii"))
    value["final_findings"][0]["reviewer_state"] = "pending"
    _rewrite_synthesis_and_index(tmp_path, value)

    with pytest.raises(SynthesisArtifactError, match="reviewer|publish"):
        _ = store.accepted_attempt


@pytest.mark.parametrize("invalid_kind", ("duplicate", "orphan"))
def test_reader_rejects_invalid_objection_identity(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1)
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(attempt, corpus)
    synthesis_path = next((tmp_path / "synthesis" / "attempts").iterdir()) / (
        "synthesis.json"
    )
    value = json.loads(synthesis_path.read_text(encoding="ascii"))
    objection = {
        "objection_id": "material-objection",
        "finding_id": (
            "unknown-finding" if invalid_kind == "orphan" else "invite-control"
        ),
        "severity": "material",
        "message": "The severity needs review.",
        "evidence_refs": [],
        "reviewer_role": "report-evidence-auditor",
        "resolved": False,
        "resolution": None,
        "resolved_by_role": None,
        "resolution_evidence_refs": [],
    }
    value["objections"] = (
        [objection, dict(objection)] if invalid_kind == "duplicate" else [objection]
    )
    _rewrite_synthesis_and_index(tmp_path, value)

    with pytest.raises(SynthesisArtifactError, match="objection|candidate"):
        _ = store.accepted_attempt


def test_reader_rejects_corrupted_final_core_claim(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1)
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(attempt, corpus)
    synthesis_path = next((tmp_path / "synthesis" / "attempts").iterdir()) / (
        "synthesis.json"
    )
    value = json.loads(synthesis_path.read_text(encoding="ascii"))
    value["final_findings"][0]["issue"] = (
        "A corrupted artifact replaced the reviewed claim."
    )
    _rewrite_synthesis_and_index(tmp_path, value)

    with pytest.raises(SynthesisArtifactError, match="reviewed|candidate"):
        _ = store.accepted_attempt


@pytest.mark.parametrize("read_boundary", ("attempts", "accepted_attempt"))
@pytest.mark.parametrize(
    "reference_field",
    ("evidence_refs", "resolution_evidence_refs"),
)
@pytest.mark.parametrize("invalid_kind", ("nonexistent", "mismatched", "boolean"))
def test_reader_and_selection_reject_objection_reference_outside_exact_corpus(
    tmp_path: Path,
    read_boundary: str,
    reference_field: str,
    invalid_kind: str,
) -> None:
    corpus = _corpus(tmp_path)
    finding = _finding()
    attempt = _attempt(
        corpus,
        sequence=1,
        findings=(finding,),
        objections=(_resolved_blocking_objection(finding.finding_id),),
    )
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(attempt, corpus)
    synthesis_path = next((tmp_path / "synthesis" / "attempts").iterdir()) / (
        "synthesis.json"
    )
    value = json.loads(synthesis_path.read_text(encoding="ascii"))
    hostile_ref = value["objections"][0][reference_field][0]
    if invalid_kind == "nonexistent":
        hostile_ref["evidence_id"] = "event:run-a:999"
        hostile_ref["replay_sequence"] = 999
    elif invalid_kind == "boolean":
        hostile_ref["replay_sequence"] = True
    else:
        hostile_ref["replay_sequence"] = 2
    _rewrite_synthesis_and_index(tmp_path, value)

    with pytest.raises(SynthesisArtifactError, match="objection|evidence|corpus"):
        if read_boundary == "attempts":
            _ = store.attempts
        else:
            _ = store.accepted_attempt


@pytest.mark.parametrize(
    "artifact_path",
    (1, {"path": "runs/run-a/event.json"}),
)
def test_evidence_ref_deserializer_rejects_non_string_artifact_path(
    artifact_path: object,
) -> None:
    with pytest.raises(SynthesisArtifactError, match="artifact_path.*string"):
        synthesis_artifacts._evidence_ref_from_dict(
            {
                "evidence_id": "event:run-a:1",
                "kind": "event",
                "run_id": "run-a",
                "artifact_path": artifact_path,
            }
        )


def test_storage_module_does_not_import_application_corpus() -> None:
    source = Path(synthesis_artifacts.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(
        module.startswith("ux_analyzer.application") for module in imported_modules
    )


def test_reader_rejects_unsupported_artifact_schema_version(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1)
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(attempt, corpus)
    synthesis_path = next((tmp_path / "synthesis" / "attempts").iterdir()) / (
        "synthesis.json"
    )
    value = json.loads(synthesis_path.read_text(encoding="ascii"))
    value["artifact_schema_version"] = "synthesis-artifact-v0"
    _rewrite_synthesis_and_index(tmp_path, value)

    with pytest.raises(SynthesisArtifactError, match="artifact schema"):
        _ = store.attempts


def test_reader_rejects_attempt_timestamp_mismatch(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1)
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(attempt, corpus)
    synthesis_path = next((tmp_path / "synthesis" / "attempts").iterdir()) / (
        "synthesis.json"
    )
    value = json.loads(synthesis_path.read_text(encoding="ascii"))
    value["created_at"] = "2026-08-11T12:00:00+00:00"
    _rewrite_synthesis_and_index(tmp_path, value)

    with pytest.raises(SynthesisArtifactError, match="timestamp|created_at"):
        _ = store.attempts


def test_reader_rejects_attempt_digest_prefix_mismatch(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    attempt = _attempt(corpus, sequence=1)
    store = SynthesisArtifactStore(tmp_path)
    store.write_attempt(attempt, corpus)
    attempts_root = tmp_path / "synthesis" / "attempts"
    old_path = attempts_root / attempt.attempt_id
    new_id = f"20260810T120000Z-{'0' * 12}-1"
    new_path = attempts_root / new_id
    old_path.rename(new_path)
    synthesis_path = new_path / "synthesis.json"
    value = json.loads(synthesis_path.read_text(encoding="ascii"))
    value["attempt_id"] = new_id
    _rewrite_synthesis_and_index(tmp_path, value)

    with pytest.raises(SynthesisArtifactError, match="digest prefix"):
        _ = store.attempts


def test_version_identifiers_are_always_hashed() -> None:
    version = "a" * 64

    assert (
        synthesis_artifacts._digest_identifier(version)
        == hashlib.sha256(version.encode("utf-8")).hexdigest()
    )
    assert synthesis_artifacts._digest_identifier(version) != version
