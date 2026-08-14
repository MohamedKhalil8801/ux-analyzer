from __future__ import annotations

import ast
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

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
    return SynthesisObjection(
        objection_id="blocking-objection",
        finding_id=finding_id,
        severity=ObjectionSeverity.BLOCKING,
        message="The published claim remains contradicted by recorded evidence.",
        resolved=False,
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

    monkeypatch.setattr(synthesis_artifacts, "_replace_index", fail_index_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        store.write_attempt(second, corpus)

    assert store.accepted_attempt == first
    assert not (tmp_path / "synthesis" / "attempts" / second.attempt_id).exists()


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
