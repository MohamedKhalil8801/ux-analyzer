"""Integration tests for the immutable redesign attempt store (Task 5)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ux_analyzer.domain.redesign import (
    DeliberateChoiceCheck,
    DesignCategory,
    DesignProposal,
    Effort,
    Impact,
    RedesignAttempt,
    RedesignAttemptStatus,
    SectionReference,
)
from ux_analyzer.storage.redesign_artifacts import (
    REDESIGN_INDEX_SCHEMA,
    REDESIGN_PAYLOAD_SCHEMA,
    RedesignArtifactError,
    RedesignAttemptStore,
    attempt_from_payload,
    attempt_to_payload,
    new_attempt_id,
)


def _proposal(proposal_id: str = "p1") -> DesignProposal:
    return DesignProposal(
        proposal_id=proposal_id,
        page_url="https://fixture.test/",
        category=DesignCategory.WHITESPACE,
        title="Widen card spacing",
        observation="Cards read as one block.",
        rationale="Separation clarifies groups.",
        change="Raise gap to 32px.",
        principle_ids=("gestalt-proximity",),
        impact=Impact.MEDIUM,
        effort=Effort.SMALL,
        section_refs=(
            SectionReference(
                url="https://fixture.test/",
                section_label="Pricing cards",
                box={"x": 0.0, "y": 400.0, "width": 1280.0, "height": 600.0},
                summary="Pricing block.",
            ),
        ),
    )


def _attempt(
    attempt_id: str,
    status: RedesignAttemptStatus = RedesignAttemptStatus.ACCEPTED,
    **overrides: object,
) -> RedesignAttempt:
    kwargs: dict[str, object] = {
        "attempt_id": attempt_id,
        "status": status,
        "proposals": (_proposal(),),
        "page_understanding": (),
        "consistency_notes": ("Unified card spacing.",),
        "pack_version": "redesign-principles-2026-09",
        "audience": "",
        "created_at": "2026-09-11T00:00:00Z",
    }
    kwargs.update(overrides)
    return RedesignAttempt(**kwargs)  # type: ignore[arg-type]


def test_publish_and_select_newest_valid_attempt(tmp_path: Path) -> None:
    store = RedesignAttemptStore(tmp_path)
    first = "redesign-20260911T100000Z-aaaa0001"
    second = "redesign-20260911T100001Z-bbbb0002"
    store.publish(_attempt(first), captures_digest="digest-a")
    store.publish(_attempt(second), captures_digest="digest-b")

    selection = store.newest_valid_attempt()
    assert selection.attempt is not None
    assert selection.attempt.attempt_id == second
    assert selection.skipped == ()
    assert (tmp_path / "redesign" / first / "payload.json").is_file()


def test_second_publish_never_mutates_first(tmp_path: Path) -> None:
    store = RedesignAttemptStore(tmp_path)
    first = "redesign-20260911T100000Z-aaaa0001"
    store.publish(_attempt(first))
    before = (tmp_path / "redesign" / first / "payload.json").read_bytes()
    store.publish(_attempt("redesign-20260911T100001Z-bbbb0002"))
    after = (tmp_path / "redesign" / first / "payload.json").read_bytes()
    assert before == after
    with pytest.raises(RedesignArtifactError, match="overwrite refused"):
        store.publish(_attempt(first))


def test_corrupt_newer_attempt_is_skipped_not_silent(tmp_path: Path) -> None:
    store = RedesignAttemptStore(tmp_path)
    older = "redesign-20260911T100000Z-aaaa0001"
    newer = "redesign-20260911T100001Z-bbbb0002"
    store.publish(_attempt(older))
    store.publish(_attempt(newer))
    # Corrupt the newer payload without touching the index digest.
    payload_path = tmp_path / "redesign" / newer / "payload.json"
    payload_path.write_bytes(
        payload_path.read_bytes().replace(b"Widen", b"Widex")
    )
    selection = store.newest_valid_attempt()
    assert selection.attempt is not None
    assert selection.attempt.attempt_id == older
    assert selection.skipped and "corrupt" in selection.skipped[0]


def test_payload_round_trip_preserves_attempt(tmp_path: Path) -> None:
    attempt = _attempt(
        new_attempt_id(),
        proposals=(
            DesignProposal(
                proposal_id="p1",
                page_url="https://fixture.test/",
                category=DesignCategory.RELOCATION,
                title="Move the CTA",
                observation="CTA is below the fold.",
                rationale="Primary action should be visible.",
                change="Relocate the CTA into the hero.",
                principle_ids=("hierarchy-f-pattern",),
                impact=Impact.HIGH,
                effort=Effort.MEDIUM,
                section_refs=(
                    SectionReference(
                        url="https://fixture.test/",
                        section_label="Pricing cards",
                        box={"x": 0.0, "y": 400.0, "width": 1280.0, "height": 600.0},
                        summary="Pricing block.",
                    ),
                ),
                deliberate_choice_check=DeliberateChoiceCheck(
                    pattern="long-scroll storytelling",
                    rationale="The story is not depth-ordered; CTA wins.",
                ),
            ),
        ),
    )
    payload = attempt_to_payload(attempt, captures_digest="abc")
    assert payload["schema"] == REDESIGN_PAYLOAD_SCHEMA
    restored = attempt_from_payload(json.loads(json.dumps(payload)))
    assert restored.attempt_id == attempt.attempt_id
    assert restored.status is attempt.status
    assert restored.proposals[0] == attempt.proposals[0]


def test_oversized_but_structurally_valid_attempt_is_not_selected(tmp_path: Path) -> None:
    """The report-path read is size-bounded: a structurally valid payload
    over MAX_REDESIGN_JSON_BYTES is skipped like any corrupt attempt, not
    slurped into memory (the renderer test used to pass only because the
    oversized payload happened to lack attempt_id)."""

    store = RedesignAttemptStore(tmp_path)
    attempt_id = "redesign-20260911T100000Z-aabbccdd"
    payload = attempt_to_payload(_attempt(attempt_id), captures_digest="")
    payload["pad"] = "x" * (8 * 1024 * 1024 + 4096)
    payload_bytes = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    directory = tmp_path / "redesign" / attempt_id
    directory.mkdir(parents=True)
    (directory / "index.json").write_text(
        json.dumps(
            {
                "schema": REDESIGN_INDEX_SCHEMA,
                "attempt_id": attempt_id,
                "status": "accepted",
                "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (directory / "payload.json").write_bytes(payload_bytes)

    selection = store.newest_valid_attempt()
    assert selection.attempt is None
    assert selection.skipped and attempt_id in selection.skipped[0]


def test_missing_root_is_empty_selection(tmp_path: Path) -> None:
    store = RedesignAttemptStore(tmp_path / "absent")
    selection = store.newest_valid_attempt()
    assert selection.attempt is None
    assert selection.skipped == ()


def test_non_attempt_directories_are_ignored(tmp_path: Path) -> None:
    store = RedesignAttemptStore(tmp_path)
    (tmp_path / "redesign" / "random-junk").mkdir(parents=True)
    (tmp_path / "redesign" / "random-junk" / "payload.json").write_text("{}")
    attempt_id = new_attempt_id()
    store.publish(_attempt(attempt_id))
    selection = store.newest_valid_attempt()
    assert selection.attempt is not None
    assert selection.attempt.attempt_id == attempt_id


def test_unavailable_attempt_persists_reason(tmp_path: Path) -> None:
    store = RedesignAttemptStore(tmp_path)
    attempt = _attempt(
        new_attempt_id(),
        status=RedesignAttemptStatus.UNAVAILABLE,
        proposals=(),
        unavailable_reason="proposer failed for https://fixture.test/: boom",
    )
    store.publish(attempt)
    selection = store.newest_valid_attempt()
    assert selection.attempt is not None
    assert selection.attempt.status is RedesignAttemptStatus.UNAVAILABLE
    assert "proposer failed" in selection.attempt.unavailable_reason


def test_rejected_attempt_persists_reasons(tmp_path: Path) -> None:
    store = RedesignAttemptStore(tmp_path)
    attempt = _attempt(
        new_attempt_id(),
        status=RedesignAttemptStatus.REJECTED,
        proposals=(),
        rejection_reasons=("p1: unknown principle ids: ['made-up']",),
    )
    store.publish(attempt)
    selection = store.newest_valid_attempt()
    assert selection.attempt is not None
    assert selection.attempt.rejection_reasons[0].startswith("p1:")
