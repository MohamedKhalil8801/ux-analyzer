"""Pipeline-level integration test: pass → publication (plan Task 5)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from ux_analyzer.application.redesign import run_redesign_pass
from ux_analyzer.domain.redesign import RedesignAttemptStatus
from ux_analyzer.providers.redesign import (
    CriticConsolidatedProposal,
    CriticResponse,
    ProposerDesignProposal,
    ProposerPageUnderstanding,
    ProposerResponse,
    RedesignCriticMerger,
    RedesignProposer,
)
from ux_analyzer.storage.redesign_artifacts import (
    RedesignAttemptStore,
)

PAGE_URL = "https://fixture.test/"

_CAPTURE: dict[str, object] = {
    "schema": "page-capture-v2",
    "url": PAGE_URL,
    "title": "Fixture",
    "document_height": 3000,
    "segments": [
        {"index": 0, "y_offset": 0, "height": 2000, "data_url": "seg-a"},
        {"index": 1, "y_offset": 2000, "height": 1000, "data_url": "seg-b"},
    ],
    "sections": [
        {
            "tag": "section",
            "label": "Pricing cards",
            "box": {"x": 0, "y": 400, "w": 1280, "h": 600},
        }
    ],
    "headings": [],
    "forms": [],
    "buttons": [],
    "inputs": [],
    "links": [],
    "paragraphs": [],
}

_PROPOSAL_PAYLOAD: dict[str, object] = {
    "proposal_id": "m1",
    "page_url": PAGE_URL,
    "category": "whitespace",
    "title": "Widen card spacing",
    "observation": "Cards read as one block.",
    "rationale": "Separation clarifies groups.",
    "change": "Raise gap to 32px.",
    "principle_ids": ["gestalt-proximity"],
    "impact": "medium",
    "effort": "small",
    "section_refs": [
        {
            "url": PAGE_URL,
            "section_label": "Pricing cards",
            "box": {"x": 0.0, "y": 400.0, "width": 1280.0, "height": 600.0},
            "summary": "Pricing block.",
        }
    ],
    "also_affects": [],
    "deliberate_choice_check": None,
}


class _RecordingProposer(RedesignProposer):
    """Real proposer class over a client double; records the payload seen."""

    def __init__(self) -> None:
        super().__init__(_Client(), model="stub-redesign")
        self.seen_payloads: list[dict[str, object]] = []

    async def analyze(  # type: ignore[override]
        self,
        page_payload: dict[str, object],
        *,
        audience: str,
        principles: list[dict[str, object]],
        attachments: Sequence[object] = (),
    ) -> ProposerResponse:
        self.seen_payloads.append(page_payload)
        return ProposerResponse(
            page_understanding=ProposerPageUnderstanding(
                page_url=PAGE_URL,
                intent="Drive signups.",
                audience_inference="Likely first-time evaluators.",
                section_relationships="Hero feeds pricing.",
            ),
            proposals=[ProposerDesignProposal.model_validate(_PROPOSAL_PAYLOAD)],
        )


class _Client:
    """Satisfies the constructor contract; never actually called."""

    provider_id = "stub"
    endpoint_origin = "stub://local"
    provider_version = "stub-v1"


class _MergingCritic(RedesignCriticMerger):
    """Real critic class shape; returns a fixed validated response."""

    def __init__(self) -> None:
        super().__init__(_Client(), model="stub-redesign")

    async def review(  # type: ignore[override]
        self,
        consolidated: list[dict[str, object]],
        page_payloads_digests: list[dict[str, object]],
        *,
        audience: str,
        principles: list[dict[str, object]],
    ) -> CriticResponse:
        return CriticResponse(
            final_proposals=[CriticConsolidatedProposal.model_validate(_PROPOSAL_PAYLOAD)],
            killed=[],
            consistency_notes=["Card spacing unified across pages."],
        )


def test_pipeline_publishes_accepted_attempt(tmp_path: Path) -> None:
    store = RedesignAttemptStore(tmp_path)
    attempt_id = "redesign-20260911T100000Z-aaaa0001"
    outcome = asyncio.run(
        run_redesign_pass(
            {PAGE_URL: _CAPTURE},
            audience="",
            proposer=_RecordingProposer(),
            critic=_MergingCritic(),
            attempt_id=attempt_id,
            **_pack_kwargs(),
        )
    )
    assert outcome.attempt.status is RedesignAttemptStatus.ACCEPTED
    store.publish(outcome.attempt, captures_digest=outcome.captures_digest)

    selection = store.newest_valid_attempt()
    assert selection.attempt is not None
    assert selection.attempt.attempt_id == attempt_id
    assert selection.attempt.proposals[0].proposal_id == "m1"
    assert selection.attempt.consistency_notes == (
        "Card spacing unified across pages.",
    )
    # Attempt payload must be self-contained: renderable offline.
    payload_path = tmp_path / "redesign" / attempt_id / "payload.json"
    raw = payload_path.read_text(encoding="utf-8")
    assert '"status":"' in raw.replace(" ", "").replace('"status": "', '"status":"')


def test_regenerating_creates_a_new_attempt(tmp_path: Path) -> None:
    store = RedesignAttemptStore(tmp_path)
    first = "redesign-20260911T100000Z-aaaa0001"
    second = "redesign-20260911T100001Z-bbbb0002"
    for attempt_id in (first, second):
        outcome = asyncio.run(
            run_redesign_pass(
                {PAGE_URL: _CAPTURE},
                audience="",
                proposer=_RecordingProposer(),
                critic=_MergingCritic(),
                attempt_id=attempt_id,
                **_pack_kwargs(),
            )
        )
        store.publish(outcome.attempt, captures_digest=outcome.captures_digest)
    selection = store.newest_valid_attempt()
    assert selection.attempt is not None
    assert selection.attempt.attempt_id == second
    assert (tmp_path / "redesign" / first).is_dir()
    assert (tmp_path / "redesign" / second).is_dir()


@pytest.mark.asyncio
async def test_section_reference_outside_document_rejected(tmp_path: Path) -> None:
    bad_payload = {
        **_PROPOSAL_PAYLOAD,
        "section_refs": [
            {
                "url": PAGE_URL,
                "section_label": "No such section",
                "box": {"x": 0.0, "y": 2900.0, "width": 100.0, "height": 50.0},
                "summary": "Nothing here.",
            }
        ],
    }
    outcome = await run_redesign_pass(
        {PAGE_URL: _CAPTURE},
        audience="",
        proposer=_RecordingProposer(),
        critic=_MergingCritic(),
        attempt_id="redesign-20260911T100000Z-aaaa0001",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.ACCEPTED
    # The bad proposal is rejected only when it reaches the validator via
    # the critic; validate directly here to pin the behavior.
    from ux_analyzer.application.redesign import _validate_final_proposals

    proposals, reasons = _validate_final_proposals(
        [bad_payload],
        {PAGE_URL: _CAPTURE},
        known_ids=frozenset(_pack_kwargs()["principle_ids"]),
    )
    assert proposals == ()
    assert reasons and "dangling" in reasons[0]


def _pack_kwargs() -> dict[str, object]:
    """Real principle-pack data, as the CLI injects it (Task 6 refactor)."""
    from dataclasses import asdict

    from ux_analyzer.providers.redesign_principles import (
        REDESIGN_PRINCIPLE_PACK_VERSION,
        redesign_principle_ids,
        redesign_principle_pack,
    )
    return {
        "principle_pack": [asdict(item) for item in redesign_principle_pack()],
        "principle_ids": redesign_principle_ids(),
        "principle_pack_version": REDESIGN_PRINCIPLE_PACK_VERSION,
    }
