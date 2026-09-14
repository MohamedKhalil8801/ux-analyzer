"""Unit tests for the redesign pass pipeline (plan Task 5)."""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from ux_analyzer.application.redesign import (
    MAX_PROPOSALS_PER_PAGE,
    RedesignPassOutcome,
    captures_digest,
    page_payload_digests,
    run_redesign_pass,
    section_ref_resolves,
)
from ux_analyzer.domain.redesign import RedesignAttemptStatus
from ux_analyzer.providers.redesign import (
    CriticConsolidatedProposal,
    CriticResponse,
    ProposerDesignProposal,
    ProposerPageUnderstanding,
    ProposerResponse,
)

PAGE_URL = "https://fixture.test/"

_CAPTURE: dict[str, object] = {
    "schema": "page-capture-v2",
    "url": PAGE_URL,
    "title": "Fixture",
    "document_height": 3000,
    "captured_height": 3000,
    "truncated": False,
    "segments": [
        {"index": 0, "y_offset": 0, "height": 2000, "data_url": "a"},
        {"index": 1, "y_offset": 2000, "height": 1000, "data_url": "b"},
    ],
    "sections": [
        {
            "tag": "section",
            "label": "Pricing cards",
            "box": {"x": 0, "y": 400, "w": 1280, "h": 600},
        }
    ],
    "headings": [{"label": "Simple pricing", "box": {"x": 0, "y": 420, "w": 300, "h": 40}}],
    "buttons": [{"label": "Start free trial", "box": {"x": 10, "y": 900, "w": 200, "h": 48}}],
}

_CAPTURES = {PAGE_URL: _CAPTURE}


def _ref(**overrides: object) -> dict[str, object]:
    ref: dict[str, object] = {
        "url": PAGE_URL,
        "section_label": "Pricing cards",
        "box": {"x": 0.0, "y": 400.0, "width": 1280.0, "height": 600.0},
        "summary": "Pricing block.",
    }
    ref.update(overrides)
    return ref


def _proposal_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "proposal_id": "p1",
        "page_url": PAGE_URL,
        "category": "whitespace",
        "title": "Widen card spacing",
        "observation": "Cards read as one block.",
        "rationale": "Separation clarifies groups.",
        "change": "Raise gap to 32px.",
        "principle_ids": ["gestalt-proximity"],
        "impact": "medium",
        "effort": "small",
        "section_refs": [_ref()],
        "also_affects": [],
        "deliberate_choice_check": None,
    }
    payload.update(overrides)
    return payload


class _FakeModelFailure(RuntimeError):
    """Mimics the transport's ModelFailureError provider attributes."""

    def __init__(self, reason: str, **attributes: object) -> None:
        super().__init__(reason)
        self.reason = reason
        for name, value in attributes.items():
            setattr(self, name, value)


class _FakeProposer:
    def __init__(self, response: ProposerResponse | Exception) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []
        self.attachments_seen: list[tuple[object, ...]] = []

    async def analyze(
        self,
        page_payload: dict[str, object],
        *,
        audience: str,
        principles: list[dict[str, object]],
        attachments: Sequence[object] = (),
    ) -> ProposerResponse:
        self.calls.append(page_payload)
        recorded: list[tuple[str, str, bytes]] = []
        for attachment in attachments:
            attachment = cast(Any, attachment)
            recorded.append(
                (
                    str(attachment.media_type),
                    str(attachment.sha256),
                    Path(attachment.path).read_bytes(),
                )
            )
        self.attachments_seen.append(tuple(recorded))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeCritic:
    def __init__(
        self, response: CriticResponse | Exception
    ) -> None:
        self._response = response
        self.calls: list[tuple[list[dict[str, object]], list[dict[str, object]]]] = []

    async def review(
        self,
        consolidated: list[dict[str, object]],
        page_payloads_digests: list[dict[str, object]],
        *,
        audience: str,
        principles: list[dict[str, object]],
    ) -> CriticResponse:
        self.calls.append((consolidated, page_payloads_digests))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _proposer_response(
    proposals: list[dict[str, object]] | None = None,
) -> ProposerResponse:
    payloads = proposals if proposals is not None else [_proposal_payload()]
    return ProposerResponse(
        page_understanding=ProposerPageUnderstanding(
            page_url=PAGE_URL,
            intent="Drive signups.",
            audience_inference="Likely first-time evaluators.",
            section_relationships="Hero feeds pricing.",
        ),
        proposals=[
            ProposerDesignProposal.model_validate(payload) for payload in payloads
        ],
    )


def _critic_response(
    *,
    final: list[dict[str, object]] | None = None,
    killed: list[dict[str, object]] | None = None,
    notes: list[str] | None = None,
) -> CriticResponse:
    return CriticResponse(
        final_proposals=[
            CriticConsolidatedProposal.model_validate(item)
            for item in (final if final is not None else [_proposal_payload()])
        ],
        killed=killed or [],
        consistency_notes=notes or [],
    )


# ---------------------------------------------------------------------------
# Validation gate helpers
# ---------------------------------------------------------------------------


def test_section_ref_resolves_by_label_and_box() -> None:
    assert section_ref_resolves(_ref(), _CAPTURE)
    assert section_ref_resolves(
        _ref(section_label="simple pricing"), _CAPTURE
    )
    # Overlapping box with unknown label resolves via box.
    assert section_ref_resolves(_ref(section_label="Unknown", box={"x": 5.0, "y": 450.0, "width": 100.0, "height": 100.0}), _CAPTURE)


def test_section_ref_rejects_dangling_and_out_of_bounds() -> None:
    # In-bounds but overlapping nothing and matching no label → dangling.
    assert not section_ref_resolves(
        _ref(
            section_label="Nonexistent section",
            box={"x": 0.0, "y": 2900.0, "width": 100.0, "height": 50.0},
        ),
        _CAPTURE,
    )
    assert not section_ref_resolves(
        _ref(
            section_label="Unknown",
            box={"x": 0.0, "y": 9000.0, "width": 100.0, "height": 100.0},
        ),
        _CAPTURE,
    )


def test_captures_digest_is_order_independent_and_content_sensitive() -> None:
    first = captures_digest({PAGE_URL: _CAPTURE})
    second = captures_digest({PAGE_URL: _CAPTURE})
    assert first == second
    flipped = {"segments": _CAPTURE["segments"][::-1], **_CAPTURE}
    flipped.pop("segments")
    flipped = {**_CAPTURE, "segments": list(_CAPTURE["segments"])[::-1]}  # type: ignore[index]
    assert captures_digest({PAGE_URL: flipped}) != first  # type: ignore[arg-type]
    changed = {**_CAPTURE, "title": "Other"}
    assert captures_digest({PAGE_URL: changed}) != first  # type: ignore[arg-type]


def test_page_payload_digests_exclude_segment_data() -> None:
    digests = page_payload_digests(_CAPTURES)
    assert digests[0]["url"] == PAGE_URL
    assert digests[0]["segment_count"] == 2
    assert "Pricing cards" in digests[0]["section_labels"]
    assert "data_url" not in str(digests)


# ---------------------------------------------------------------------------
# Proposer wire payload (segments ride as attachments, not JSON text)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proposer_json_payload_strips_segment_pixels_to_metadata() -> None:
    """The JSON text carries ordered segment metadata, not the pixels.

    Embedding the base64 pixels a second time in the JSON would roughly
    double the per-page transport body; the pixels ride exactly once, as
    image attachments, while the JSON keeps index/y-offset/height/digest.
    """

    import hashlib
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(90, 90, 90)).save(buffer, format="JPEG")
    pixel_bytes = buffer.getvalue()
    capture: dict[str, object] = {
        **_CAPTURE,
        "segments": [
            {
                "index": 0,
                "y_offset": 0,
                "height": 2000,
                "data_url": "data:image/jpeg;base64,"
                + base64.b64encode(pixel_bytes).decode("ascii"),
            }
        ],
    }
    proposer = _FakeProposer(_proposer_response())
    outcome = await run_redesign_pass(
        {PAGE_URL: capture},
        audience="",
        proposer=proposer,
        critic=_FakeCritic(_critic_response()),
        attempt_id="redesign-20260911T000000Z-strip01",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.ACCEPTED
    sent_payload = proposer.calls[0]
    assert "data_url" not in json.dumps(sent_payload)
    segments = sent_payload["segments"]
    assert isinstance(segments, list) and len(segments) == 1
    meta = segments[0]
    assert meta["y_offset"] == 0 and meta["height"] == 2000
    assert meta["digest"] == hashlib.sha256(pixel_bytes).hexdigest()


@pytest.mark.asyncio
async def test_understanding_uses_capture_key_not_model_echo() -> None:
    """A provider-supplied understanding URL is never trusted over the
    canonical capture key (the page set is the only source of truth)."""

    proposer_response = ProposerResponse(
        page_understanding=ProposerPageUnderstanding(
            page_url="https://NotTheCapture.test/",
            intent="Intent.",
            audience_inference="Likely evaluators.",
            section_relationships="Hero feeds pricing.",
        ),
        proposals=[ProposerDesignProposal.model_validate(_proposal_payload())],
    )
    outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(proposer_response),
        critic=_FakeCritic(_critic_response()),
        attempt_id="redesign-20260911T000000Z-echo01",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.ACCEPTED
    assert outcome.attempt.page_understanding[0].page_url == PAGE_URL


# ---------------------------------------------------------------------------
# Pass outcomes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_accepts_validated_proposals() -> None:
    outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(_proposer_response()),
        critic=_FakeCritic(
            _critic_response(notes=["Unified card spacing across pages."])
        ),
        attempt_id="redesign-20260911T000000Z-aaaa1111",
        **_pack_kwargs(),
    )
    attempt = outcome.attempt
    assert attempt.status is RedesignAttemptStatus.ACCEPTED
    assert len(attempt.proposals) == 1
    assert attempt.proposals[0].proposal_id == "p1"
    assert attempt.proposals[0].category.value == "whitespace"
    assert attempt.consistency_notes == ("Unified card spacing across pages.",)
    assert attempt.page_understanding[0].audience_inference.startswith("Likely")
    assert outcome.captures_digest


@pytest.mark.asyncio
async def test_capture_segments_are_attached_in_order() -> None:
    """The proposer receives the persisted capture segments as attachments.

    The Transport contract (ADR 0007) says the proposer reads the whole page
    via its segment screenshots in order with y-offsets; the pipeline must
    therefore surface the persisted segments to the provider instead of
    sending a text-only payload.
    """

    import hashlib
    import io

    from PIL import Image

    def segment_bytes(color: int) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color=(color, color, color)).save(
            buffer, format="JPEG"
        )
        return buffer.getvalue()

    first = segment_bytes(90)
    second = segment_bytes(200)
    capture: dict[str, object] = {
        **_CAPTURE,
        "document_height": 4200,
        "segments": [
            {
                "y_offset": 0,
                "data_url": "data:image/jpeg;base64,"
                + base64.b64encode(first).decode("ascii"),
            },
            {
                "y_offset": 2000,
                "data_url": "data:image/jpeg;base64,"
                + base64.b64encode(second).decode("ascii"),
            },
        ],
    }
    proposer = _FakeProposer(_proposer_response())
    outcome = await run_redesign_pass(
        {PAGE_URL: capture},
        audience="",
        proposer=proposer,
        critic=_FakeCritic(_critic_response()),
        attempt_id="redesign-20260911T000000Z-attach01",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.ACCEPTED
    assert len(proposer.attachments_seen) == 1
    attachments = proposer.attachments_seen[0]
    assert len(attachments) == 2
    assert attachments[0][0] == "image/jpeg"
    assert attachments[0][1] == hashlib.sha256(first).hexdigest()
    assert attachments[1][1] == hashlib.sha256(second).hexdigest()
    assert attachments[0][2] == first
    assert attachments[1][2] == second


@pytest.mark.asyncio
async def test_critic_output_overrides_per_page_proposals() -> None:
    proposer_response = _proposer_response(
        [_proposal_payload(), _proposal_payload(proposal_id="p2")]
    )
    critic_response = _critic_response(
        final=[_proposal_payload(proposal_id="m1")],
        killed=[
            {"proposal_id": "p1", "reason": "merged into m1"},
            {"proposal_id": "p2", "reason": "duplicate of m1"},
        ],
    )
    outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(proposer_response),
        critic=_FakeCritic(critic_response),
        attempt_id="redesign-20260911T000000Z-bbbb2222",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.ACCEPTED
    assert [item.proposal_id for item in outcome.attempt.proposals] == ["m1"]
    assert [(k.proposal_id, k.reason) for k in outcome.attempt.killed] == [
        ("p1", "merged into m1"),
        ("p2", "duplicate of m1"),
    ]


@pytest.mark.asyncio
async def test_empty_final_proposals_is_valid_no_proposals() -> None:
    outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(_proposer_response()),
        critic=_FakeCritic(_critic_response(final=[], killed=[{"proposal_id": "p1", "reason": "unsupported"}])),
        attempt_id="redesign-20260911T000000Z-cccc3333",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.NO_PROPOSALS
    assert outcome.attempt.proposals == ()


@pytest.mark.asyncio
async def test_no_captures_is_unavailable() -> None:
    outcome = await run_redesign_pass(
        {},
        audience="",
        proposer=_FakeProposer(_proposer_response()),
        critic=_FakeCritic(_critic_response()),
        attempt_id="redesign-20260911T000000Z-dddd4444",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.UNAVAILABLE
    assert "no page captures" in outcome.attempt.unavailable_reason


@pytest.mark.asyncio
async def test_transport_failures_become_unavailable() -> None:
    proposer_outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(RuntimeError("boom")),
        critic=_FakeCritic(_critic_response()),
        attempt_id="redesign-20260911T000000Z-eeee5555",
        **_pack_kwargs(),
    )
    assert proposer_outcome.attempt.status is RedesignAttemptStatus.UNAVAILABLE
    assert "proposer failed" in proposer_outcome.attempt.unavailable_reason


@pytest.mark.asyncio
async def test_model_failure_error_adds_safe_details_to_unavailable_reason() -> None:
    """Provider metadata and sanitized structural diagnostics ride along in
    the unavailable reason; response content never does."""

    failure = _FakeModelFailure(
        "invalid structured output",
        status_code=200,
        error_code="INVALID_REQUEST",
        request_id="req-123",
        diagnostics={
            "stage": "schema_validation",
            "response_mode": "json-object",
            "attempt_count": 3,
            "finish_reason": "stop",
            "top_level_keys": ["private-key-shape"],
        },
    )
    outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(failure),
        critic=_FakeCritic(_critic_response()),
        attempt_id="redesign-20260911T000000Z-gggg7777",
        **_pack_kwargs(),
    )
    reason = outcome.attempt.unavailable_reason
    assert "proposer failed" in reason
    assert "status_code=200" in reason
    assert "error_code=INVALID_REQUEST" in reason
    assert "request_id=req-123" in reason
    assert "stage=schema_validation" in reason
    assert "mode=json-object" in reason
    assert "attempts=3" in reason
    assert "private-key-shape" not in reason
    assert len(reason) <= 300


@pytest.mark.asyncio
async def test_plain_exception_leaves_unavailable_reason_unchanged() -> None:
    outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(RuntimeError("boom")),
        critic=_FakeCritic(_critic_response()),
        attempt_id="redesign-20260911T000000Z-hhhh8888",
        **_pack_kwargs(),
    )
    assert outcome.attempt.unavailable_reason == (
        "proposer failed for https://fixture.test/: RuntimeError: boom"
    )
    critic_outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(_proposer_response()),
        critic=_FakeCritic(RuntimeError("boom")),
        attempt_id="redesign-20260911T000000Z-ffff6666",
        **_pack_kwargs(),
    )
    assert critic_outcome.attempt.status is RedesignAttemptStatus.UNAVAILABLE
    assert "critic/merger failed" in critic_outcome.attempt.unavailable_reason


@pytest.mark.asyncio
async def test_unknown_principle_id_rejects() -> None:
    outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(_proposer_response()),
        critic=_FakeCritic(
            _critic_response(
                final=[_proposal_payload(principle_ids=["laws-of-ux-verdict"])]
            )
        ),
        attempt_id="redesign-20260911T000000Z-aaaa7777",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.REJECTED
    assert any("unknown principle ids" in r for r in outcome.attempt.rejection_reasons)


@pytest.mark.asyncio
async def test_rejected_attempt_keeps_proposals_that_did_pass() -> None:
    """One malformed proposal must not erase the valid ones from the record:
    a rejected attempt persists the surviving, fully validated proposals
    alongside the rejection reasons."""

    outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(_proposer_response()),
        critic=_FakeCritic(
            _critic_response(
                final=[
                    _proposal_payload(proposal_id="good"),
                    _proposal_payload(
                        proposal_id="bad",
                        principle_ids=["laws-of-ux-verdict"],
                    ),
                ]
            )
        ),
        attempt_id="redesign-20260911T000000Z-bbbb8888",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.REJECTED
    assert any("bad" in r for r in outcome.attempt.rejection_reasons)
    assert [p.proposal_id for p in outcome.attempt.proposals] == ["good"]


@pytest.mark.asyncio
async def test_dangling_section_ref_rejects() -> None:
    outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(_proposer_response()),
        critic=_FakeCritic(
            _critic_response(
                final=[
                    _proposal_payload(
                        section_refs=[
                            _ref(
                                section_label="No such section",
                                box={
                                    "x": 0.0,
                                    "y": 2900.0,
                                    "width": 100.0,
                                    "height": 50.0,
                                },
                            )
                        ]
                    )
                ]
            )
        ),
        attempt_id="redesign-20260911T000000Z-bbbb8888",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.REJECTED
    assert any("dangling" in r for r in outcome.attempt.rejection_reasons)


@pytest.mark.asyncio
async def test_missing_deliberate_check_rejects_guardrail_category() -> None:
    """The deterministic gate catches violations regardless of source.

    The critic schema already blocks guardrail violations upstream (see the
    Task 4 unit tests); the gate is the second, transport-independent line.
    """

    from ux_analyzer.application.redesign import _validate_final_proposals

    proposals, reasons = _validate_final_proposals(
        [_proposal_payload(category="grouping")], _CAPTURES, known_ids=frozenset(_pack_kwargs()["principle_ids"])
    )
    assert proposals == ()
    assert reasons and "deliberate_choice_check" in reasons[0]


@pytest.mark.asyncio
async def test_guardrail_category_with_deliberate_check_accepts() -> None:
    payload = _proposal_payload(
        category="relocation",
        deliberate_choice_check={
            "pattern": "intentional edge anchoring",
            "rationale": "The anchor is unbounded; relocation restores order.",
        },
    )
    outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(_proposer_response()),
        critic=_FakeCritic(_critic_response(final=[payload])),
        attempt_id="redesign-20260911T000000Z-dddd0000",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.ACCEPTED
    assert outcome.attempt.proposals[0].deliberate_choice_check is not None


@pytest.mark.asyncio
async def test_pipeline_guards_against_oversized_json_view() -> None:
    """A hostile/oversized capture text cannot blow the proposer request.

    The pixels ride as attachments, so the text-only page payload has a hard
    budget; a capture whose inventory text alone exceeds it is surfaced as
    UNAVAILABLE instead of shipping a multi-megabyte JSON body (local guard
    against re-embedding segment pixels in the text)."""

    bloated = {
        **_CAPTURE,
        "sections": [{"label": "x" * 1_100_000}],
    }
    outcome = await run_redesign_pass(
        {PAGE_URL: bloated},
        audience="",
        proposer=_FakeProposer(_proposer_response()),
        critic=_FakeCritic(_critic_response()),
        attempt_id="redesign-20260911T000000Z-budget1",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.UNAVAILABLE
    assert "JSON budget" in outcome.attempt.unavailable_reason


@pytest.mark.asyncio
async def test_per_page_cap_enforced() -> None:
    proposals = [
        _proposal_payload(proposal_id=f"p{index}")
        for index in range(MAX_PROPOSALS_PER_PAGE + 1)
    ]
    outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(_proposer_response(proposals)),
        critic=_FakeCritic(
            _critic_response(
                final=[
                    _proposal_payload(proposal_id=f"m{index}")
                    for index in range(MAX_PROPOSALS_PER_PAGE + 1)
                ]
            )
        ),
        attempt_id="redesign-20260911T000000Z-eeee1111",
        **_pack_kwargs(),
    )
    assert outcome.attempt.status is RedesignAttemptStatus.REJECTED
    assert any("exceed the per-page bound" in r for r in (
        outcome.attempt.rejection_reasons
    ))


def test_malformed_proposals_do_not_inflate_per_page_count() -> None:
    """The per-page bound counts validated proposals only.

    A page with 12 valid plus a malformed final gets the real reasons for
    the malformed entry; the valid set is never falsely accused of exceeding
    the per-page bound.
    """

    from ux_analyzer.application.redesign import _validate_final_proposals

    valid = [
        _proposal_payload(proposal_id=f"p{index}")
        for index in range(MAX_PROPOSALS_PER_PAGE)
    ]
    malformed = _proposal_payload(proposal_id="bad", impact="huge")
    proposals, reasons = _validate_final_proposals(
        [dict(item) for item in [*valid, malformed]],
        _CAPTURES,
        known_ids=frozenset(_pack_kwargs()["principle_ids"]),
    )
    assert len(proposals) == MAX_PROPOSALS_PER_PAGE
    assert all(item.proposal_id.startswith("p") for item in proposals)
    assert any("bad" in reason for reason in reasons)
    assert not any(
        "exceed the per-page bound" in reason for reason in reasons
    )


@pytest.mark.asyncio
async def test_proposals_ordered_impact_then_effort() -> None:
    final = [
        _proposal_payload(proposal_id="low", impact="low", effort="large"),
        _proposal_payload(proposal_id="high", impact="high", effort="large"),
        _proposal_payload(proposal_id="mid", impact="high", effort="small"),
    ]
    outcome = await run_redesign_pass(
        _CAPTURES,
        audience="",
        proposer=_FakeProposer(_proposer_response()),
        critic=_FakeCritic(_critic_response(final=final)),
        attempt_id="redesign-20260911T000000Z-ffff2222",
        **_pack_kwargs(),
    )
    assert [p.proposal_id for p in outcome.attempt.proposals] == [
        "mid",
        "high",
        "low",
    ]


# ---------------------------------------------------------------------------
# Deterministic accessibility target-size guard (hit-target claims must
# reference a control the capture inventory actually sizes below 44px)
# ---------------------------------------------------------------------------


def _capture_with_buttons(buttons: list[dict[str, object]]) -> dict[str, object]:
    capture = dict(_CAPTURE)
    capture = {**capture, "buttons": buttons}
    return capture


_BIG_VIDEO_BUTTON = {
    "label": "Watch with sound",
    "box": {"x": 173, "y": 1607, "w": 320, "h": 569},
}


def _ref_box_from_inventory(box: dict[str, object]) -> dict[str, object]:
    """Inventory boxes use ``w``/``h``; proposal refs use ``width``/``height``."""

    return {
        "x": box["x"],
        "y": box["y"],
        "width": box["w"],
        "height": box["h"],
    }


def test_target_guard_rejects_claim_on_control_already_44px() -> None:
    """A whole video card carrying a caption label is not a small button."""

    from ux_analyzer.application.redesign import (
        _accessibility_target_size_reason,
        _validate_final_proposals,
    )

    capture = _capture_with_buttons([_BIG_VIDEO_BUTTON])
    ref = _ref(
        section_label="Watch with sound",
        box=_ref_box_from_inventory(_BIG_VIDEO_BUTTON["box"]),
    )
    reason = _accessibility_target_size_reason(ref, capture)
    assert reason is not None
    assert "already at least 44px" in reason
    assert "320x569" in reason

    payload = _proposal_payload(
        category="accessibility",
        title="Give the demo CTA a bigger tap target",
        change="Raise the tap target to 48px.",
        section_refs=[ref],
    )
    proposals, reasons = _validate_final_proposals(
        [payload],
        {PAGE_URL: capture},
        known_ids=frozenset(_pack_kwargs()["principle_ids"]),
    )
    assert proposals == ()
    assert any("hit-target claim" in item for item in reasons)


def test_target_guard_uses_effective_tap_box_before_own_box() -> None:
    """A control's effective tappable surface (its tap-reacting ancestor)
    is the measure, not the control's own painted box: a small widget inside
    a clickable card is already a large target in practice."""

    from ux_analyzer.application.redesign import (
        _accessibility_target_size_reason,
    )

    nested = {
        "label": "Play",
        "box": {"x": 20, "y": 20, "w": 30, "h": 30},
        "tap_box": {"x": 0, "y": 0, "w": 320, "h": 240},
    }
    capture = _capture_with_buttons([nested])
    ref = _ref(
        section_label="Play",
        box=_ref_box_from_inventory(nested["box"]),
    )
    reason = _accessibility_target_size_reason(ref, capture)
    assert reason is not None
    assert "effective tap target" in reason
    assert "320x240" in reason


def test_target_guard_accepts_genuinely_small_effective_surface() -> None:
    """When even the effective tappable surface is below the minimum, the
    hit-target claim stands."""

    from ux_analyzer.application.redesign import (
        _accessibility_target_size_reason,
    )

    small = {
        "label": "Show all 7",
        "box": {"x": 44, "y": 600, "w": 75, "h": 21},
        "tap_box": {"x": 44, "y": 600, "w": 75, "h": 21},
    }
    capture = _capture_with_buttons([small])
    ref = _ref(
        section_label="Show all 7",
        box=_ref_box_from_inventory(small["box"]),
    )
    assert _accessibility_target_size_reason(ref, capture) is None


def test_target_guard_resolves_the_best_matching_control() -> None:
    """A ref that overlaps several controls is measured against the one it
    most overlaps (intersection-over-union), not whichever is biggest: a
    tight ref on a small button inside a large decorative row must drop into
    the small button, whose effective surface is genuinely small."""

    from ux_analyzer.application.redesign import (
        _accessibility_target_size_reason,
    )

    big_row = {
        "label": "Featured article",
        "box": {"x": 0, "y": 0, "w": 590, "h": 50},
    }
    small_button = {
        "label": "Read more",
        "box": {"x": 40, "y": 5, "w": 75, "h": 21},
    }
    capture = _capture_with_buttons([big_row, small_button])
    # No label match on purpose: the ref names a region, and the tight box
    # must resolve to the small button by intersection-over-union, not to
    # the large decorative row it happens to sit inside.
    ref = _ref(
        section_label="Promo area",
        box=_ref_box_from_inventory(small_button["box"]),
    )
    assert _accessibility_target_size_reason(ref, capture) is None


def test_target_size_vocabulary_requires_a_size_judgment() -> None:
    """A target word alone is not a size claim: contrast and other
    accessibility write-ups that merely mention a tap target in passing must
    not be routed into the target-size guard."""

    from ux_analyzer.application.redesign import _mentions_target_size

    assert _mentions_target_size(("Raise the tap target to 48px.",)) is True
    assert _mentions_target_size(("The icon button is a tiny tap target.",)) is True
    assert _mentions_target_size(("Darken the label to AA contrast.",)) is False
    # Mentions a tap target, but no size judgment: contrast proposal.
    assert (
        _mentions_target_size(
            "Small labels have poor contrast; the gold eyebrows also read as "
             "quiet tap targets."
        )
        is False
    )



def test_target_guard_passes_genuinely_small_control() -> None:
    from ux_analyzer.application.redesign import _validate_final_proposals

    small = {"label": "Icon button", "box": {"x": 0, "y": 500, "w": 20, "h": 20}}
    capture = _capture_with_buttons([small])
    payload = _proposal_payload(
        category="accessibility",
        title="Enlarge the icon tap target",
        section_refs=[
            _ref(
                section_label="Icon button",
                box=_ref_box_from_inventory(small["box"]),
            )
        ],
    )
    proposals, reasons = _validate_final_proposals(
        [payload],
        {PAGE_URL: capture},
        known_ids=frozenset(_pack_kwargs()["principle_ids"]),
    )
    assert len(proposals) == 1
    assert proposals[0].proposal_id == "p1"
    assert not reasons


def test_target_guard_leaves_non_target_refs_alone() -> None:
    """A section-scoped ref with no interactive match is not a hit-target
    claim and must pass the gate untouched."""

    from ux_analyzer.application.redesign import _validate_final_proposals

    capture = _capture_with_buttons([_BIG_VIDEO_BUTTON])
    payload = _proposal_payload(category="accessibility")
    proposals, reasons = _validate_final_proposals(
        [payload],
        {PAGE_URL: capture},
        known_ids=frozenset(_pack_kwargs()["principle_ids"]),
    )
    assert len(proposals) == 1
    assert proposals[0].proposal_id == "p1"
    assert not reasons


def test_target_guard_skips_non_target_accessibility_claims() -> None:
    """A color-contrast proposal whose section overlaps a large control is
    not a hit-target claim; the guard must leave it alone."""

    from ux_analyzer.application.redesign import _validate_final_proposals

    capture = _capture_with_buttons([_BIG_VIDEO_BUTTON])
    payload = _proposal_payload(
        category="accessibility",
        title="Raise eyebrow-label contrast",
        change="Darken the label to AA contrast.",
    )
    proposals, reasons = _validate_final_proposals(
        [payload],
        {PAGE_URL: capture},
        known_ids=frozenset(_pack_kwargs()["principle_ids"]),
    )
    assert len(proposals) == 1
    assert proposals[0].proposal_id == "p1"
    assert not reasons
    """Proposers sometimes mirror the capture's ``w``/``h`` spelling; the
    gate must normalize it to the persisted ``width``/``height`` keys."""

    from ux_analyzer.application.redesign import (
        _normalize_ref_box,
        _validate_final_proposals,
    )

    assert _normalize_ref_box({"x": 1, "y": 2, "w": 3, "h": 4}) == {
        "x": 1,
        "y": 2,
        "width": 3,
        "height": 4,
    }
    # extra keys are dropped, missing keys still yield None (strict domain
    # check keeps rejecting genuinely malformed boxes)
    assert _normalize_ref_box({"x": 1, "y": 2, "w": 3, "h": 4, "junk": 9}) == {
        "x": 1,
        "y": 2,
        "width": 3,
        "height": 4,
    }
    assert _normalize_ref_box({"x": 1, "y": 2, "w": 3}) is None
    assert _normalize_ref_box("nope") is None

    capture = _capture_with_buttons([_BIG_VIDEO_BUTTON])
    payload = _proposal_payload(
        category="accessibility",
        title="Give the demo CTA a bigger tap target",
        change="Raise the tap target to 48px.",
        section_refs=[
            _ref(
                section_label="Watch with sound",
                box=dict(_BIG_VIDEO_BUTTON["box"]),  # w/h spelling on purpose
            )
        ],
    )
    proposals, reasons = _validate_final_proposals(
        [payload],
        {PAGE_URL: capture},
        known_ids=frozenset(_pack_kwargs()["principle_ids"]),
    )
    # Still rejected deterministically by the target-size guard, but for the
    # right reason (big control), not for the box key spelling.
    assert proposals == ()
    assert any("hit-target claim" in item for item in reasons)
    assert not any("exactly x, y, width, height" in item for item in reasons)


def test_pass_outcome_is_immutable() -> None:
    from dataclasses import FrozenInstanceError

    outcome = RedesignPassOutcome(
        attempt=None,  # type: ignore[arg-type]
        captures_digest="",
    )
    with pytest.raises(FrozenInstanceError):
        outcome.captures_digest = "x"  # type: ignore[misc]


def test_base_model_subclass_reference_documented() -> None:
    """Pydantic models reach the validator as subclasses of BaseModel."""

    assert issubclass(ProposerDesignProposal, BaseModel)


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
