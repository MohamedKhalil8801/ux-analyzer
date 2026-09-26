"""Prompt-prefix stability for the report-synthesis roles.

The recorded five-attempt series spent 1.38M prompt tokens with a 0.3-13%
prompt-cache hit ratio even though every attempt ran against one identical
corpus (digest ``98bf4207803b...``). The endpoint does report cached tokens, so
the miss was not an endpoint limitation - it was the message layout.

Provider prompt caching only reuses a *prefix* of the request, so the reusable
bytes have to come first. Serializing the user payload with sorted keys put the
0.5 kB per-round retrieval policy ahead of the 17 kB principle pack and the
5 kB response schema, which truncated the reusable prefix to the 2.8 kB corpus
manifest. This module pins the fix:

1. every static block is serialized before anything that varies per round;
2. the serialized bytes are byte-stable for a given corpus, role, and prompt
   version - the property the cache depends on;
3. no information is lost or renamed: the payload's key set and values are
   unchanged, only the order the keys are written in;
4. an undeclared payload key fails closed instead of silently landing after
   the reusable prefix.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from ux_analyzer.application.evidence_corpus import EvidenceCorpus, EvidenceEntry
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.synthesis import EvidenceRef
from ux_analyzer.ports.models import ModelRole
from ux_analyzer.providers.report_synthesis import (
    _REPORT_PAYLOAD_KEY_ORDER,  # pyright: ignore[reportPrivateUsage]
    AnalystResponse,
    EvidenceAuditor,
    PatternReviewer,
    ReportAdjudicator,
    ReportAnalyst,
    _ordered_payload_json,  # pyright: ignore[reportPrivateUsage]
)
from ux_analyzer.providers.ux_principles import ux_principles

EVIDENCE_ID = "event:run-a:1"
SECOND_EVIDENCE_ID = "event:run-a:2"

STATIC_KEYS = (
    "corpus_manifest",
    "response_schema",
    "ux_principle_pack",
    "role_input",
)
VARYING_KEYS = (
    "evidence_request_policy",
    "prior_structured_output",
    "resolved_evidence",
    "resolved_evidence_context",
    "validation_feedback",
    "final_response_correction",
)


class _PayloadClient:
    """Client double that records only the user payload of each call."""

    endpoint_origin = "https://llm.example.test/v1"
    provider_id = "payload-client"
    provider_version = "recording-v1"

    def __init__(self) -> None:
        self.payloads: list[str] = []

    async def complete(
        self,
        schema: type[Any],
        messages: Sequence[Any],
        model: str,
        role: ModelRole,
    ) -> object:
        del model
        self.payloads.append(str(list(messages)[-1].content))
        payload: dict[str, object] = {"complete": True}
        if role is ModelRole.REPORT_ANALYST:
            payload["candidate_findings"] = []
        elif role is ModelRole.REPORT_ADJUDICATOR:
            payload["final_findings"] = []
            payload["objection_resolutions"] = []
        else:
            payload["objections"] = []
        return schema.model_validate(payload)


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
            EvidenceEntry(
                ref=EvidenceRef(
                    SECOND_EVIDENCE_ID, "event", "run-a", replay_sequence=2
                ),
                evidence_class=EvidenceClass.DETERMINISTIC_FACT,
                summary="User reached the invite form.",
                payload={"sequence": 2, "action": "interact"},
            ),
        ),
    )


def _static_prefix_bytes(payload: str) -> int:
    """Bytes of *payload* that belong to the static (cacheable) blocks."""

    parsed = cast(Mapping[str, Any], json.loads(payload))
    assert list(parsed) == [
        key for key in _REPORT_PAYLOAD_KEY_ORDER if key in parsed
    ], f"payload keys are out of the declared order: {list(parsed)}"
    lengths = {
        key: len(
            json.dumps(
                parsed[key], ensure_ascii=True, separators=(",", ":"), allow_nan=False
            )
        )
        + 1
        for key in parsed
    }
    total = 1
    for key in parsed:
        if key in VARYING_KEYS:
            return total
        total += len(key) + 1 + lengths[key] + 1
    return total


async def test_static_blocks_precede_every_varying_block(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    client = _PayloadClient()
    analyst = ReportAnalyst(client, model="m")

    for round_number in (1, 2, 3):
        await analyst.analyze(corpus, ux_principles(), retrieval_round=round_number)

    assert len(client.payloads) == 3
    for payload in client.payloads:
        parsed = cast(Mapping[str, Any], json.loads(payload))
        keys = list(parsed)
        assert set(keys) >= set(STATIC_KEYS)
        last_static = max(keys.index(key) for key in STATIC_KEYS if key in keys)
        first_varying = min(
            (keys.index(key) for key in VARYING_KEYS if key in keys),
            default=len(keys),
        )
        assert last_static < first_varying, keys
        assert set(keys) <= set(_REPORT_PAYLOAD_KEY_ORDER)


async def test_analyst_payload_is_byte_stable_across_retrieval_rounds(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    client = _PayloadClient()
    analyst = ReportAnalyst(client, model="m")

    for round_number in (1, 2, 3):
        await analyst.analyze(corpus, ux_principles(), retrieval_round=round_number)

    first, second, third = client.payloads
    shared = 0
    for left, right in ((first, second), (second, third), (first, third)):
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        shared = max(shared, index)
    static = _static_prefix_bytes(first)
    assert static > 0
    assert shared >= static, (
        f"rounds share {shared} bytes but the static prefix is {static} bytes; "
        "the cacheable prefix regressed"
    )


async def test_every_role_serializes_static_blocks_first(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    client = _PayloadClient()
    finding = {
        "finding_id": "invite-control",
        "title": "Invite control is hard to find",
        "issue": "People look in unrelated areas before finding the invite control.",
        "impact": "Important collaboration tasks take longer.",
        "root_cause": "The entry point is labeled around internal structure.",
        "fixes": ["Label the entry point around the user's goal."],
        "severity": "medium",
        "confidence": 0.8,
        "evidence_refs": [
            {
                "evidence_id": EVIDENCE_ID,
                "kind": "event",
                "run_id": "run-a",
                "replay_sequence": 1,
            }
        ],
        "severity_justification": "The evidence shows extra navigation on a key task.",
    }
    candidate = AnalystResponse(
        complete=True, candidate_findings=[finding]
    ).candidate_findings[0]

    await EvidenceAuditor(client, model="m").audit(corpus, ux_principles(), [candidate])
    await PatternReviewer(client, model="m").review(
        corpus, ux_principles(), [candidate]
    )
    await ReportAdjudicator(client, model="m").adjudicate(
        corpus, ux_principles(), [candidate], []
    )

    assert len(client.payloads) == 3
    for payload in client.payloads:
        keys = list(cast(Mapping[str, Any], json.loads(payload)))
        assert (
            keys[: len(STATIC_KEYS) - 1]
            == [key for key in STATIC_KEYS if key in keys][: len(STATIC_KEYS) - 1]
        )
        assert _static_prefix_bytes(payload) > 0


async def test_reordering_preserves_every_key_and_value(tmp_path: Path) -> None:
    """The fix is serialization order only; nothing is renamed or dropped."""

    corpus = _corpus(tmp_path)
    client = _PayloadClient()
    analyst = ReportAnalyst(client, model="m")
    await analyst.analyze(corpus, ux_principles())

    payload = cast(Mapping[str, Any], json.loads(client.payloads[0]))
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    round_tripped = json.loads(canonical)
    assert round_tripped == payload
    assert set(payload) <= set(_REPORT_PAYLOAD_KEY_ORDER)
    # Re-serializing in the declared order is a pure permutation of the keys.
    reordered = cast(
        Mapping[str, Any],
        json.loads(_ordered_payload_json(cast(Mapping[str, object], payload))),
    )
    assert reordered == payload
    assert list(reordered) == [
        key for key in _REPORT_PAYLOAD_KEY_ORDER if key in payload
    ]


def test_ordered_payload_fails_closed_on_an_undeclared_key() -> None:
    with pytest.raises(ValueError, match="missing from the declared order"):
        _ordered_payload_json({"undeclared_block": {"a": 1}})


def test_ordered_payload_is_stable_for_equal_content() -> None:
    payload: Mapping[str, object] = {
        "role_input": {"b": 2, "a": 1},
        "corpus_manifest": {"z": [3, {"y": 4, "x": 5}]},
    }
    reordered = {
        "corpus_manifest": {"z": [3, {"y": 4, "x": 5}]},
        "role_input": {"a": 1, "b": 2},
    }
    assert _ordered_payload_json(payload) == _ordered_payload_json(reordered)
    assert _ordered_payload_json(payload) == (
        '{"corpus_manifest":{"z":[3,{"x":5,"y":4}]},"role_input":{"a":1,"b":2}}'
    )


def test_declared_key_order_covers_every_payload_block() -> None:
    assert set(_REPORT_PAYLOAD_KEY_ORDER) == set(STATIC_KEYS) | set(VARYING_KEYS)
    positions = [
        _REPORT_PAYLOAD_KEY_ORDER.index(key) for key in STATIC_KEYS + VARYING_KEYS
    ]
    assert positions == sorted(positions)
    for static in STATIC_KEYS:
        for varying in VARYING_KEYS:
            assert _REPORT_PAYLOAD_KEY_ORDER.index(static) < (
                _REPORT_PAYLOAD_KEY_ORDER.index(varying)
            )
