from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from ux_analyzer.providers.ux_principles import (
    UX_PRINCIPLE_PACK_VERSION,
    UxPrinciple,
    ux_principle_digest,
    ux_principles,
)

EXPECTED_PRINCIPLE_IDS = {
    "aesthetic-usability-effect",
    "choice-overload",
    "chunking",
    "cognitive-load",
    "common-region",
    "doherty-threshold",
    "fitts-law",
    "flow",
    "goal-gradient-effect",
    "hicks-law",
    "jakobs-law",
    "mental-models",
    "millers-law",
    "occams-razor",
    "pareto-principle",
    "paradox-of-the-active-user",
    "parkinsons-law",
    "peak-end-rule",
    "postels-law",
    "pragnanz",
    "proximity",
    "selective-attention",
    "serial-position-effect",
    "similarity",
    "teslers-law",
    "uniform-connectedness",
    "von-restorff-effect",
    "working-memory",
    "zeigarnik-effect",
}


def test_principle_pack_has_exact_static_unique_ids() -> None:
    principles = ux_principles()
    principle_ids = [principle.principle_id for principle in principles]

    assert UX_PRINCIPLE_PACK_VERSION == "ux-principles-v1"
    assert len(principle_ids) == len(EXPECTED_PRINCIPLE_IDS)
    assert len(set(principle_ids)) == len(principle_ids)
    assert set(principle_ids) == EXPECTED_PRINCIPLE_IDS
    assert principles == ux_principles()


def test_principle_pack_is_immutable_and_caller_safe() -> None:
    principles = ux_principles()
    first = principles[0]

    assert isinstance(principles, tuple)
    assert isinstance(first.diagnostic_questions, tuple)
    assert isinstance(first.applicability_cues, tuple)
    with pytest.raises(FrozenInstanceError):
        setattr(first, "name", "Changed")

    detached = list(principles)
    detached.clear()
    assert len(ux_principles()) == len(EXPECTED_PRINCIPLE_IDS)


def test_principle_pack_fields_are_nonempty_and_local_only() -> None:
    for principle in ux_principles():
        assert principle.principle_id.strip()
        assert principle.name.strip()
        assert principle.explanation.strip()
        assert principle.misuse_warning.strip()
        assert principle.diagnostic_questions
        assert principle.applicability_cues
        assert all(question.strip() for question in principle.diagnostic_questions)
        assert all(cue.strip() for cue in principle.applicability_cues)

        rendered = repr(principle).lower()
        assert "http://" not in rendered
        assert "https://" not in rendered
        assert "www." not in rendered


def test_ux_principle_normalizes_mutable_input_collections() -> None:
    questions = ["What changes for the user?"]
    cues = ["A user must compare alternatives."]
    principle = UxPrinciple(
        principle_id="example",
        name="Example",
        explanation="An operational explanation.",
        diagnostic_questions=questions,
        misuse_warning="Do not use this as evidence.",
        applicability_cues=cues,
    )

    questions.append("Can caller mutation leak in?")
    cues.clear()

    assert principle.diagnostic_questions == ("What changes for the user?",)
    assert principle.applicability_cues == ("A user must compare alternatives.",)


def test_principle_digest_is_canonical_and_repeatable() -> None:
    payload = {
        "principles": [
            asdict(principle)
            for principle in sorted(ux_principles(), key=lambda item: item.principle_id)
        ],
        "version": UX_PRINCIPLE_PACK_VERSION,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    expected = hashlib.sha256(canonical_json).hexdigest()

    assert ux_principle_digest() == expected
    assert ux_principle_digest() == ux_principle_digest()
    assert len(expected) == 64
