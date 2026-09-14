"""Unit tests for redesign model roles (plan Task 4)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ux_analyzer.ports.models import (
    ChatMessage,
    ModelManifest,
    ModelRole,
)
from ux_analyzer.providers.redesign import (
    COMMON_REDESIGN_PROMPT,
    REDESIGN_CRITIC_MERGER_PROMPT_VERSION,
    REDESIGN_PROPOSER_PROMPT_VERSION,
    CriticConsolidatedProposal,
    CriticResponse,
    ProposerDeliberateChoiceCheck,
    ProposerDesignProposal,
    ProposerPageUnderstanding,
    ProposerResponse,
    ProposerSectionRef,
    RedesignCriticMerger,
    RedesignProposer,
)


def _section_ref(page_url: str = "https://fixture.test/") -> ProposerSectionRef:
    return ProposerSectionRef(
        url=page_url,
        section_label="Pricing cards",
        box={"x": 0.0, "y": 400.0, "width": 1280.0, "height": 600.0},
        summary="Three pricing cards in a row with feature lists.",
    )


def _proposal_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "proposal_id": "p1",
        "page_url": "https://fixture.test/",
        "category": "whitespace",
        "title": "Widen spacing between pricing cards",
        "observation": "Cards sit 8px apart and read as one block.",
        "rationale": "Grouping clarity suffers without separation.",
        "change": "Raise the gap to 32px.",
        "principle_ids": ["gestalt-proximity"],
        "impact": "medium",
        "effort": "small",
        "section_refs": [_section_ref()],
    }
    kwargs.update(overrides)
    return kwargs


_STUB_SECTION_REF = {
    "url": "https://fixture.test/",
    "section_label": "Pricing cards",
    "box": {"x": 0.0, "y": 400.0, "width": 1280.0, "height": 600.0},
    "summary": "Three pricing cards in a row.",
}

_STUB_PROPOSAL = {
    "proposal_id": "p1",
    "page_url": "https://fixture.test/",
    "category": "whitespace",
    "title": "Widen spacing between pricing cards",
    "observation": "Cards sit 8px apart and read as one block.",
    "rationale": "Grouping clarity suffers without separation.",
    "change": "Raise the gap to 32px.",
    "principle_ids": ["gestalt-proximity"],
    "impact": "medium",
    "effort": "small",
    "section_refs": [_STUB_SECTION_REF],
    "also_affects": [],
    "deliberate_choice_check": None,
}

_STUB_RESPONSES: dict[str, dict[str, object]] = {
    "ProposerResponse": {
        "page_understanding": {
            "page_url": "https://fixture.test/",
            "intent": "Explain the product and drive signups.",
            "audience_inference": "Likely first-time visitors evaluating pricing.",
            "section_relationships": "Hero feeds a feature row and a pricing block.",
        },
        "proposals": [_STUB_PROPOSAL],
    },
    "CriticResponse": {
        "final_proposals": [
            {**_STUB_PROPOSAL, "proposal_id": "m1"}
        ],
        "killed": [{"proposal_id": "p2", "reason": "duplicates m1"}],
        "consistency_notes": ["Applied the same card gap rule on every page."],
    },
}


class _StubClient:
    """Minimal StructuredModelClient double recording calls."""

    provider_id = "stub-structured"
    endpoint_origin = "stub://local"
    provider_version = "stub-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[type, tuple[ChatMessage, ...], str, ModelRole]] = []

    async def complete(self, schema, messages, model, role):  # type: ignore[no-untyped-def]
        self.calls.append((schema, messages, model, role))
        return schema.model_validate(_STUB_RESPONSES[schema.__name__])


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


def test_deliberate_choice_required_for_guardrail_categories() -> None:
    for category in ("grouping", "unification", "relocation", "simplification"):
        with pytest.raises(ValidationError, match="deliberate_choice_check"):
            ProposerDesignProposal.model_validate(
                _proposal_kwargs(category=category)
            )


def test_deliberate_choice_forbidden_outside_guardrail_categories() -> None:
    check = ProposerDeliberateChoiceCheck(
        pattern="intentional tight rhythm",
        rationale="The density is a deliberate brand statement.",
    )
    with pytest.raises(ValidationError, match="null outside"):
        ProposerDesignProposal.model_validate(
            _proposal_kwargs(deliberate_choice_check=check)
        )


def test_unknown_category_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown design category"):
        ProposerDesignProposal.model_validate(_proposal_kwargs(category="make-it-pop"))


def test_impact_and_effort_enums_enforced() -> None:
    with pytest.raises(ValidationError, match="impact must be one of"):
        ProposerDesignProposal.model_validate(_proposal_kwargs(impact="huge"))
    with pytest.raises(ValidationError, match="effort must be one of"):
        ProposerDesignProposal.model_validate(_proposal_kwargs(effort="tiny"))


def test_section_refs_must_reference_own_page() -> None:
    with pytest.raises(ValidationError, match="own page_url"):
        ProposerDesignProposal.model_validate(
            _proposal_kwargs(page_url="https://fixture.test/pricing")
        )


def test_empty_text_fields_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        ProposerDesignProposal.model_validate(_proposal_kwargs(title="  "))
    with pytest.raises(ValidationError, match="must not be empty"):
        ProposerDesignProposal.model_validate(_proposal_kwargs(change=""))


def test_principle_ids_must_be_unique_and_non_empty() -> None:
    with pytest.raises(ValidationError):
        ProposerDesignProposal.model_validate(
            _proposal_kwargs(principle_ids=["gestalt-proximity", "gestalt-proximity"])
        )
    with pytest.raises(ValidationError):
        ProposerDesignProposal.model_validate(_proposal_kwargs(principle_ids=[]))


def test_critic_consolidated_proposal_shares_invariants() -> None:
    with pytest.raises(ValidationError, match="deliberate_choice_check"):
        CriticConsolidatedProposal.model_validate(
            _proposal_kwargs(category="grouping")
        )
    accepted = CriticConsolidatedProposal.model_validate(
        _proposal_kwargs(
            category="grouping",
            deliberate_choice_check={
                "pattern": "scattered layout may aid scanning",
                "rationale": "The scatter is unbounded; a card restores order.",
            },
        )
    )
    assert accepted.category == "grouping"


def test_role_enum_contains_both_redesign_roles() -> None:
    assert ModelRole.REDESIGN_PROPOSER.value == "redesign-proposer"
    assert ModelRole.REDESIGN_CRITIC_MERGER.value == "redesign-critic-merger"


def test_prompt_versions_are_frozen_strings() -> None:
    assert RedesignProposer.prompt_version == REDESIGN_PROPOSER_PROMPT_VERSION
    assert RedesignCriticMerger.prompt_version == REDESIGN_CRITIC_MERGER_PROMPT_VERSION
    assert REDESIGN_PROPOSER_PROMPT_VERSION == "redesign-proposer-v1"
    assert REDESIGN_CRITIC_MERGER_PROMPT_VERSION == "redesign-critic-merger-v1"


def test_prompts_encode_doctrine() -> None:
    proposer = RedesignProposer(_StubClient(), model="gpt-redesign")
    critic = RedesignCriticMerger(_StubClient(), model="gpt-redesign")
    for prompt in (proposer.prompt, critic.prompt):
        assert "model estimate" in prompt
        assert "Never cite evidence IDs" in prompt
        assert "deliberate_choice_check" in prompt
        assert "inference" in prompt
    assert COMMON_REDESIGN_PROMPT in proposer.prompt
    assert COMMON_REDESIGN_PROMPT in critic.prompt
    assert "Redesign Proposer" in proposer.prompt
    assert "Critic/Merger" in critic.prompt


def test_manifest_describes_role_and_versions() -> None:
    client = _StubClient()
    proposer = RedesignProposer(client, model="gpt-redesign")
    manifest = proposer.manifest
    assert isinstance(manifest, ModelManifest)
    assert manifest.role is ModelRole.REDESIGN_PROPOSER
    assert manifest.prompt_version == REDESIGN_PROPOSER_PROMPT_VERSION
    assert manifest.schema_version == "redesign-proposer-v1"
    critic = RedesignCriticMerger(client, model="gpt-redesign")
    assert critic.manifest.role is ModelRole.REDESIGN_CRITIC_MERGER
    assert critic.manifest.schema_version == "redesign-critic-merger-v1"


def test_empty_model_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        RedesignProposer(_StubClient(), model="  ")
    with pytest.raises(ValueError, match="must not be empty"):
        RedesignCriticMerger(_StubClient(), model="")


# ---------------------------------------------------------------------------
# Role calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proposer_analyze_sends_payload_principles_and_schema() -> None:
    client = _StubClient()
    proposer = RedesignProposer(client, model="gpt-redesign")
    response = await proposer.analyze(
        {"url": "https://fixture.test/", "title": "Fixture"},
        audience="designers",
        principles=[{"id": "gestalt-proximity", "name": "Proximity"}],
    )
    assert response.proposals[0].proposal_id == "p1"
    schema, messages, model, role = client.calls[0]
    assert schema is ProposerResponse
    assert model == "gpt-redesign"
    assert role is ModelRole.REDESIGN_PROPOSER
    assert messages[0].role == "system"
    payload = json.loads(messages[1].content)
    assert payload["role_input"]["page_payload"]["title"] == "Fixture"
    assert payload["role_input"]["requested_audience"] == "designers"
    assert payload["role_input"]["redesign_principle_pack"][0]["id"] == (
        "gestalt-proximity"
    )
    assert payload["response_schema"]["schema_version"] == "redesign-proposer-v1"


@pytest.mark.asyncio
async def test_critic_review_sends_consolidated_and_digests() -> None:
    client = _StubClient()
    critic = RedesignCriticMerger(client, model="gpt-redesign")
    response = await critic.review(
        [{"proposal_id": "p1", "page_url": "https://fixture.test/"}],
        [{"url": "https://fixture.test/", "digest": "abc123"}],
        audience="",
        principles=[],
    )
    assert response.final_proposals[0].proposal_id == "m1"
    assert response.killed[0].proposal_id == "p2"
    assert response.consistency_notes
    schema, messages, _model, role = client.calls[0]
    assert schema is CriticResponse
    assert role is ModelRole.REDESIGN_CRITIC_MERGER
    payload = json.loads(messages[1].content)
    assert payload["role_input"]["consolidated_proposals"][0]["proposal_id"] == "p1"
    assert payload["role_input"]["page_payloads_digests"][0]["digest"] == "abc123"


def test_proposer_response_validates_schema_shape() -> None:
    response = ProposerResponse.model_validate(_STUB_RESPONSES["ProposerResponse"])
    assert response.schema_version == "redesign-proposer-v1"
    understanding: ProposerPageUnderstanding = response.page_understanding
    assert understanding.audience_inference.startswith("Likely")


def test_critic_response_allows_empty_finals() -> None:
    response = CriticResponse.model_validate(
        {
            "final_proposals": [],
            "killed": [{"proposal_id": "p1", "reason": "unsupported by capture"}],
            "consistency_notes": [],
        }
    )
    assert response.final_proposals == []
    assert response.killed[0].reason == "unsupported by capture"


def test_proposer_response_allows_zero_proposals() -> None:
    """A page with nothing worth proposing is a valid proposer outcome.

    The critic prompt already treats zero finals as valid; the proposer must
    be equally allowed to say "no changes" instead of failing the whole pass
    through schema validation (asymmetry fix).
    """

    response = ProposerResponse.model_validate(
        {
            "page_understanding": {
                "page_url": "https://fixture.test/",
                "intent": "Explain the product.",
                "audience_inference": "Likely first-time visitors.",
                "section_relationships": "Hero feeds pricing.",
            },
            "proposals": [],
        }
    )
    assert response.proposals == []
