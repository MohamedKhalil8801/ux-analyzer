"""Invariants for Design Proposal domain contracts (plan Task 1)."""

from dataclasses import FrozenInstanceError

import pytest

from ux_analyzer.domain.redesign import (
    DELIBERATE_CHOICE_CATEGORIES,
    DeliberateChoiceCheck,
    DesignCategory,
    DesignProposal,
    Effort,
    Impact,
    KilledProposal,
    PageUnderstanding,
    RedesignAttempt,
    RedesignAttemptStatus,
    SectionReference,
)


def _check(**overrides: object) -> DeliberateChoiceCheck:
    values: dict[str, object] = {
        "pattern": "The clustered buttons may be an intentional quick-actions group.",
        "rationale": "Even as a deliberate group, the labels still do not say what each action does.",
    }
    values.update(overrides)
    return DeliberateChoiceCheck(**values)  # type: ignore[arg-type]


def _ref(**overrides: object) -> SectionReference:
    values: dict[str, object] = {
        "url": "https://fixture.test/pricing",
        "section_label": "Pricing cards",
        "box": {"x": 24.0, "y": 480.0, "width": 640.0, "height": 320.0},
        "summary": "Three pricing cards in one row with dense copy.",
    }
    values.update(overrides)
    return SectionReference(**values)  # type: ignore[arg-type]


def _proposal(**overrides: object) -> DesignProposal:
    values: dict[str, object] = {
        "proposal_id": "prop-001",
        "page_url": "https://fixture.test/pricing",
        "category": DesignCategory.GROUPING,
        "title": "Group the pricing action buttons under one labeled block",
        "observation": "The three action buttons sit in separate card corners.",
        "rationale": "Gestalt proximity: related actions read as unrelated when scattered.",
        "change": "Wrap the three buttons in one group labeled 'Choose a plan'.",
        "principle_ids": ("gestalt-proximity",),
        "impact": Impact.HIGH,
        "effort": Effort.SMALL,
        "section_refs": (_ref(),),
        "also_affects": (),
        "deliberate_choice_check": _check(),
    }
    values.update(overrides)
    category = DesignCategory(values["category"])  # type: ignore[arg-type]
    guardrail = category in DELIBERATE_CHOICE_CATEGORIES
    if "deliberate_choice_check" not in overrides:
        values["deliberate_choice_check"] = _check() if guardrail else None
    return DesignProposal(**values)  # type: ignore[arg-type]


def test_design_category_enumerates_nine_families() -> None:
    assert {value.value for value in DesignCategory} == {
        "grouping",
        "whitespace",
        "unification",
        "radical-redesign",
        "copy",
        "relocation",
        "simplification",
        "new-section",
        "accessibility",
    }


def test_impact_and_effort_enums() -> None:
    assert {value.value for value in Impact} == {"low", "medium", "high"}
    assert {value.value for value in Effort} == {"small", "medium", "large"}


def test_attempt_status_enumerates_all_four_values() -> None:
    assert {value.value for value in RedesignAttemptStatus} == {
        "accepted",
        "no-proposals",
        "unavailable",
        "rejected",
    }


def test_deliberate_choice_categories_match_plan() -> None:
    assert DELIBERATE_CHOICE_CATEGORIES == frozenset(
        {
            DesignCategory.GROUPING,
            DesignCategory.UNIFICATION,
            DesignCategory.SIMPLIFICATION,
            DesignCategory.RELOCATION,
        }
    )


def test_proposal_requires_non_empty_core_fields() -> None:
    for field_name in ("proposal_id", "page_url", "title", "observation", "rationale", "change"):
        with pytest.raises(ValueError, match=field_name):
            _proposal(**{field_name: "   "})


def test_proposal_enum_fields_are_validated() -> None:
    assert _proposal(category="copy").category is DesignCategory.COPY
    assert _proposal(impact="medium").impact is Impact.MEDIUM
    assert _proposal(effort="large").effort is Effort.LARGE
    with pytest.raises(ValueError):
        _proposal(category="vibes")
    with pytest.raises(ValueError):
        _proposal(impact="enormous")
    with pytest.raises(ValueError):
        _proposal(effort="trivial")


def test_deliberate_choice_required_exactly_for_guardrail_categories() -> None:
    for category in DesignCategory:
        proposal = _proposal(category=category)
        if category in DELIBERATE_CHOICE_CATEGORIES:
            assert proposal.deliberate_choice_check is not None
        else:
            assert proposal.deliberate_choice_check is None

    for category in DELIBERATE_CHOICE_CATEGORIES:
        with pytest.raises(ValueError, match="deliberate_choice_check"):
            _proposal(category=category, deliberate_choice_check=None)
    for category in set(DesignCategory) - DELIBERATE_CHOICE_CATEGORIES:
        with pytest.raises(ValueError, match="deliberate_choice_check"):
            _proposal(category=category, deliberate_choice_check=_check())


def test_deliberate_choice_fields_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="pattern"):
        _check(pattern="  ")
    with pytest.raises(ValueError, match="rationale"):
        _check(rationale="  ")


def test_section_refs_non_empty_and_url_consistent() -> None:
    with pytest.raises(ValueError, match="section_refs"):
        _proposal(section_refs=())
    with pytest.raises(ValueError, match="section_refs"):
        _proposal(
            section_refs=(_ref(url="https://fixture.test/other"),),
        )


def test_section_reference_validation() -> None:
    assert _ref().box["x"] == pytest.approx(24.0)
    with pytest.raises(ValueError, match="url"):
        _ref(url="   ")
    with pytest.raises(ValueError, match="section_label"):
        _ref(section_label=" ")
    with pytest.raises(ValueError, match="summary"):
        _ref(summary="")
    with pytest.raises(ValueError, match="box"):
        _ref(box={"x": 1.0, "y": 2.0})
    with pytest.raises(ValueError, match="box"):
        _ref(box={"x": -1.0, "y": 0.0, "width": 10.0, "height": 10.0})
    with pytest.raises(TypeError, match="box"):
        _ref(box="top-left")


def test_principle_ids_unique_and_non_empty() -> None:
    with pytest.raises(ValueError):
        _proposal(principle_ids=("dup", "dup"))
    with pytest.raises(ValueError):
        _proposal(principle_ids=(" ",))


def test_proposals_are_frozen() -> None:
    proposal = _proposal()
    with pytest.raises(FrozenInstanceError):
        proposal.title = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        proposal.section_refs[0].url = "https://fixture.test/other"  # type: ignore[misc]


def _attempt(**overrides: object) -> RedesignAttempt:
    values: dict[str, object] = {
        "attempt_id": "attempt-1",
        "status": RedesignAttemptStatus.ACCEPTED,
        "proposals": (_proposal(),),
        "pack_version": "redesign-principles-2026-09",
    }
    values.update(overrides)
    return RedesignAttempt(**values)  # type: ignore[arg-type]


def test_attempt_requires_reason_for_unavailable_and_rejected() -> None:
    with pytest.raises(ValueError, match="unavailable_reason"):
        _attempt(status=RedesignAttemptStatus.UNAVAILABLE)
    with pytest.raises(ValueError, match="rejection_reasons"):
        _attempt(status=RedesignAttemptStatus.REJECTED)
    assert _attempt(status=RedesignAttemptStatus.UNAVAILABLE, unavailable_reason="no captures").status is (
        RedesignAttemptStatus.UNAVAILABLE
    )
    rejected = _attempt(
        status=RedesignAttemptStatus.REJECTED,
        proposals=(),
        rejection_reasons=("unknown-principle-id",),
    )
    assert rejected.rejection_reasons == ("unknown-principle-id",)


def test_attempt_unique_proposal_ids_and_killed_not_published() -> None:
    duplicate = _proposal(proposal_id="prop-001")
    with pytest.raises(ValueError, match="proposal_id"):
        _attempt(proposals=(_proposal(), duplicate))
    with pytest.raises(ValueError, match="killed"):
        _attempt(
            killed=(KilledProposal(proposal_id="prop-001", reason="duplicate of prop-002"),),
        )


def test_attempt_killed_proposals_require_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        KilledProposal(proposal_id="prop-009", reason=" ")


def test_attempt_accepts_string_status_and_coerces() -> None:
    attempt = _attempt(status="no-proposals", proposals=())
    assert attempt.status is RedesignAttemptStatus.NO_PROPOSALS


def test_attempt_page_understanding_is_typed() -> None:
    understanding = PageUnderstanding(
        page_url="https://fixture.test/pricing",
        intent="Help visitors compare plans and pick one.",
        audience_inference="First-time buyers comparing tiers.",
        section_relationships="Pricing cards sit under a comparison table.",
    )
    attempt = _attempt(page_understanding=(understanding,))
    assert attempt.page_understanding[0].page_url == "https://fixture.test/pricing"
    with pytest.raises(ValueError, match="intent"):
        PageUnderstanding(
            page_url="https://fixture.test/pricing",
            intent=" ",
            audience_inference="x",
            section_relationships="y",
        )


def test_attempt_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        _attempt().status = RedesignAttemptStatus.REJECTED  # type: ignore[misc]
