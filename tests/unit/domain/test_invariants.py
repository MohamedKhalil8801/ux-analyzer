from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from ux_analyzer.domain.attention import (
    AttentionAction,
    AttentionState,
    Complete,
    FullScent,
    InspectElement,
    InteractWithElement,
    NoticeElements,
    PersonaVisibleRegion,
    ProgressiveObservation,
)
from ux_analyzer.domain.benchmark import (
    ApplicationVersion,
    ApplicationVersionKind,
    Budget,
    ExperimentPolicy,
    FixtureInputs,
    Persona,
    Scenario,
    ScenarioEvaluationTarget,
    VisibleResultVerifierSpec,
)
from ux_analyzer.domain.findings import (
    EvidenceClass,
    Finding,
    UnsupportedHumanClaimError,
)
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    PrivateExecutionReference,
    RegionSnapshot,
    ViewportSnapshot,
)
from ux_analyzer.domain.run import (
    ArtifactChecksum,
    ObservationRecorded,
    ProviderManifest,
    RunSpec,
    RunStarted,
    RunState,
    RunTerminated,
    VerificationResult,
    VerifiedSuccess,
    ViewportCaptured,
)


def element_snapshot(
    element_id: str = "target",
    *,
    noticed_label: str = "Invite teammate",
    actionable: bool = True,
    viewport_id: str = "viewport-1",
) -> ElementSnapshot:
    return ElementSnapshot(
        id=element_id,
        role="button",
        label=noticed_label,
        bounds=BoundingBox(x=10, y=20, width=120, height=40),
        visibility_fraction=1.0,
        actionable=actionable,
        provider_id="fixture-provider",
        execution_reference=PrivateExecutionReference(
            provider_id="fixture-provider",
            viewport_id=viewport_id,
            token=f"token-{element_id}",
        ),
        selector=f"[data-testid='{element_id}']",
        test_id=element_id,
        hidden_label="private label",
        destination_url="https://fixture.invalid/private",
    )


def test_element_snapshot_preserves_legacy_positional_optional_order() -> None:
    snapshot = ElementSnapshot(
        "target",
        "button",
        "Visible label",
        BoundingBox(x=10, y=20, width=120, height=40),
        1.0,
        True,
        True,
        "region-1",
        "provider-1",
    )

    assert snapshot.disabled is True
    assert snapshot.region_id == "region-1"
    assert snapshot.provider_id == "provider-1"
    assert snapshot.rendered_text is None


def viewport_snapshot(
    viewport_id: str = "viewport-1",
    elements: tuple[ElementSnapshot, ...] = (),
) -> ViewportSnapshot:
    snapshot_elements = elements or (element_snapshot(viewport_id=viewport_id),)
    return ViewportSnapshot(
        id=viewport_id,
        provider_id="fixture-provider",
        elements=tuple(snapshot_elements),
    )


def run_spec() -> RunSpec:
    version = ApplicationVersion(
        id="app-improved",
        kind=ApplicationVersionKind.IMPROVED,
        label="Improved",
    )
    scenario = Scenario(
        id="invite",
        name="Invite",
        goal="Invite teammate",
        application_version_ids=(version.id,),
        start_state="dashboard",
        fixture_inputs=FixtureInputs(values={"email": "person@example.com"}),
        budget=Budget(
            max_steps=5,
            max_observations=3,
            max_interactions=2,
            timeout_seconds=10,
        ),
        verifier=VisibleResultVerifierSpec(
            type="visible-result", text="Invitation sent"
        ),
        safeguards=(),
        eligible_persona_ids=("persona",),
        expected_evidence=(),
        evaluation_target=ScenarioEvaluationTarget(
            labels_by_version={"improved": "Invite teammate"},
            role="button",
        ),
    )
    return RunSpec(
        run_id="run-1",
        seed=7,
        scenario=scenario,
        application_version=version,
        persona=Persona(
            id="persona",
            name="Persona",
            working_memory_capacity=3,
            initial_confidence=0.5,
            initial_frustration=0.1,
            abandonment_threshold=0.9,
            attention_temperature=1.0,
        ),
        policy=ExperimentPolicy.FULL_LIST,
        config_digest="config-digest",
    )


def test_evaluation_target_resolves_label_and_role_per_version() -> None:
    target = ScenarioEvaluationTarget(
        labels_by_version={"defective": "Share", "improved": "Invite teammate"},
        roles_by_version={"defective": "button", "improved": "link"},
    )
    defective = ApplicationVersion(
        id="app-defective",
        kind=ApplicationVersionKind.DEFECTIVE,
        label="Defective",
    )
    improved = ApplicationVersion(
        id="app-improved",
        kind=ApplicationVersionKind.IMPROVED,
        label="Improved",
    )

    assert target.label_for(defective) == "Share"
    assert target.role_for(defective) == "button"
    assert target.label_for(improved) == "Invite teammate"
    assert target.role_for(improved) == "link"


def terminal_event(
    *,
    verification: VerificationResult | None = None,
) -> RunTerminated:
    return RunTerminated(
        outcome=VerifiedSuccess(),
        verification=verification,
        provider_manifests=(
            ProviderManifest(
                provider_id="fixture-provider",
                role="observation",
                model_id=None,
                endpoint_origin="fixture.invalid",
                version="1",
            ),
        ),
        configuration_digest="config-digest",
        artifact_checksums=(ArtifactChecksum(path="timeline.jsonl", sha256="abc"),),
    )


def started_state() -> RunState:
    return RunState.initial(run_spec()).apply(RunStarted(run_id="run-1"))


def test_private_execution_reference_is_not_serialized_to_agent() -> None:
    visible = viewport_snapshot().persona_visible_elements()[0]

    dumped = visible.model_dump()

    assert "execution_reference" not in dumped
    assert "provider_id" not in dumped
    assert "selector" not in dumped
    assert "test_id" not in dumped
    assert "hidden_label" not in dumped
    assert "destination_url" not in dumped


def test_persona_projection_uses_rendered_text_without_aria_fallback() -> None:
    aria_only = element_snapshot(noticed_label="Switch dark theme")
    aria_only = replace(aria_only, rendered_text="")
    synthetic = element_snapshot(noticed_label="Synthetic label")

    assert viewport_snapshot(elements=(aria_only,)).persona_visible_elements()[0].label == ""
    assert viewport_snapshot(elements=(synthetic,)).persona_visible_elements()[0].label == "Synthetic label"


def test_persona_region_projection_uses_rendered_label_with_synthetic_fallback() -> None:
    live_region = RegionSnapshot(
        id="navigation",
        label="Primary secret",
        rendered_label="Navigation",
    )
    synthetic_region = RegionSnapshot(id="section", label="Synthetic section")

    assert PersonaVisibleRegion.from_snapshot(live_region).label == "Navigation"
    assert PersonaVisibleRegion.from_snapshot(synthetic_region).label == "Synthetic section"


def test_runtime_state_is_frozen() -> None:
    snapshot = viewport_snapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.id = "changed"  # type: ignore[misc]


def test_invalid_bounds_and_visibility_are_rejected() -> None:
    with pytest.raises(ValueError, match="width"):
        BoundingBox(x=0, y=0, width=0, height=10)

    with pytest.raises(ValueError, match="visibility_fraction"):
        ElementSnapshot(
            id="bad",
            role="button",
            label="Bad",
            bounds=BoundingBox(x=0, y=0, width=10, height=10),
            visibility_fraction=1.1,
            actionable=True,
        )


def test_budget_allows_no_overall_deadline_and_rejects_non_positive_deadline() -> None:
    unlimited = Budget(max_steps=5, max_observations=3, max_interactions=2)

    assert unlimited.timeout_seconds is None

    with pytest.raises(ValueError, match="timeout_seconds"):
        Budget(
            max_steps=5,
            max_observations=3,
            max_interactions=2,
            timeout_seconds=0,
        )


def test_duplicate_snapshot_element_ids_are_rejected() -> None:
    first = element_snapshot("same")
    second = element_snapshot("same")

    with pytest.raises(ValueError, match="duplicate element"):
        viewport_snapshot(elements=(first, second))


def test_private_execution_reference_rejects_stale_viewport() -> None:
    snapshot = viewport_snapshot()
    reference = snapshot.elements[0].execution_reference
    assert reference is not None

    with pytest.raises(ValueError, match="viewport"):
        snapshot.validate_execution_reference("target", reference.with_viewport("old"))


def test_observation_references_existing_elements_and_limits_new_batch() -> None:
    snapshot = viewport_snapshot(
        elements=(
            element_snapshot("one"),
            element_snapshot("two"),
            element_snapshot("three"),
            element_snapshot("four"),
        )
    )

    observation = ProgressiveObservation.from_snapshot(
        snapshot, newly_revealed_ids=("one", "two", "three")
    )
    assert len(observation.newly_revealed_elements) == 3

    with pytest.raises(ValueError, match="1 to 3"):
        ProgressiveObservation.from_snapshot(
            snapshot, newly_revealed_ids=("one", "two", "three", "four")
        )
    with pytest.raises(ValueError, match="not present"):
        ProgressiveObservation.from_snapshot(snapshot, newly_revealed_ids=("missing",))


def test_distinct_contact_evidence_remains_retainable_in_bounded_memory() -> None:
    snapshot = viewport_snapshot(
        elements=(
            replace(
                element_snapshot("email"),
                label="m.khalil.bus@gmail.com",
                rendered_text="m.khalil.bus@gmail.com",
            ),
            replace(
                element_snapshot("note"),
                label="Open to remote frontend work",
                rendered_text="Open to remote frontend work",
            ),
        )
    )
    state = AttentionState.initial(
        budget=run_spec().scenario.budget,
        confidence=0.5,
        frustration=0.1,
        memory_capacity=2,
    )
    state = state.after_observation(
        ProgressiveObservation.from_snapshot(snapshot, newly_revealed_ids=("email",))
    )
    state = state.after_observation(
        ProgressiveObservation.from_snapshot(
            snapshot,
            newly_revealed_ids=("note",),
            remembered_ids=("email",),
        )
    )

    assert [item.label for item in state.memory] == [
        "m.khalil.bus@gmail.com",
        "Open to remote frontend work",
    ]


def test_full_scent_requires_notice() -> None:
    state = AttentionState.initial(
        budget=run_spec().scenario.budget,
        confidence=0.5,
        frustration=0.1,
    )

    with pytest.raises(ValueError, match="noticed"):
        FullScent.for_element(state, element_id="target", score=0.8)


def test_interaction_requires_current_remembered_noticed_actionable_element() -> None:
    snapshot = viewport_snapshot()
    state = AttentionState.initial(
        budget=run_spec().scenario.budget,
        confidence=0.5,
        frustration=0.1,
    )
    action = InteractWithElement(element_id="target")

    with pytest.raises(ValueError, match="noticed"):
        state.validate_action(action, snapshot)

    observed = ProgressiveObservation.from_snapshot(
        snapshot, newly_revealed_ids=("target",)
    )
    state = state.after_observation(observed)
    state = state.after_action(InspectElement(element_id="target"))
    state.validate_action(action, snapshot)


def test_attention_actions_are_discriminated_by_kind() -> None:
    actions: tuple[AttentionAction, ...] = (
        NoticeElements(element_ids=("target",)),
        InspectElement(element_id="target"),
        InteractWithElement(element_id="target"),
        Complete(),
    )

    assert tuple(action.kind for action in actions) == (
        "notice-elements",
        "inspect-element",
        "interact-with-element",
        "complete",
    )


def test_complete_action_requires_current_viewport_and_consumes_step_only() -> None:
    snapshot = viewport_snapshot()
    state = AttentionState.initial(
        budget=run_spec().scenario.budget,
        confidence=0.5,
        frustration=0.1,
    ).after_observation(
        ProgressiveObservation.from_snapshot(snapshot, newly_revealed_ids=("target",))
    )

    state.validate_action(Complete(), snapshot)
    next_state = state.after_action(Complete())

    assert next_state.budgets.steps == state.budgets.steps - 1
    assert next_state.budgets.interactions == state.budgets.interactions

    with pytest.raises(ValueError, match="stale viewport"):
        state.validate_action(Complete(), viewport_snapshot("viewport-2"))


def test_run_state_apply_is_only_lifecycle_transition_and_finalizes() -> None:
    state = started_state()
    snapshot = viewport_snapshot()
    state = state.apply(ViewportCaptured(snapshot=snapshot))
    observation = ProgressiveObservation.from_snapshot(
        snapshot, newly_revealed_ids=("target",)
    )
    state = state.apply(ObservationRecorded(observation=observation))
    finalized = state.apply(
        terminal_event(
            verification=VerificationResult(verified=True, evidence_ids=("e1",))
        )
    )

    assert finalized.is_finalized
    assert isinstance(finalized.outcome, VerifiedSuccess)
    assert isinstance(finalized.events[-1], RunTerminated)

    with pytest.raises(ValueError, match="terminal"):
        finalized.apply(RunStarted(run_id="run-1"))


def test_terminal_event_requires_verification() -> None:
    with pytest.raises(ValueError, match="verification"):
        started_state().apply(terminal_event())


def test_finding_rejects_unsupported_human_claim() -> None:
    with pytest.raises(UnsupportedHumanClaimError, match="unsupported human claim"):
        Finding(
            finding_id="finding-1",
            category="satisfaction",
            severity="high",
            reproducibility="not-reproducible",
            evidence_class=EvidenceClass.UNSUPPORTED_HUMAN_CLAIM,
            evidence_ids=("human-opinion",),
            limitations=(),
        )


def test_supported_finding_references_evidence() -> None:
    finding = Finding(
        finding_id="finding-1",
        category="target-discovery",
        severity="medium",
        reproducibility="seeded",
        evidence_class=EvidenceClass.DETERMINISTIC_FACT,
        evidence_ids=("event-1",),
        limitations=("fixture-only",),
    )

    assert finding.evidence_ids == ("event-1",)
