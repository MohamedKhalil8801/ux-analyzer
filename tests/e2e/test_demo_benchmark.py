from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import httpx
import pytest

from fixture_app.app import app as fixture_app
from tests.e2e.test_private_data_leakage import (
    _assert_no_leaks,
    _RecordingModelClient,
)
from tests.e2e.test_private_data_leakage import (
    _snapshot as _leakage_snapshot,
)
from ux_analyzer.application.evaluation import (
    EvaluationMetric,
    EvaluationTarget,
    RunEvaluationInputs,
    evaluate_run,
)
from ux_analyzer.application.experiment import (
    ExperimentContext,
    ExperimentRunner,
    expand_experiment,
)
from ux_analyzer.application.report import render_experiment_report
from ux_analyzer.application.run_agent import RunAgent
from ux_analyzer.config.loader import LoadedProject, load_project
from ux_analyzer.domain.attention import (
    AttentionState,
    InteractWithElement,
    ProgressiveObservation,
)
from ux_analyzer.domain.benchmark import Budget, ExperimentPolicy
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.interface import (
    BoundingBox,
    ElementSnapshot,
    PrivateExecutionReference,
    ViewportSnapshot,
)
from ux_analyzer.domain.run import (
    ActionExecuted,
    ObservationRecorded,
    RunSpec,
    VerificationResult,
)
from ux_analyzer.ports.artifacts import BundleManifest, RedactionPolicy, RunBundleWriter
from ux_analyzer.ports.models import ModelManifest, ModelRole
from ux_analyzer.ports.observation import (
    ClearTextAction,
    ObservationCapture,
    ObservationSessionConfig,
    PlatformAction,
    PlatformActionResult,
    SessionHandle,
    TypeTextAction,
    ViewportSize,
)
from ux_analyzer.ports.observation import (
    TestAccountId as _TestAccountId,
)
from ux_analyzer.providers.attention_policy import (
    AttentionPolicyConfig,
    ObservationSelection,
    ProgressiveAttentionPolicy,
)
from ux_analyzer.providers.cognitive import CognitiveDecision
from ux_analyzer.providers.finding_rules import FindingRuleSet
from ux_analyzer.providers.prominence import HeuristicProminenceProvider
from ux_analyzer.providers.scent import (
    StructuredCoarseScentEvaluator,
    StructuredFullScentEvaluator,
)
from ux_analyzer.storage.run_bundle import FilesystemRunBundleWriter

DEMO_PROJECT = Path(__file__).parents[2] / "benchmarks" / "demo" / "project.yaml"
CI_SEEDS = (7,)


@dataclass(frozen=True, slots=True)
class _ScriptedDecision:
    action: object
    reason: str


class _DeterministicProgressivePolicy:
    selection_mode = "progressive-acceptance"

    def __init__(self, scenario_id: str, version: str) -> None:
        self.scenario_id = scenario_id
        self.version = version

    def next_observation(
        self,
        state: AttentionState,
        snapshot: ViewportSnapshot,
        scores: object,
        coarse_scent: object,
        rng: object,
    ) -> ObservationSelection:
        del scores, coarse_scent, rng
        candidates = tuple(
            element
            for element in snapshot.elements
            if element.visibility_fraction > 0 and element.id not in state.noticed_ids
        )
        if not candidates:
            raise ValueError("no unobserved visible elements remain")

        def first_matching(*predicates: object) -> object | None:
            for element in candidates:
                if all(predicate(element) for predicate in predicates):
                    return element
            return None

        def actionable(element: object) -> bool:
            return bool(getattr(element, "actionable", False))

        def label(text: str):
            return lambda element: text in element.label.lower()

        target = None
        if self.scenario_id == "invite-teammate":
            if any(element.role.value == "input" for element in candidates):
                target = first_matching(
                    actionable, lambda element: element.role.value == "input"
                )
            if target is None:
                target = first_matching(actionable, label("invitation"))
            if target is None and self.version == "improved":
                target = first_matching(actionable, label("invite teammate"))
            if target is None:
                target = first_matching(actionable, label("members"))
            if target is None:
                target = first_matching(actionable, label("nk"))
        else:
            if any(element.role.value == "input" for element in candidates):
                target = first_matching(
                    actionable, lambda element: element.role.value == "input"
                )
            if target is None:
                target = first_matching(actionable, label("protection"))
            if target is None:
                target = first_matching(actionable, label("two-factor"))
        if target is None:
            target = next(
                (element for element in candidates if element.actionable), candidates[0]
            )
        selected = (target.id,)
        if not selected:
            raise ValueError("no unobserved visible elements remain")
        remembered = tuple(
            item.element_id
            for item in state.memory
            if item.element_id in {element.id for element in snapshot.elements}
            and item.element_id not in selected
        )
        region_id = snapshot.element(selected[0]).region_id
        observation = ProgressiveObservation.from_snapshot(
            snapshot,
            newly_revealed_ids=selected,
            remembered_ids=remembered,
            region_id=region_id,
        )
        return ObservationSelection(
            observation=observation,
            region_id=region_id,
            element_probabilities={element_id: 1.0 for element_id in selected},
            region_probabilities={region_id: 1.0},
            selection_mode=self.selection_mode,
        )


class _RecordedObservationProvider:
    id = "recorded-web"
    platform = "web"
    version = "recorded-web-v1"

    def __init__(self, snapshots: tuple[ViewportSnapshot, ...]) -> None:
        self.snapshots = snapshots
        self.capture_index = 0
        self.executed: list[PlatformAction] = []
        self.last_session: SessionHandle | None = None

    async def start_session(self, config: ObservationSessionConfig) -> SessionHandle:
        self.last_session = SessionHandle(
            session_id=config.session_id,
            test_account_id=_TestAccountId(
                f"test-{config.session_id.removeprefix('run-')}"
            ),
            viewport=config.viewport,
            trace_path=config.trace_path,
            blocked_events=[],
        )
        return self.last_session

    async def capture(self, session: SessionHandle) -> ObservationCapture:
        base = self.snapshots[min(self.capture_index, len(self.snapshots) - 1)]
        self.capture_index += 1
        snapshot_id = f"{base.id}-capture-{self.capture_index}"
        snapshot = replace(
            base,
            id=snapshot_id,
            elements=tuple(
                replace(
                    element,
                    id=f"{snapshot_id}-{index}",
                    execution_reference=(
                        None
                        if element.execution_reference is None
                        else replace(
                            element.execution_reference,
                            viewport_id=snapshot_id,
                        )
                    ),
                )
                for index, element in enumerate(base.elements)
            ),
        )
        return ObservationCapture(
            session_id=session.session_id,
            viewport_id=snapshot.id,
            url="http://fixture.test/recorded",
            title="Recorded fixture",
            viewport=session.viewport,
            screenshot=b"recorded-screenshot",
            snapshot=snapshot,
        )

    async def execute(
        self, session: SessionHandle, action: PlatformAction
    ) -> PlatformActionResult:
        self.executed.append(action)
        return PlatformActionResult(
            succeeded=True,
            url="http://fixture.test/recorded",
            duration_ms=1,
            state_changed=True,
        )

    async def reset(self, session: SessionHandle) -> None:
        del session
        self.capture_index = 0
        self.executed.clear()

    async def end_session(self, session: SessionHandle) -> None:
        del session


class _AcceptanceVerifier:
    def __init__(self, provider: _RecordedObservationProvider, spec: RunSpec) -> None:
        self.provider = provider
        self.spec = spec

    async def verify(self, session: SessionHandle) -> VerificationResult:
        del session
        action_kinds = [action.kind for action in self.provider.executed]
        has_typed_input = "type-text" in action_kinds
        click_count = action_kinds.count("click")
        required_clicks = 2 if self.spec.scenario.id == "invite-teammate" else 1
        verified = has_typed_input and click_count >= required_clicks
        return VerificationResult(
            verified=verified,
            evidence_ids=(f"recorded:{self.spec.run_id}",),
            details="recorded fixture state matched" if verified else "not complete",
        )


class _ScriptedCognitiveAgent:
    id = "recorded-cognitive-agent"
    version = "cognitive-recording-v1"

    def __init__(self, endpoint_origin: str) -> None:
        self.manifest = ModelManifest(
            provider_id="recorded-model",
            role=ModelRole.COGNITIVE,
            model_id="cognitive-model",
            endpoint_origin=endpoint_origin,
            prompt_version="cognitive-v1",
            schema_version="cognitive-v1",
        )
        self.cleared: set[str] = set()
        self.typed: set[str] = set()

    async def decide(
        self, goal: str, observation: ProgressiveObservation
    ) -> _ScriptedDecision:
        del goal
        elements = (
            *observation.newly_revealed_elements,
            *observation.remembered_elements,
        )
        actionable = [element for element in elements if element.actionable]

        completed_submit = next(
            (
                element
                for element in actionable
                if "invitation" in element.label.lower()
                or "two-factor" in element.label.lower()
                or "protection" in element.label.lower()
            ),
            None,
        )
        if completed_submit is not None and any(
            element.label in self.typed for element in elements
        ):
            return _ScriptedDecision(
                InteractWithElement(element_id=completed_submit.id),
                "Submit typed fixture input.",
            )

        def choose(label: str) -> object | None:
            return next(
                (element for element in actionable if element.label == label), None
            )

        target = choose("Invite teammate")
        if target is not None:
            return _ScriptedDecision(
                InteractWithElement(element_id=target.id),
                "Visible goal action matches invite task.",
            )
        menu = choose("Open account menu") or choose("NK")
        if menu is not None:
            return _ScriptedDecision(
                CognitiveDecision.model_validate(
                    {
                        "action": {"kind": "interact", "element_id": menu.id},
                        "reason": "Open account navigation.",
                    }
                ),
                "Open account navigation.",
            )
        members = choose("Members")
        if members is not None:
            return _ScriptedDecision(
                CognitiveDecision.model_validate(
                    {
                        "action": {"kind": "interact", "element_id": members.id},
                        "reason": "Open team management.",
                    }
                ),
                "Open team management.",
            )
        email = next(
            (element for element in actionable if element.label == "Email address"),
            None,
        )
        if email is not None:
            if "email" not in self.cleared:
                self.cleared.add("email")
                return _ScriptedDecision(
                    ClearTextAction(element_id=email.id, bounds=email.bounds),
                    "Clear fixture form value before typed fixture input.",
                )
            if "email" not in self.typed:
                self.typed.add("email")
                return _ScriptedDecision(
                    TypeTextAction(
                        element_id=email.id,
                        text="invite_email",
                        bounds=email.bounds,
                    ),
                    "Use scenario-provided invite input.",
                )
        code = next(
            (
                element
                for element in actionable
                if element.label in {"Setup code", "Verification code"}
            ),
            None,
        )
        if code is not None:
            if "code" not in self.typed:
                self.typed.add("code")
                return _ScriptedDecision(
                    TypeTextAction(
                        element_id=code.id,
                        text="totp_code",
                        bounds=code.bounds,
                    ),
                    "Use scenario-provided 2FA input.",
                )
        submit = next(
            (
                element
                for element in actionable
                if "invitation" in element.label.lower()
                or "two-factor" in element.label.lower()
                or "protection" in element.label.lower()
            ),
            None,
        )
        if submit is not None:
            return _ScriptedDecision(
                InteractWithElement(element_id=submit.id),
                "Submit completed fixture form.",
            )
        if actionable:
            return _ScriptedDecision(
                InteractWithElement(element_id=actionable[0].id),
                "Advance through recorded fixture control.",
            )
        return _ScriptedDecision(
            CognitiveDecision.model_validate(
                {
                    "action": {"kind": "wait"},
                    "reason": "Reveal next bounded observation.",
                }
            ),
            "Reveal next bounded observation.",
        )


class _BundleFactory:
    def __init__(self, output: Path, endpoint_origin: str) -> None:
        self.output = output
        self.endpoint_origin = endpoint_origin

    def start(self, spec: RunSpec) -> RunBundleWriter:
        manifest = BundleManifest.from_run_spec(
            spec,
            endpoint_origin=self.endpoint_origin,
            model_ids={"scent": "scent-model", "cognitive": "cognitive-model"},
            prompt_versions={
                "coarse-scent": "scent-coarse-v1",
                "full-scent": "scent-full-v1",
                "cognitive": "cognitive-v1",
            },
            provider_versions={
                "observation": "fixture-web-v1",
                "models": "recorded-model-v1",
            },
        )
        return FilesystemRunBundleWriter.start(
            self.output,
            manifest,
            redaction=RedactionPolicy.from_fixture_inputs(spec.scenario.fixture_inputs),
        )


def _recorded_snapshot(
    snapshot_id: str, elements: tuple[tuple[str, str, bool], ...]
) -> ViewportSnapshot:
    snapshots = tuple(
        ElementSnapshot(
            id=f"{snapshot_id}-{index}",
            role=role,
            label=label,
            bounds=BoundingBox(x=20 + index * 140, y=40, width=120, height=40),
            visibility_fraction=1,
            actionable=actionable,
            provider_id="recorded-web",
            execution_reference=PrivateExecutionReference(
                provider_id="recorded-web",
                viewport_id=snapshot_id,
                token=f"token-{snapshot_id}-{index}",
            ),
        )
        for index, (label, role, actionable) in enumerate(elements)
    )
    return ViewportSnapshot(
        id=snapshot_id,
        elements=snapshots,
        provider_id="recorded-web",
    )


async def _capture_fixture_snapshots() -> dict[
    tuple[str, str], tuple[ViewportSnapshot, ...]
]:
    captured: dict[tuple[str, str], tuple[ViewportSnapshot, ...]] = {}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fixture_app),
        base_url="http://fixture.test",
    ) as client:
        for version in ("defective", "improved"):
            invite_session = f"capture-invite-{version}"
            reset = await client.post(
                "/__control/reset", json={"session_id": invite_session}
            )
            assert reset.status_code == 200
            dashboard = await client.get(f"/app/{invite_session}/{version}")
            team = await client.get(f"/app/{invite_session}/{version}/team")
            assert dashboard.status_code == 200
            assert team.status_code == 200
            if version == "defective":
                assert "Share" in dashboard.text
            else:
                assert 'aria-label="Invite teammate"' in dashboard.text
            dashboard = _recorded_snapshot(
                f"recorded-{version}-dashboard",
                (("NK", "button", True), ("Invite teammate", "link", True)),
            )
            team = _recorded_snapshot(
                f"recorded-{version}-team",
                (
                    ("Email address", "input", True),
                    ("Send invitation", "button", True),
                    ("Form ready", "text", False),
                ),
            )
            captured[("invite-teammate", version)] = (
                (dashboard, dashboard, team)
                if version == "defective"
                else (dashboard, team)
            )

            twofa_session = f"capture-twofa-{version}"
            reset = await client.post(
                "/__control/reset", json={"session_id": twofa_session}
            )
            assert reset.status_code == 200
            settings = await client.get(f"/app/{twofa_session}/{version}/settings")
            assert settings.status_code == 200
            if version == "defective":
                assert "Protection" in settings.text
                code_label, button_label = "Verification code", "Enable protection"
            else:
                assert "Two-factor authentication" in settings.text
                code_label, button_label = (
                    "Setup code",
                    "Enable two-factor authentication",
                )
            captured[("enable-2fa", version)] = (
                _recorded_snapshot(
                    f"recorded-{version}-settings",
                    (
                        (code_label, "input", True),
                        (button_label, "button", True),
                        ("Security status", "text", False),
                    ),
                ),
            )
    return captured


def _project_matrix() -> tuple[LoadedProject, tuple[RunSpec, ...]]:
    loaded = load_project(DEMO_PROJECT)
    definition = next(
        experiment
        for experiment in loaded.project.experiments
        if experiment.id == "core-pair"
    )
    reduced = replace(definition, seeds=CI_SEEDS, run_count=len(CI_SEEDS))
    project = replace(
        loaded.project,
        scenarios=tuple(
            replace(
                scenario,
                budget=replace(scenario.budget, timeout_seconds=3),
            )
            for scenario in loaded.project.scenarios
        ),
    )
    specs = expand_experiment(
        ExperimentContext(
            definition=reduced,
            project=project,
            config_digest=loaded.config_digest,
        )
    )
    return loaded, specs


def _session_config(
    spec: RunSpec, output: Path, origin: str
) -> ObservationSessionConfig:
    page = (
        ""
        if spec.scenario.start_state == "dashboard"
        else f"/{spec.scenario.start_state}"
    )
    return ObservationSessionConfig(
        session_id=spec.run_id,
        start_url=f"{origin}/app/{spec.run_id}/{spec.application_version.kind.value}{page}",
        test_account_id=f"test-{spec.run_id.removeprefix('run-')}",
        viewport=ViewportSize(width=1280, height=800),
        trace_path=output / "traces" / f"{spec.run_id}.zip",
    )


def _target_for(result: object, scenario_id: str) -> EvaluationTarget:
    state = result.state
    selected: str | None = None
    for event in state.events:
        if not isinstance(event, ActionExecuted) or not isinstance(
            event.action, InteractWithElement
        ):
            continue
        snapshot = next(
            snapshot for snapshot in state.snapshots if snapshot.id == event.viewport_id
        )
        label = snapshot.element(event.action.element_id).label.lower()
        if scenario_id == "invite-teammate" and "invitation" in label:
            selected = event.action.element_id
        if scenario_id == "enable-2fa" and (
            "two-factor" in label or "protection" in label
        ):
            selected = event.action.element_id
    if selected is None:
        selected = next(
            (
                event.action.element_id
                for event in reversed(state.events)
                if isinstance(event, ActionExecuted)
                and isinstance(event.action, InteractWithElement)
            ),
            None,
        )
    if selected is None:
        raise AssertionError(f"no target action recorded for {scenario_id}")
    return EvaluationTarget(selected)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_demo_benchmark_runs_fixture_matrix_and_records_supported_evidence(
    tmp_path: Path,
) -> None:
    _, specs = _project_matrix()
    assert len(specs) == 16
    assert {
        (spec.scenario.id, spec.application_version.kind.value, spec.persona.id)
        for spec in specs
    } == {
        (scenario, version, persona)
        for scenario in ("invite-teammate", "enable-2fa")
        for version in ("defective", "improved")
        for persona in ("first-time-nontechnical", "impatient")
    }

    progressive_specs = tuple(
        spec
        for spec in specs
        if spec.policy is ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT
    )
    assert len(progressive_specs) == 8

    origin = "http://fixture.test"
    fixture_snapshots = await _capture_fixture_snapshots()
    client = _RecordingModelClient()
    providers: dict[str, _RecordedObservationProvider] = {}

    def factory(spec: RunSpec) -> RunAgent:
        provider = _RecordedObservationProvider(
            fixture_snapshots[(spec.scenario.id, spec.application_version.kind.value)]
        )
        providers[spec.run_id] = provider
        verifier = _AcceptanceVerifier(provider, spec)
        return RunAgent(
            observation_provider=provider,
            prominence_provider=HeuristicProminenceProvider(),
            attention_policy=_DeterministicProgressivePolicy(
                spec.scenario.id, spec.application_version.kind.value
            ),
            cognitive_agent=_ScriptedCognitiveAgent(client.endpoint_origin),
            verifier=verifier,
            bundle_factory=_BundleFactory(tmp_path, client.endpoint_origin),
            session_config_factory=lambda current_spec: _session_config(
                current_spec, tmp_path, origin
            ),
            coarse_scent_evaluator=StructuredCoarseScentEvaluator(
                client, model="scent-model"
            ),
            full_scent_evaluator=StructuredFullScentEvaluator(
                client, model="scent-model"
            ),
        )

    result = await ExperimentRunner(factory).run(progressive_specs, workers=1)

    assert result.failures == ()
    assert len(result.results) == 8
    assert len(client.requests) > 8

    metrics_by_identity: dict[tuple[str, str, str], object] = {}
    for spec, run_result in zip(progressive_specs, result.results, strict=True):
        assert run_result.outcome.kind == "verified-success", (
            f"{spec.scenario.id}/{spec.application_version.kind.value}/"
            f"{spec.persona.id}: {run_result.outcome.kind} "
            f"{run_result.terminal_reason}"
        )
        assert run_result.verification.verified
        assert not run_result.agent_claimed_success
        assert run_result.bundle_path is not None
        bundle = Path(run_result.bundle_path)
        assert all(
            (bundle / filename).exists()
            for filename in (
                "manifest.json",
                "timeline.jsonl",
                "result.json",
                "checksums.sha256",
            )
        )
        events = [
            json.loads(line)
            for line in (bundle / "timeline.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert events[-1]["kind"] == "run-terminated"
        assert [event["sequence"] for event in events] == list(
            range(1, len(events) + 1)
        )
        assert run_result.state.provider_manifests
        assert {manifest.role for manifest in run_result.state.provider_manifests} >= {
            "observation",
            "coarse-scent",
            "full-scent",
            "cognitive",
            "verification",
        }

        for event in run_result.state.events:
            if not isinstance(event, ObservationRecorded):
                continue
            assert 1 <= len(event.observation.newly_revealed_elements) <= 3
            snapshot = next(
                snapshot
                for snapshot in run_result.state.snapshots
                if snapshot.id == event.observation.viewport_id
            )
            for element in event.observation.newly_revealed_elements:
                assert element.id in {item.id for item in snapshot.elements}
                assert element.visibility_fraction > 0
                assert element.bounds.y < 800
                public = asdict(element)
                assert "execution_reference" not in public
                assert "selector" not in public
                assert "destination_url" not in public

        target = _target_for(run_result, spec.scenario.id)
        target_snapshot = next(
            snapshot
            for snapshot in reversed(run_result.state.snapshots)
            if target.element_id in {element.id for element in snapshot.elements}
        )
        prominence = HeuristicProminenceProvider().score(target_snapshot)
        target_score = next(
            score for score in prominence if score.element_id == target.element_id
        )
        assert target_score.feature_contributions
        inputs = RunEvaluationInputs(
            target_prominence=target_score.normalized_probability,
            scent_scores={target.element_id: 0.5},
            target_below_fold=False,
            ambiguous_target=(
                spec.application_version.kind.value == "defective"
                and spec.scenario.id == "invite-teammate"
            ),
            unexpected_hierarchy=(
                spec.application_version.kind.value == "defective"
                and spec.scenario.id == "enable-2fa"
            ),
            navigation_depth=(
                2
                if spec.application_version.kind.value == "defective"
                and spec.scenario.id == "invite-teammate"
                else 1
            ),
            feedback_observed=True,
            uncertainty=(
                0.8 if spec.application_version.kind.value == "defective" else 0.2
            ),
            model_dependent=True,
        )
        metrics = evaluate_run(run_result, target, inputs=inputs)
        findings = FindingRuleSet.default().evaluate(metrics)
        assert metrics.metric(EvaluationMetric.VERIFIED_COMPLETION).value == 1
        assert all(
            finding.evidence_class is not EvidenceClass.UNSUPPORTED_HUMAN_CLAIM
            for finding in findings
        )
        if inputs.ambiguous_target:
            assert any(
                finding.category == "ambiguous-icon-label" for finding in findings
            )
        metrics_by_identity[
            (spec.scenario.id, spec.application_version.kind.value, spec.persona.id)
        ] = metrics

        provider = providers[spec.run_id]
        assert provider.last_session is not None
        assert provider.last_session.blocked_events == []
        assert {action.kind for action in provider.executed} >= {"click"}
        if spec.scenario.id == "enable-2fa":
            assert "type-text" in {action.kind for action in provider.executed}

    for persona in ("first-time-nontechnical", "impatient"):
        defective = metrics_by_identity[("invite-teammate", "defective", persona)]
        improved = metrics_by_identity[("invite-teammate", "improved", persona)]
        assert improved.discovery_cost.total < defective.discovery_cost.total

    forbidden_values = {
        "api-key": "ci-api-key",
        "private-control-path": "/__control/",
        "fixture-state-key": "invited_email",
    }
    for index, request in enumerate(client.requests, start=1):
        _assert_no_leaks(f"model-request-{index}", request, forbidden_values)
    for spec, run_result in zip(progressive_specs, result.results, strict=True):
        for event in run_result.state.events:
            if isinstance(event, ObservationRecorded):
                _assert_no_leaks(
                    f"{spec.run_id}:observation",
                    asdict(event.observation),
                    {
                        **forbidden_values,
                        "sensitive-fixture-value": next(
                            iter(spec.scenario.fixture_inputs.values.values())
                        ),
                    },
                )

    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    report = output.read_text(encoding="utf-8").lower()
    assert "simulated benchmark evidence" in report
    assert "not real-user completion or satisfaction" in report
    assert "unsupported-human-claim" in report


@pytest.mark.e2e
def test_seeded_progressive_attention_has_repeatable_but_varying_paths() -> None:
    snapshot = _leakage_snapshot()
    state = AttentionState.initial(
        Budget(
            max_steps=10, max_observations=10, max_interactions=5, timeout_seconds=10
        ),
        confidence=0.5,
        frustration=0,
    )
    scores = HeuristicProminenceProvider().score(snapshot)
    policy = ProgressiveAttentionPolicy(AttentionPolicyConfig(batch_size=1))
    first = policy.next_observation(state, snapshot, scores, (), random.Random(7))
    repeat = policy.next_observation(state, snapshot, scores, (), random.Random(7))
    paths = {
        policy.next_observation(
            state, snapshot, scores, (), random.Random(seed)
        ).selected_ids
        for seed in range(1, 20)
    }
    assert first.selected_ids == repeat.selected_ids
    assert len(paths) > 1
