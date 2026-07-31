from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from ux_analyzer.application.experiment import ExperimentResult, ExperimentRunner
from ux_analyzer.domain.benchmark import (
    Application,
    ApplicationVersion,
    ApplicationVersionKind,
    BenchmarkProject,
    Budget,
    ExperimentDefinition,
    ExperimentPolicy,
    FixtureInputs,
    Persona,
    Scenario,
    ScenarioEvaluationTarget,
    VisibleResultVerifierSpec,
)
from ux_analyzer.domain.run import RunSpec


@dataclass
class FakeResult:
    run_id: str


class FakeAgent:
    def __init__(self, tracker: dict[str, int], *, failing: bool = False) -> None:
        self.tracker = tracker
        self.failing = failing

    async def execute(self, spec: RunSpec) -> FakeResult:
        self.tracker["active"] += 1
        self.tracker["max_active"] = max(
            self.tracker["max_active"], self.tracker["active"]
        )
        try:
            await asyncio.sleep(0.01)
            if self.failing:
                raise RuntimeError(f"failed {spec.run_id}")
            return FakeResult(spec.run_id)
        finally:
            self.tracker["active"] -= 1

    async def cleanup(self) -> None:
        self.tracker["cleaned"] += 1


def _spec(run_id: str) -> RunSpec:
    from ux_analyzer.application.experiment import expand_experiment

    defective = ApplicationVersion(
        id="app-defective", kind=ApplicationVersionKind.DEFECTIVE, label="Defective"
    )
    improved = ApplicationVersion(
        id="app-improved", kind=ApplicationVersionKind.IMPROVED, label="Improved"
    )
    project = BenchmarkProject(
        id="demo",
        name="Demo",
        applications=(
            Application(id="app", name="App", versions=(defective, improved)),
        ),
        scenarios=(
            Scenario(
                id="invite",
                name="Invite",
                goal="Invite teammate",
                application_version_ids=(defective.id, improved.id),
                start_state="dashboard",
                fixture_inputs=FixtureInputs(values={"email": "person@example.com"}),
                budget=Budget(
                    max_steps=10,
                    max_observations=10,
                    max_interactions=5,
                    timeout_seconds=10,
                ),
                verifier=VisibleResultVerifierSpec(type="visible-result", text="Sent"),
                safeguards=(),
                eligible_persona_ids=("persona",),
                expected_evidence=(),
                evaluation_target=ScenarioEvaluationTarget(
                    labels_by_version={
                        "defective": "Invite teammate",
                        "improved": "Invite teammate",
                    },
                    role="button",
                ),
            ),
        ),
        personas=(
            Persona(
                id="persona",
                name="Persona",
                working_memory_capacity=3,
                initial_confidence=0.5,
                initial_frustration=0.0,
                abandonment_threshold=0.8,
                attention_temperature=1.0,
            ),
        ),
        experiments=(),
    )
    definition = ExperimentDefinition(
        id="core",
        name="Core",
        scenario_ids=("invite",),
        application_version_ids=(defective.id, improved.id),
        persona_ids=("persona",),
        policies=(ExperimentPolicy.FULL_LIST,),
        seeds=(1,),
        run_count=1,
    )
    spec = expand_experiment(definition, project=project, config_digest="config-sha")[0]
    return RunSpec(
        run_id=run_id,
        seed=spec.seed,
        scenario=spec.scenario,
        application_version=spec.application_version,
        persona=spec.persona,
        policy=spec.policy,
        config_digest=spec.config_digest,
    )


@pytest.mark.asyncio
async def test_runner_defaults_to_serial_and_returns_partial_failures() -> None:
    tracker = {"active": 0, "max_active": 0, "cleaned": 0}
    created: list[str] = []
    specs = (_spec("run-1"), _spec("run-2"), _spec("run-3"))

    def factory(spec: RunSpec) -> FakeAgent:
        created.append(spec.run_id)
        return FakeAgent(tracker, failing=spec.run_id == "run-2")

    result = await ExperimentRunner(factory).run(specs)

    assert isinstance(result, ExperimentResult)
    assert tracker["max_active"] == 1
    assert [item.run_id for item in result.results] == ["run-1", "run-3"]
    assert [item.run_id for item in result.failures] == ["run-2"]
    assert created == ["run-1", "run-2", "run-3"]
    assert tracker["cleaned"] == 3


@pytest.mark.asyncio
async def test_runner_bounds_concurrency() -> None:
    tracker = {"active": 0, "max_active": 0, "cleaned": 0}
    specs = tuple(_spec(f"run-{index}") for index in range(5))

    result = await ExperimentRunner(lambda _: FakeAgent(tracker)).run(specs, workers=2)

    assert len(result.results) == 5
    assert tracker["max_active"] == 2
    assert tracker["cleaned"] == 5


@pytest.mark.asyncio
async def test_runner_cancellation_cleans_up_active_agents() -> None:
    tracker = {
        "active": 0,
        "max_active": 0,
        "cleanup_started": 0,
        "cleanup_finished": 0,
        "finalized": 0,
    }
    started = asyncio.Event()
    cleanup_started = asyncio.Event()

    class BlockingAgent(FakeAgent):
        async def execute(self, spec: RunSpec) -> FakeResult:
            self.tracker["active"] += 1
            started.set()
            try:
                await asyncio.Future()
                self.tracker["finalized"] += 1
                return FakeResult(spec.run_id)
            finally:
                self.tracker["active"] -= 1

        async def cleanup(self) -> None:
            self.tracker["cleanup_started"] += 1
            cleanup_started.set()
            await asyncio.sleep(0.01)
            self.tracker["cleanup_finished"] += 1

    runner = ExperimentRunner(lambda _: BlockingAgent(tracker))
    task = asyncio.create_task(runner.run((_spec("run-1"),), workers=1))
    await started.wait()
    task.cancel()
    await cleanup_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert tracker["cleanup_started"] == 1
    assert tracker["cleanup_finished"] == 1
    assert tracker["finalized"] == 0
