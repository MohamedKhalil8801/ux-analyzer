"""Expand benchmark matrices and execute isolated runs with bounded concurrency."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, cast

from ux_analyzer.domain.benchmark import (
    BenchmarkProject,
    ExperimentDefinition,
    ExperimentPolicy,
)
from ux_analyzer.domain.run import RunSpec


def _empty_model_config() -> dict[str, str]:
    return {}


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    """Resolved experiment inputs shared by every policy comparison cell."""

    definition: ExperimentDefinition
    project: BenchmarkProject
    config_digest: str
    model_config: Mapping[str, str] = field(default_factory=_empty_model_config)

    def __post_init__(self) -> None:
        if not self.config_digest:
            raise ValueError("experiment config digest must not be empty")
        copied_model_config: dict[str, str] = dict(self.model_config)
        object.__setattr__(self, "model_config", MappingProxyType(copied_model_config))


@dataclass(frozen=True, slots=True)
class ExperimentFailure:
    """Sanitized failure record for one run that did not complete."""

    run_id: str
    error_type: str
    message: str
    spec: RunSpec

    @property
    def error(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Ordered partial result for one experiment execution."""

    specs: tuple[RunSpec, ...]
    results: tuple[object, ...]
    failures: tuple[ExperimentFailure, ...]
    cancelled: bool = False

    @property
    def run_results(self) -> tuple[object, ...]:
        return self.results

    @property
    def completed(self) -> tuple[object, ...]:
        return self.results

    @property
    def failed(self) -> tuple[ExperimentFailure, ...]:
        return self.failures

    @property
    def is_complete(self) -> bool:
        return not self.cancelled and not self.failures


class RunAgentExecutor(Protocol):
    """Minimal run-agent contract consumed by the experiment runner."""

    def execute(self, spec: RunSpec) -> Awaitable[object] | object: ...


AgentFactory = Callable[[RunSpec], RunAgentExecutor | Awaitable[RunAgentExecutor]]
ExperimentProgressCallback = Callable[
    [RunSpec, object | None, ExperimentFailure | None], Awaitable[None] | None
]


def deterministic_run_id(
    *,
    experiment_id: str,
    scenario_id: str,
    application_version_id: str,
    persona_id: str,
    policy: ExperimentPolicy,
    seed: int,
    config_digest: str,
) -> str:
    """Hash semantic cell inputs into a stable run identity."""

    payload = {
        "application_version_id": application_version_id,
        "config_digest": config_digest,
        "experiment_id": experiment_id,
        "persona_id": persona_id,
        "policy": policy.value,
        "scenario_id": scenario_id,
        "seed": seed,
    }
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"run-{hashlib.sha256(canonical).hexdigest()}"


def expand_experiment(
    definition: ExperimentDefinition | ExperimentContext,
    project: BenchmarkProject | None = None,
    *,
    config_digest: str | None = None,
) -> tuple[RunSpec, ...]:
    """Expand one resolved definition in deterministic Cartesian-product order."""

    if isinstance(definition, ExperimentContext):
        context = definition
        if project is not None:
            raise ValueError("project must not be supplied with experiment context")
    else:
        if project is None:
            raise ValueError("project or resolution context is required")
        context = ExperimentContext(
            definition=definition,
            project=project,
            config_digest=config_digest or project.id,
        )
    selected = context.definition
    scenarios = _index(context.project.scenarios, "scenario")
    versions = _index(
        (
            version
            for application in context.project.applications
            for version in application.versions
        ),
        "application version",
    )
    personas = _index(context.project.personas, "persona")
    chosen_scenarios = tuple(
        _resolve(scenarios, item, "scenario") for item in selected.scenario_ids
    )
    chosen_versions = tuple(
        _resolve(versions, item, "application version")
        for item in selected.application_version_ids
    )
    chosen_personas = tuple(
        _resolve(personas, item, "persona") for item in selected.persona_ids
    )
    policies = tuple(ExperimentPolicy(policy) for policy in selected.policies)
    seeds = selected.seeds or tuple(range(selected.run_count))
    if len(seeds) != len(set(seeds)):
        raise ValueError("experiment seeds must be unique")
    specs: list[RunSpec] = []
    seen_run_ids: set[str] = set()
    for scenario in chosen_scenarios:
        for version in chosen_versions:
            if version.id not in scenario.application_version_ids:
                continue
            for persona in chosen_personas:
                if persona.id not in scenario.eligible_persona_ids:
                    continue
                for policy in policies:
                    policy_seeds = seeds if policy.uses_seeded_attention else seeds[:1]
                    for seed in policy_seeds:
                        run_id = deterministic_run_id(
                            experiment_id=selected.id,
                            scenario_id=scenario.id,
                            application_version_id=version.id,
                            persona_id=persona.id,
                            policy=policy,
                            seed=seed,
                            config_digest=context.config_digest,
                        )
                        if run_id in seen_run_ids:
                            raise ValueError(f"duplicate generated run ID: {run_id}")
                        seen_run_ids.add(run_id)
                        specs.append(
                            RunSpec(
                                run_id=run_id,
                                seed=seed,
                                scenario=scenario,
                                application_version=version,
                                persona=persona,
                                policy=policy,
                                config_digest=context.config_digest,
                            )
                        )
    if not specs:
        raise ValueError("experiment expansion produced no eligible run specs")
    return tuple(specs)


class ExperimentRunner:
    """Execute isolated run agents while retaining partial experiment output."""

    def __init__(
        self,
        agent_factory: AgentFactory | RunAgentExecutor | None = None,
        *,
        run_agent_factory: AgentFactory | None = None,
        run_agent: RunAgentExecutor | None = None,
    ) -> None:
        selected: AgentFactory | RunAgentExecutor | None = (
            run_agent_factory or agent_factory or run_agent
        )
        if selected is None:
            raise ValueError("experiment runner needs run agent or agent factory")
        if callable(selected) and not hasattr(selected, "execute"):
            self._agent_factory: AgentFactory = selected
        else:
            shared_agent = cast(RunAgentExecutor, selected)
            self._agent_factory = lambda _spec: shared_agent

    async def run(
        self,
        specs: Sequence[RunSpec],
        workers: int = 1,
        *,
        on_complete: ExperimentProgressCallback | None = None,
    ) -> ExperimentResult:
        """Execute specs in input order with at most ``workers`` active runs."""

        if workers <= 0:
            raise ValueError("workers must be greater than zero")
        ordered_specs = tuple(specs)
        run_ids = [spec.run_id for spec in ordered_specs]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("experiment specs contain duplicate run IDs")
        if not ordered_specs:
            return ExperimentResult((), (), ())

        semaphore = asyncio.Semaphore(min(workers, len(ordered_specs)))
        results: dict[int, object] = {}
        failures: dict[int, ExperimentFailure] = {}
        completion_lock = asyncio.Lock()

        async def execute_one(index: int, spec: RunSpec) -> None:
            async with semaphore:
                agent: RunAgentExecutor | None = None
                error: BaseException | None = None
                cancellation: asyncio.CancelledError | None = None
                result: object | None = None
                try:
                    agent_value = await _await_value(self._agent_factory(spec))
                    agent = _require_agent(agent_value)
                    result = await _await_value(agent.execute(spec))
                except asyncio.CancelledError as caught:
                    cancellation = caught
                except Exception as caught:
                    error = caught
                finally:
                    if agent is not None:
                        try:
                            await _shielded_cleanup_agent(agent)
                        except BaseException as cleanup_error:
                            if cancellation is not None:
                                cause = cleanup_error.__cause__ or cleanup_error
                                raise cancellation from cause
                            if isinstance(cleanup_error, asyncio.CancelledError):
                                raise cleanup_error
                            if error is None:
                                error = cleanup_error
                if cancellation is not None:
                    raise cancellation
                if error is not None:
                    failure = ExperimentFailure(
                        run_id=spec.run_id,
                        error_type=type(error).__name__,
                        message=str(error) or type(error).__name__,
                        spec=spec,
                    )
                    failures[index] = failure
                    if on_complete is not None:
                        async with completion_lock:
                            await _await_value(on_complete(spec, None, failure))
                else:
                    results[index] = result
                    if on_complete is not None:
                        async with completion_lock:
                            await _await_value(on_complete(spec, result, None))

        tasks = [
            asyncio.create_task(execute_one(index, spec))
            for index, spec in enumerate(ordered_specs)
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return ExperimentResult(
            specs=ordered_specs,
            results=tuple(results[index] for index in sorted(results)),
            failures=tuple(failures[index] for index in sorted(failures)),
        )


def _index(items: Sequence[Any] | Any, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        item_id = getattr(item, "id", None)
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{label} must have non-empty ID")
        if item_id in result:
            raise ValueError(f"duplicate {label} ID: {item_id}")
        result[item_id] = item
    return result


def _resolve(items: Mapping[str, Any], item_id: str, label: str) -> Any:
    try:
        return items[item_id]
    except KeyError as error:
        raise ValueError(f"unknown {label} ID: {item_id}") from error


async def _await_value(value: object) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _require_agent(value: object) -> RunAgentExecutor:
    execute = getattr(value, "execute", None)
    if not callable(execute):
        raise TypeError("run agent must provide execute(spec)")
    return cast(RunAgentExecutor, value)


async def _cleanup_agent(agent: RunAgentExecutor) -> None:
    for method_name in ("cleanup", "aclose", "close", "shutdown"):
        method = getattr(agent, method_name, None)
        if callable(method):
            await _await_value(method())
            return


async def _shielded_cleanup_agent(agent: RunAgentExecutor) -> None:
    cleanup = asyncio.create_task(_cleanup_agent(agent))
    cancellation: asyncio.CancelledError | None = None
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as error:
            cancellation = error
    cleanup_error = cleanup.exception()
    if cancellation is not None:
        raise cancellation from cleanup_error
    if cleanup_error is not None:
        raise cleanup_error
