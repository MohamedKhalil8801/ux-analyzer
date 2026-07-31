"""Operator command surface for benchmark validation and execution."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NoReturn, cast
from urllib.parse import quote, urlsplit

import httpx
import typer

from ux_analyzer import __version__
from ux_analyzer.adapters.openai import (
    ModelConfigurationError,
    OpenAICompatibleSettings,
    OpenAICompatibleStructuredClient,
    load_environment_file,
)
from ux_analyzer.adapters.web.extractor import capture as capture_snapshot
from ux_analyzer.adapters.web.network_policy import BrowserAllowedOrigins
from ux_analyzer.adapters.web.session import PlaywrightSessionAdapter
from ux_analyzer.adapters.web.verifier import HttpFixtureStateClient, WebVerifier
from ux_analyzer.application.evaluation import (
    evaluate_experiment_results,
    evaluate_run,
    evaluation_inputs_for,
    evaluation_target_for,
)
from ux_analyzer.application.experiment import (
    ExperimentContext,
    ExperimentFailure,
    ExperimentResult,
    ExperimentRunner,
    expand_experiment,
)
from ux_analyzer.application.run_agent import (
    AttentionPolicy,
    ProminenceProvider,
    RunAgent,
    RunResult,
)
from ux_analyzer.application.run_agent import CognitiveAgent as RunCognitiveAgent
from ux_analyzer.config.loader import (
    LoadedProject,
    ProjectConfigError,
    RuntimeConfig,
    load_project,
)
from ux_analyzer.domain.attention import CompleteObservation, ProgressiveObservation
from ux_analyzer.domain.benchmark import ExperimentDefinition, ExperimentPolicy
from ux_analyzer.domain.interface import ViewportSnapshot
from ux_analyzer.domain.run import ProviderManifest, RunSpec
from ux_analyzer.ports.artifacts import (
    BundleManifest,
    RedactionPolicy,
    RunBundleWriter,
)
from ux_analyzer.ports.models import StructuredModelClient
from ux_analyzer.ports.observation import (
    ObservationCapture,
    ObservationProvider,
    ObservationSessionConfig,
    PlatformAction,
    PlatformActionResult,
    SessionHandle,
    TestAccountId,
    ViewportSize,
)
from ux_analyzer.providers.attention_policy import (
    AttentionPolicyConfig,
    ObservationSelection,
    ProgressiveAttentionPolicy,
)
from ux_analyzer.providers.cognitive import StructuredCognitiveAgent
from ux_analyzer.providers.finding_rules import FindingRuleSet
from ux_analyzer.providers.full_list_policy import FullListPolicy
from ux_analyzer.providers.prominence import HeuristicProminenceProvider
from ux_analyzer.providers.ranked_list_policy import ProminenceRankedListPolicy
from ux_analyzer.providers.scent import (
    StructuredCoarseScentEvaluator,
    StructuredFullScentEvaluator,
)
from ux_analyzer.reporting.renderer import render_experiment_report
from ux_analyzer.storage.run_bundle import FilesystemRunBundleWriter

app = typer.Typer(add_completion=False)
fixture_app = typer.Typer(add_completion=False)
app.add_typer(fixture_app, name="fixture")

_MODEL_CALLS_BY_POLICY = {
    ExperimentPolicy.FULL_LIST.value: 1,
    ExperimentPolicy.PROMINENCE_RANKED_LIST.value: 1,
    ExperimentPolicy.PROGRESSIVE_PROMINENCE.value: 1,
    ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT.value: 3,
}


@dataclass(frozen=True, slots=True)
class _ResolvedMatrix:
    loaded: LoadedProject
    definition: ExperimentDefinition
    specs: tuple[RunSpec, ...]


@app.callback()
def main() -> None:
    """Run UX analyzer commands."""

    load_environment_file()


@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(f"uxa {__version__}")


@app.command()
def validate(
    project: Path,
    check_env: bool = typer.Option(
        False,
        "--check-env",
        help="Check required model environment names without printing values.",
    ),
) -> None:
    """Validate one benchmark project configuration."""
    loaded = _load_project_or_exit(project)
    typer.echo(
        f"valid project: {loaded.project.id} "
        f"({len(loaded.project.scenarios)} scenarios, "
        f"{len(loaded.project.personas)} personas, "
        f"{len(loaded.project.experiments)} experiments)"
    )
    typer.echo(f"config digest: {loaded.config_digest}")
    if check_env:
        _model_settings_or_exit()


@fixture_app.command("serve")
def fixture_serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Serve bundled controlled fixture application."""
    import uvicorn

    try:
        bind_host = _loopback_bind_host(host)
    except ValueError as error:
        _exit_with_error(str(error))
    uvicorn.run("fixture_app.app:app", host=bind_host, port=port)


@app.command()
def run(
    project: Path,
    experiment: str = typer.Option("core-pair", "--experiment"),
    output: Path = typer.Option(Path(".uxa-output"), "--output"),
    workers: int = typer.Option(1, "--workers"),
    run_count: int | None = typer.Option(None, "--run-count"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    check_env: bool = typer.Option(False, "--check-env"),
    fixture_origin: str = typer.Option("http://127.0.0.1:8000", "--fixture-origin"),
) -> None:
    """Expand and execute one benchmark experiment."""
    _run_experiment_command(
        project=project,
        experiment_id=experiment,
        output=output,
        workers=workers,
        run_count=run_count,
        policies=(),
        dry_run=dry_run,
        check_env=check_env,
        fixture_origin=fixture_origin,
    )


@app.command()
def ablate(
    project: Path,
    experiment: str = typer.Option("ablations", "--experiment"),
    output: Path = typer.Option(Path(".uxa-output"), "--output"),
    workers: int = typer.Option(1, "--workers"),
    run_count: int | None = typer.Option(None, "--run-count"),
    policy: list[str] = typer.Option([], "--policy"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    check_env: bool = typer.Option(False, "--check-env"),
    fixture_origin: str = typer.Option("http://127.0.0.1:8000", "--fixture-origin"),
) -> None:
    """Execute selected attention policy ablations."""
    _run_experiment_command(
        project=project,
        experiment_id=experiment,
        output=output,
        workers=workers,
        run_count=run_count,
        policies=tuple(policy),
        dry_run=dry_run,
        check_env=check_env,
        fixture_origin=fixture_origin,
    )


@app.command()
def report(
    bundle_root: Path,
    output: Path = typer.Option(Path("report.html"), "--output"),
) -> None:
    """Regenerate static report from finalized run bundles."""
    try:
        rendered = render_experiment_report(bundle_root, output)
    except (FileNotFoundError, OSError, ValueError) as error:
        _exit_with_error(f"report failed: {error}")
    typer.echo(f"report generated: {rendered}")


@app.command("inspect-run")
def inspect_run(run_path: Path) -> None:
    """Print terminal outcome and artifact paths for one run bundle."""
    if not run_path.is_dir():
        _exit_with_error(f"run bundle does not exist: {run_path}")
    manifest = _read_json_or_exit(run_path / "manifest.json")
    result = _read_json_or_exit(run_path / "result.json", required=False)
    events = _read_events(run_path / "timeline.jsonl")
    run_id = _text(manifest.get("run_id"), run_path.name)
    outcome = _outcome(result, events)
    verification = _mapping(result.get("verification"))
    if not verification:
        verification = _last_mapping(events, "verification-recorded", "result")
    artifacts = _artifact_paths(run_path)
    typer.echo(f"run: {run_id}")
    typer.echo(f"outcome: {outcome}")
    typer.echo(f"verified: {bool(verification.get('verified', False))}")
    typer.echo(f"claimed: {bool(result.get('agent_claimed_success', False))}")
    typer.echo("artifacts:")
    for artifact in artifacts:
        typer.echo(f"- {artifact}")


def _run_experiment_command(
    *,
    project: Path,
    experiment_id: str,
    output: Path,
    workers: int,
    run_count: int | None,
    policies: Sequence[str],
    dry_run: bool,
    check_env: bool,
    fixture_origin: str,
) -> None:
    if workers <= 0:
        _exit_with_error("workers must be greater than zero")
    matrix = _resolve_matrix_or_exit(
        project,
        experiment_id,
        run_count=run_count,
        policies=policies,
    )
    settings: OpenAICompatibleSettings | None = None
    if check_env or not dry_run:
        settings = _model_settings_or_exit()
    _print_matrix(matrix, workers=workers)
    if dry_run:
        return
    if settings is None:
        _exit_with_error("model settings are required for execution")
    try:
        result = asyncio.run(
            _execute_matrix(
                matrix,
                output=output,
                workers=workers,
                fixture_origin=fixture_origin,
                settings=_settings_with_fixture_redaction(settings, matrix.loaded),
            )
        )
    except Exception as error:
        _exit_with_error(f"run failed: {error}")
    typer.echo(
        f"completed runs: {len(result.results)}; failures: {len(result.failures)}"
    )
    summary_path, report_path = _complete_experiment(
        result,
        output=output,
        runtime=matrix.loaded.runtime,
    )
    typer.echo(f"evaluation summary: {summary_path}")
    typer.echo(f"report generated: {report_path}")
    if result.failures or _evaluation_failures(result.results):
        raise typer.Exit(1)


def _resolve_matrix_or_exit(
    project_path: Path,
    experiment_id: str,
    *,
    run_count: int | None,
    policies: Sequence[str],
) -> _ResolvedMatrix:
    loaded = _load_project_or_exit(project_path)
    definition = next(
        (item for item in loaded.project.experiments if item.id == experiment_id),
        None,
    )
    if definition is None:
        available = ", ".join(item.id for item in loaded.project.experiments)
        _exit_with_error(
            f"unknown experiment {experiment_id!r}; available experiments: {available}"
        )
    if run_count is not None:
        if run_count <= 0:
            _exit_with_error("run-count must be greater than zero")
        definition = replace(definition, run_count=run_count, seeds=())
    if policies:
        selected: list[ExperimentPolicy] = []
        for policy in policies:
            try:
                selected.append(ExperimentPolicy(policy))
            except ValueError:
                _exit_with_error(
                    f"unsupported policy {policy!r}; choose one of: "
                    + ", ".join(item.value for item in ExperimentPolicy)
                )
        definition = replace(definition, policies=tuple(selected))
    try:
        specs = expand_experiment(
            ExperimentContext(
                definition=definition,
                project=loaded.project,
                config_digest=loaded.config_digest,
            )
        )
    except ValueError as error:
        _exit_with_error(f"cannot expand experiment {experiment_id!r}: {error}")
    return _ResolvedMatrix(loaded=loaded, definition=definition, specs=specs)


def _print_matrix(matrix: _ResolvedMatrix, *, workers: int) -> None:
    policy_names = tuple(policy.value for policy in matrix.definition.policies)
    seeds = tuple(sorted({spec.seed for spec in matrix.specs}))
    calls = sum(_MODEL_CALLS_BY_POLICY[spec.policy.value] for spec in matrix.specs)
    cells = Counter(
        (
            spec.scenario.id,
            spec.application_version.id,
            spec.persona.id,
            spec.policy.value,
        )
        for spec in matrix.specs
    )
    typer.echo(f"project: {matrix.loaded.project.id}")
    typer.echo(f"experiment: {matrix.definition.id}")
    typer.echo(f"policies: {', '.join(policy_names)}")
    typer.echo(f"workers: {workers}")
    typer.echo(f"seeds per cell: {len(seeds)}")
    typer.echo(f"run specs: {len(matrix.specs)}")
    typer.echo(f"estimated model calls: {calls}")
    typer.echo("matrix:")
    for (scenario, version, persona, policy), count in sorted(cells.items()):
        typer.echo(f"- {scenario}/{version}/{persona}/{policy}: {count} runs")


def _model_settings_or_exit() -> OpenAICompatibleSettings:
    try:
        settings = OpenAICompatibleSettings.from_env()
    except ModelConfigurationError as error:
        typer.echo(f"model environment error: {error}")
        raise typer.Exit(1) from error
    typer.echo(
        "model environment: configured "
        f"(endpoint origin: {settings.endpoint_origin}; "
        "API key present; scent and cognitive models configured)"
    )
    return settings


def _settings_with_fixture_redaction(
    settings: OpenAICompatibleSettings, loaded: LoadedProject
) -> OpenAICompatibleSettings:
    values = tuple(
        scenario.fixture_inputs.values[key]
        for scenario in loaded.project.scenarios
        for key in sorted(scenario.fixture_inputs.sensitive_keys)
    )
    return replace(settings, redaction_values=values)


def _load_project_or_exit(path: Path) -> LoadedProject:
    try:
        return load_project(path)
    except (ProjectConfigError, OSError) as error:
        _exit_with_error(f"validation error: {error}")


def _exit_with_error(message: str) -> NoReturn:
    typer.echo(message)
    raise typer.Exit(1)


def _read_json_or_exit(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            _exit_with_error(f"required run artifact missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _exit_with_error(f"invalid JSON artifact {path}: {error}")
    if not isinstance(value, dict):
        _exit_with_error(f"JSON artifact must contain object: {path}")
    return cast(dict[str, Any], value)


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(cast(dict[str, Any], value))
    return events


def _outcome(result: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> str:
    result_outcome = _mapping(result.get("outcome"))
    if result_outcome.get("kind"):
        return _text(result_outcome["kind"])
    for event in reversed(events):
        if event.get("kind") == "run-terminated":
            outcome = _mapping(event.get("outcome"))
            if outcome.get("kind"):
                return _text(outcome["kind"])
    return "unknown"


def _last_mapping(
    events: Sequence[Mapping[str, Any]], event_kind: str, field: str
) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("kind") == event_kind:
            return _mapping(event.get(field))
    return {}


def _artifact_paths(run_path: Path) -> tuple[str, ...]:
    checksums = run_path / "checksums.sha256"
    if checksums.is_file():
        paths: list[str] = []
        for line in checksums.read_text(encoding="utf-8").splitlines():
            if "  " in line:
                _, path = line.split("  ", maxsplit=1)
                if path.startswith("artifacts/"):
                    paths.append(path)
        return tuple(paths)
    artifacts = run_path / "artifacts"
    if not artifacts.is_dir():
        return ()
    return tuple(
        path.relative_to(run_path).as_posix()
        for path in sorted(artifacts.rglob("*"))
        if path.is_file()
    )


class _FixtureObservationProvider:
    """Add deterministic extraction and fixture reset around web sessions."""

    id = "playwright-web"
    platform: Literal["web"] = "web"
    version = "fixture-web-v1"

    def __init__(
        self,
        adapter: PlaywrightSessionAdapter,
        fixture_origin: str,
        fixture_inputs: Mapping[str, str],
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._adapter = adapter
        self._fixture_origin = fixture_origin.rstrip("/")
        self._fixture_inputs = dict(fixture_inputs)
        self._http_client = http_client

    async def start_session(self, config: ObservationSessionConfig) -> SessionHandle:
        return await self._adapter.start_session(config)

    async def capture(self, session: SessionHandle) -> ObservationCapture:
        capture = await self._adapter.capture(session)
        snapshot = await capture_snapshot(
            self._adapter.page_for_testing(session), capture.viewport_id
        )
        return replace(capture, snapshot=snapshot)

    async def execute(
        self, session: SessionHandle, action: PlatformAction
    ) -> PlatformActionResult:
        return await self._adapter.execute(session, action)

    async def reset(self, session: SessionHandle) -> None:
        if self._http_client is not None:
            response = await self._http_client.post(
                f"{self._fixture_origin}/__control/reset",
                json={"session_id": session.session_id, "inputs": self._fixture_inputs},
            )
            response.raise_for_status()
        else:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self._fixture_origin}/__control/reset",
                    json={
                        "session_id": session.session_id,
                        "inputs": self._fixture_inputs,
                    },
                )
                response.raise_for_status()
        await self._adapter.reset(session)

    async def end_session(self, session: SessionHandle) -> None:
        try:
            await self._adapter.end_session(session)
        finally:
            if self._http_client is not None:
                response = await self._http_client.delete(
                    f"{self._fixture_origin}/__control/session/{quote(session.session_id, safe='')}"
                )
                response.raise_for_status()
            else:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.delete(
                        f"{self._fixture_origin}/__control/session/"
                        f"{quote(session.session_id, safe='')}"
                    )
                    response.raise_for_status()


class _AttentionPolicyAdapter:
    """Normalize unrestricted list policy output to bounded domain observations."""

    def __init__(self, policy: object) -> None:
        self._policy = policy

    @property
    def id(self) -> str:
        return str(getattr(self._policy, "id", type(self._policy).__name__))

    @property
    def version(self) -> str:
        return str(getattr(self._policy, "version", "unknown"))

    def next_observation(
        self,
        state: object,
        snapshot: ViewportSnapshot,
        scores: Sequence[object],
        coarse_scent: object,
        rng: Any,
    ) -> ObservationSelection:
        method = getattr(self._policy, "next_observation")
        selected = method(state, snapshot, scores, coarse_scent, rng)
        observation = getattr(selected, "observation", None)
        if isinstance(observation, (ProgressiveObservation, CompleteObservation)):
            return cast(ObservationSelection, selected)
        selected_ids = tuple(
            element_id
            for element_id in getattr(selected, "selected_ids", ())
            if element_id not in getattr(state, "noticed_ids", ())
        )[:3]
        if not selected_ids:
            raise ValueError("no unobserved visible elements remain")
        progressive = ProgressiveObservation.from_snapshot(
            snapshot, newly_revealed_ids=selected_ids
        )
        return ObservationSelection(
            observation=progressive,
            region_id=None,
            element_probabilities={},
            region_probabilities={None: 1.0},
            selection_mode=str(getattr(selected, "selection_mode", "list")),
        )


def _attention_policy_for(
    policy: ExperimentPolicy | str,
    config: AttentionPolicyConfig | None = None,
) -> object:
    selected = ExperimentPolicy(policy)
    if selected is ExperimentPolicy.FULL_LIST:
        return FullListPolicy()
    if selected is ExperimentPolicy.PROMINENCE_RANKED_LIST:
        return ProminenceRankedListPolicy()
    if selected is ExperimentPolicy.PROGRESSIVE_PROMINENCE:
        settings = config or AttentionPolicyConfig()
        return ProgressiveAttentionPolicy(replace(settings, coarse_scent_weight=0.0))
    if selected is ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT:
        return ProgressiveAttentionPolicy(config or AttentionPolicyConfig())
    raise ValueError(f"unsupported experiment policy: {selected.value}")


class _BundleFactory:
    def __init__(
        self,
        output: Path,
        settings: OpenAICompatibleSettings,
        runtime: RuntimeConfig,
    ) -> None:
        self._output = output
        self._settings = settings
        self._runtime = runtime

    def start(self, spec: RunSpec) -> RunBundleWriter:
        scent_enabled = spec.policy is ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT
        model_manifests = [
            ProviderManifest(
                provider_id="openai-compatible-structured",
                role="cognitive",
                model_id=self._settings.cognitive_model,
                endpoint_origin=self._settings.endpoint_origin,
                version="openai-compatible-v1",
                prompt_version="cognitive-v1",
                schema_version="cognitive-v1",
            )
        ]
        if scent_enabled:
            model_manifests.extend(
                ProviderManifest(
                    provider_id="openai-compatible-structured",
                    role=role,
                    model_id=self._settings.scent_model,
                    endpoint_origin=self._settings.endpoint_origin,
                    version="openai-compatible-v1",
                    prompt_version=prompt_version,
                    schema_version=schema_version,
                )
                for role, prompt_version, schema_version in (
                    ("coarse-scent", "scent-coarse-v1", "scent-coarse-v1"),
                    ("full-scent", "scent-full-v1", "scent-full-v1"),
                )
            )
        manifest = BundleManifest.from_run_spec(
            spec,
            endpoint_origin=self._settings.endpoint_origin,
            model_ids={
                **({"scent": self._settings.scent_model} if scent_enabled else {}),
                "cognitive": self._settings.cognitive_model,
            },
            prompt_versions={
                **(
                    {
                        "coarse-scent": "scent-coarse-v1",
                        "full-scent": "scent-full-v1",
                    }
                    if scent_enabled
                    else {}
                ),
                "cognitive": "cognitive-v1",
            },
            provider_versions={
                "observation": "fixture-web-v1",
                "models": "openai-compatible-v1",
                "prominence": self._runtime.prominence.version,
                "attention": self._runtime.attention.version,
                "discovery_cost": self._runtime.discovery_cost.version,
                "findings": self._runtime.findings.version,
                "state_updates": self._runtime.state_updates.version,
                "expectation": "disabled",
            },
            provider_manifests=tuple(model_manifests),
        )
        return FilesystemRunBundleWriter.start(
            self._output,
            manifest,
            redaction=RedactionPolicy.from_fixture_inputs(spec.scenario.fixture_inputs),
        )


async def _execute_matrix(
    matrix: _ResolvedMatrix,
    *,
    output: Path,
    workers: int,
    fixture_origin: str,
    settings: OpenAICompatibleSettings,
) -> ExperimentResult:
    from playwright.async_api import async_playwright

    origin = _fixture_origin(fixture_origin)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        client_http = httpx.AsyncClient(timeout=settings.timeout_seconds)
        fixture_http = httpx.AsyncClient(timeout=30.0)
        adapter = PlaywrightSessionAdapter(
            browser=browser,
            allowed_origins=BrowserAllowedOrigins.fixture_only([origin]),
            trace_directory=output / "traces",
        )
        try:

            def factory(spec: RunSpec) -> RunAgent:
                client = OpenAICompatibleStructuredClient(
                    settings, http_client=client_http
                )
                return _build_agent(
                    spec,
                    adapter=adapter,
                    client=client,
                    output=output,
                    fixture_origin=origin,
                    settings=settings,
                    runtime=matrix.loaded.runtime,
                    fixture_http_client=fixture_http,
                )

            return await ExperimentRunner(factory).run(matrix.specs, workers=workers)
        finally:
            await adapter.close()
            await client_http.aclose()
            await fixture_http.aclose()
            await browser.close()


def _build_agent(
    spec: RunSpec,
    *,
    adapter: PlaywrightSessionAdapter,
    client: OpenAICompatibleStructuredClient,
    output: Path,
    fixture_origin: str,
    settings: OpenAICompatibleSettings,
    runtime: RuntimeConfig,
    fixture_http_client: httpx.AsyncClient | None = None,
) -> RunAgent:
    provider = _FixtureObservationProvider(
        adapter,
        fixture_origin,
        spec.scenario.fixture_inputs.values,
        fixture_http_client,
    )
    attention_config = replace(
        runtime.attention,
        temperature=spec.persona.attention_temperature,
    )
    policy = _AttentionPolicyAdapter(
        _attention_policy_for(spec.policy, attention_config)
    )
    scent_enabled = spec.policy is ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT
    coarse = (
        StructuredCoarseScentEvaluator(
            cast(StructuredModelClient, client), model=settings.scent_model
        )
        if scent_enabled
        else None
    )
    full = (
        StructuredFullScentEvaluator(
            cast(StructuredModelClient, client), model=settings.scent_model
        )
        if scent_enabled
        else None
    )
    cognitive = StructuredCognitiveAgent(
        cast(StructuredModelClient, client),
        model=settings.cognitive_model,
        fixture_keys=tuple(sorted(spec.scenario.fixture_inputs.values)),
    )
    verifier = WebVerifier(
        spec.scenario.verifier,
        fixture_inputs=spec.scenario.fixture_inputs,
        fixture_state_client=HttpFixtureStateClient(
            fixture_origin,
            http_client=fixture_http_client,
            timeout_seconds=30.0,
        ),
        observation_provider=cast(ObservationProvider, provider),
        snapshot_extractor=_snapshot_from_capture,
    )
    return RunAgent(
        observation_provider=cast(ObservationProvider, provider),
        prominence_provider=cast(
            ProminenceProvider, HeuristicProminenceProvider(runtime.prominence)
        ),
        attention_policy=cast(AttentionPolicy, policy),
        cognitive_agent=cast(RunCognitiveAgent, cognitive),
        verifier=verifier,
        bundle_factory=_BundleFactory(output, settings, runtime),
        session_config_factory=lambda current_spec: _session_config(
            current_spec, output=output, fixture_origin=fixture_origin
        ),
        coarse_scent_evaluator=coarse,
        full_scent_evaluator=full,
        state_update_config=replace(
            runtime.state_updates,
            abandonment_threshold=spec.persona.abandonment_threshold,
        ),
        model_record_source=client,
        result_evaluator=lambda result: _evaluate_result(result, runtime),
    )


def _evaluate_result(result: object, runtime: RuntimeConfig):
    from ux_analyzer.application.run_agent import RunResult

    if not isinstance(result, RunResult):
        raise TypeError("run evaluator requires RunResult")
    metrics = evaluate_run(
        result,
        evaluation_target_for(result),
        inputs=evaluation_inputs_for(result),
        cost_config=runtime.discovery_cost,
    )
    findings = FindingRuleSet.default(runtime.findings).evaluate(metrics)
    return replace(result, metrics=metrics, findings=findings)


def _complete_experiment(
    result: ExperimentResult,
    *,
    output: Path,
    runtime: RuntimeConfig,
) -> tuple[Path, Path]:
    del runtime
    completed: list[RunResult] = []
    for item in result.results:
        if not isinstance(item, RunResult):
            raise TypeError("experiment result contains non-RunResult value")
        completed.append(item)
    run_results = tuple(completed)
    evaluable = tuple(item for item in run_results if item.metrics is not None)
    evaluation = evaluate_experiment_results(evaluable)
    failures = [
        *(_experiment_failure_record(item) for item in result.failures),
        *(
            _evaluation_failure_record(item)
            for item in _evaluation_failures(run_results)
        ),
    ]
    summary = {
        "run_metrics": evaluation.run_metrics,
        "cell_aggregates": evaluation.cell_aggregates,
        "variant_comparisons": evaluation.variant_comparisons,
        "findings": {item.run_id: item.findings or () for item in run_results},
        "failures": failures,
    }
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "experiment.json"
    temporary = output / ".experiment.json.tmp"
    temporary.write_text(
        json.dumps(
            _json_data(summary),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    report_path = render_experiment_report(output, output / "report.html")
    return summary_path, report_path


def _evaluation_failures(results: Sequence[object]) -> tuple[RunResult, ...]:
    return tuple(
        item
        for item in results
        if isinstance(item, RunResult) and item.evaluation_failure_reason is not None
    )


def _experiment_failure_record(failure: ExperimentFailure) -> dict[str, object]:
    spec = failure.spec
    return {
        "run_id": failure.run_id,
        "error_type": failure.error_type,
        "stage": _failure_stage(failure.error_type),
        "terminal_state": "failed",
        "reason": _safe_failure_reason(failure.message, spec),
        "scenario_id": spec.scenario.id,
        "application_version_id": spec.application_version.id,
        "persona_id": spec.persona.id,
        "policy": spec.policy.value,
        "seed": spec.seed,
    }


def _evaluation_failure_record(result: RunResult) -> dict[str, object]:
    spec = result.state.spec
    reason = result.evaluation_failure_reason or "result evaluation failed"
    return {
        "run_id": result.run_id,
        "error_type": "EvaluationFailure",
        "stage": "evaluation",
        "terminal_state": "finalized",
        "reason": _safe_failure_reason(reason, spec),
        "scenario_id": spec.scenario.id,
        "application_version_id": spec.application_version.id,
        "persona_id": spec.persona.id,
        "policy": spec.policy.value,
        "seed": spec.seed,
    }


def _failure_stage(error_type: str) -> str:
    return (
        "bundle-finalization" if error_type == "RunFinalizationError" else "execution"
    )


def _safe_failure_reason(message: str, spec: RunSpec) -> str:
    safe = message.strip() or "run failed"
    for value in spec.scenario.fixture_inputs.values.values():
        if value:
            safe = safe.replace(value, "[REDACTED]")
    return safe


def _json_data(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _json_data(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = cast(Sequence[object], value)
        return [_json_data(item) for item in sequence]
    if is_dataclass(value):
        return {
            item.name: _json_data(getattr(value, item.name)) for item in fields(value)
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_data(model_dump(mode="json"))
    raise TypeError(f"cannot serialize experiment value {type(value)!r}")


def _session_config(
    spec: RunSpec, *, output: Path, fixture_origin: str
) -> ObservationSessionConfig:
    page = (
        ""
        if spec.scenario.start_state == "dashboard"
        else f"/{quote(spec.scenario.start_state)}"
    )
    version = spec.application_version.kind.value
    return ObservationSessionConfig(
        session_id=spec.run_id,
        start_url=f"{fixture_origin}/app/{quote(spec.run_id)}/{version}{page}",
        test_account_id=TestAccountId(f"test-{spec.run_id.removeprefix('run-')}"),
        viewport=ViewportSize(
            width=spec.scenario.viewport_width,
            height=spec.scenario.viewport_height,
        ),
        trace_path=output / "traces" / f"{spec.run_id}.zip",
        artifact_redaction=RedactionPolicy.from_fixture_inputs(
            spec.scenario.fixture_inputs
        ),
    )


def _snapshot_from_capture(capture: ObservationCapture) -> ViewportSnapshot:
    if capture.snapshot is None:
        raise RuntimeError("fixture capture has no normalized snapshot")
    return capture.snapshot


def _fixture_origin(value: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("fixture origin must be an HTTP(S) origin")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("fixture origin must not contain credentials")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("fixture origin must not contain path or query")
    return f"{parsed.scheme}://{parsed.netloc}"


def _loopback_bind_host(value: str) -> str:
    host = value.strip().lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "fixture host must be loopback-only: use 127.0.0.1, localhost, or ::1"
        )
    return host


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, Any], value)


def _text(value: object, default: str = "") -> str:
    return default if value is None else str(value)
