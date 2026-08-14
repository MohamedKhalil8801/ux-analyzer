"""Operator command surface for benchmark validation and execution."""

from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NoReturn, cast
from urllib.parse import quote, urlsplit

import httpx
import typer
from pydantic import TypeAdapter

from ux_analyzer import __version__
from ux_analyzer.adapters.openai import (
    ModelConfigurationError,
    OpenAICompatibleSettings,
    create_structured_model_client,
    load_environment_file,
)
from ux_analyzer.adapters.saliency.foveacast import FoveacastSaliencyProvider
from ux_analyzer.adapters.web.extractor import (
    capture_with_diagnostics as capture_snapshot_with_diagnostics,
)
from ux_analyzer.adapters.web.network_policy import BrowserAllowedOrigins
from ux_analyzer.adapters.web.session import PlaywrightSessionAdapter
from ux_analyzer.adapters.web.verifier import HttpFixtureStateClient, WebVerifier
from ux_analyzer.application.checkpoint import (
    CheckpointError,
    ExperimentCheckpointStore,
    finalized_bundle_is_valid,
    read_finalized_bundle,
)
from ux_analyzer.application.evaluation import (
    RunMetrics,
    aggregate_cells,
    compare_variants,
    comparison_sample_is_valid,
    evaluate_experiment_results,
    evaluate_run,
    evaluation_inputs_for,
    evaluation_target_for,
    persisted_comparison_sample_is_valid,
)
from ux_analyzer.application.evidence_corpus import (
    EvidenceCorpus,
    EvidenceCorpusBuilder,
)
from ux_analyzer.application.experiment import (
    ExperimentContext,
    ExperimentFailure,
    ExperimentResult,
    ExperimentRunner,
    expand_experiment,
)
from ux_analyzer.application.report_synthesis import ReportSynthesisService
from ux_analyzer.application.run_agent import (
    AttentionPolicy,
    ModelRecordSource,
    ProminenceProvider,
    RunAgent,
    RunProfiler,
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
from ux_analyzer.domain.benchmark import (
    PROMINENCE_PROVIDER_REGISTRY,
    ApplicationVersionKind,
    ExperimentDefinition,
    ExperimentPolicy,
    resolve_prominence_provider_id,
)
from ux_analyzer.domain.interface import ViewportSnapshot
from ux_analyzer.domain.run import ProviderManifest, RunSpec
from ux_analyzer.domain.synthesis import SynthesisAttempt
from ux_analyzer.ports.artifacts import (
    BundleManifest,
    RedactionPolicy,
    RunBundleWriter,
)
from ux_analyzer.ports.models import ModelRole, StructuredModelClient
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
from ux_analyzer.providers.report_synthesis import (
    EvidenceAuditor,
    PatternReviewer,
    ReportAdjudicator,
    ReportAnalyst,
)
from ux_analyzer.providers.saliency_prominence import (
    AttentionStageSelector,
    FoveacastProminenceProvider,
)
from ux_analyzer.providers.scent import (
    StructuredCoarseScentEvaluator,
    StructuredFullScentEvaluator,
)
from ux_analyzer.providers.ux_principles import ux_principles
from ux_analyzer.reporting.renderer import render_experiment_report
from ux_analyzer.saliency.model_registry import (
    DEFAULT_MODEL_ID,
    DEFAULT_PRECISION,
    ModelRegistry,
    ModelRegistryError,
    load_manifest,
)
from ux_analyzer.storage.run_bundle import FilesystemRunBundleWriter
from ux_analyzer.storage.saliency_cache import SaliencyCache
from ux_analyzer.storage.synthesis_artifacts import (
    SynthesisArtifactError,
    SynthesisArtifactStore,
    synthesis_attempt_position,
)

app = typer.Typer(add_completion=False)
fixture_app = typer.Typer(add_completion=False)
models_app = typer.Typer(add_completion=False)
app.add_typer(fixture_app, name="fixture")
app.add_typer(models_app, name="models")

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


@dataclass(frozen=True, slots=True)
class _FinalizedRunReference:
    run_id: str
    bundle_path: Path


@app.callback()
def main() -> None:
    """Run UX analyzer commands."""

    live_tests_were_explicit = "UXA_RUN_LIVE_TESTS" in os.environ
    load_environment_file()
    if not live_tests_were_explicit:
        os.environ.pop("UXA_RUN_LIVE_TESTS", None)


@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(f"uxa {__version__}")


def _registry_for(model_id: str) -> ModelRegistry:
    """Load packaged registry manifest for one operator-selected model."""

    try:
        return ModelRegistry(manifest=load_manifest(model_id))
    except ModelRegistryError as error:
        _exit_with_error(f"unable to load model {model_id}: {error}")


@models_app.command("install")
def models_install(
    model_id: str = typer.Argument(DEFAULT_MODEL_ID),
    precision: str = typer.Option(DEFAULT_PRECISION, "--precision"),
) -> None:
    """Explicitly download and verify one saliency model release."""

    try:
        result = _registry_for(model_id).install(model_id, precision=precision)
    except ModelRegistryError as error:
        _exit_with_error(str(error))
    action = "downloaded" if result.downloaded else "already installed"
    typer.echo(f"{model_id}: {action}")
    if result.downloaded:
        typer.echo(f"artifacts: {', '.join(result.downloaded)}")
    typer.echo(f"license attribution: {result.attribution}")


@models_app.command("status")
def models_status(
    model_id: str = typer.Argument(DEFAULT_MODEL_ID),
    provider: str = typer.Option("cpu", "--provider"),
) -> None:
    """Inspect runtime and local saliency model artifacts without downloading."""

    try:
        status = _registry_for(model_id).status(model_id, provider=provider)
    except ModelRegistryError as error:
        _exit_with_error(str(error))
    typer.echo(f"{model_id}: {status.state.value}")
    for diagnostic in status.diagnostics[1:]:
        typer.echo(f"diagnostic: {diagnostic}")
    for artifact in status.artifacts:
        if artifact.state.value != "ready":
            typer.echo(f"{artifact.artifact.filename}: {artifact.state.value}")
    typer.echo(f"license attribution: {status.attribution}")


@models_app.command("remove")
def models_remove(
    model_id: str = typer.Argument(DEFAULT_MODEL_ID),
    precision: str = typer.Option(DEFAULT_PRECISION, "--precision"),
) -> None:
    """Explicitly remove one saliency model release from local storage."""

    try:
        result = _registry_for(model_id).remove(model_id, precision=precision)
    except ModelRegistryError as error:
        _exit_with_error(str(error))
    if result.removed:
        typer.echo(f"{model_id}: removed {len(result.removed)} artifacts")
    else:
        typer.echo(f"{model_id}: already absent")


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
        _model_settings_or_exit(
            report_synthesis_enabled=loaded.runtime.report_synthesis.enabled
        )


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
    output: Path = typer.Option(Path("reports"), "--output"),
    workers: int = typer.Option(1, "--workers"),
    run_count: int | None = typer.Option(None, "--run-count"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    check_env: bool = typer.Option(False, "--check-env"),
    fixture_origin: str = typer.Option("http://127.0.0.1:8000", "--fixture-origin"),
    resume: bool = typer.Option(False, "--resume"),
    no_synthesis: bool = typer.Option(
        False,
        "--no-synthesis",
        help="Skip configured post-run report synthesis.",
    ),
    profile_output: Path | None = typer.Option(
        None,
        "--profile-output",
        help="Write per-run stage timings as JSON files under this directory.",
    ),
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
        resume=resume,
        profile_output=profile_output,
        no_synthesis=no_synthesis,
    )


@app.command("run-one")
def run_one(
    project: Path,
    scenario: str = typer.Option(..., "--scenario"),
    version: str = typer.Option(..., "--version"),
    persona: str = typer.Option(..., "--persona"),
    policy: str = typer.Option(..., "--policy"),
    seed: int = typer.Option(..., "--seed"),
    prominence_provider_id: str = typer.Option(
        "heuristic", "--prominence-provider", "--prominence-provider-id"
    ),
    output: Path = typer.Option(Path("reports"), "--output"),
    fixture_origin: str = typer.Option("http://127.0.0.1:8000", "--fixture-origin"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    check_env: bool = typer.Option(False, "--check-env"),
    profile_output: Path | None = typer.Option(
        None,
        "--profile-output",
        help="Write this run's stage timings as JSON under this directory.",
    ),
) -> None:
    """Execute exactly one semantically selected benchmark run."""

    matrix = _resolve_single_run(
        project,
        scenario_id=scenario,
        version_id=version,
        persona_id=persona,
        policy=policy,
        seed=seed,
        prominence_provider_id=prominence_provider_id,
    )
    settings: OpenAICompatibleSettings | None = None
    if check_env or not dry_run:
        settings = _model_settings_or_exit()
    _print_matrix(
        matrix,
        workers=1,
        max_concurrent_calls=(
            settings.max_concurrent_calls if settings is not None else None
        ),
    )
    if dry_run:
        return
    if settings is None:
        _exit_with_error("model settings are required for execution")
    try:
        result = asyncio.run(
            _execute_matrix(
                matrix,
                output=output,
                workers=1,
                fixture_origin=fixture_origin,
                settings=_settings_with_fixture_redaction(settings, matrix.loaded),
                profile_output=profile_output,
            )
        )
    except Exception as error:
        _exit_with_error(f"run failed: {error}")
    _print_result_summary(result)
    summary_path, report_path = _complete_experiment(
        result,
        output=output,
        runtime=matrix.loaded.runtime,
    )
    typer.echo(f"evaluation summary: {summary_path}")
    typer.echo(f"report generated: {report_path}")
    if profile_output is not None:
        typer.echo(f"profile output: {profile_output}")
    if (
        result.failures
        or _evaluation_failures(result.results)
        or _invalid_ux_samples(result.results)
    ):
        raise typer.Exit(1)


@app.command()
def ablate(
    project: Path,
    experiment: str = typer.Option("ablations", "--experiment"),
    output: Path = typer.Option(Path("reports"), "--output"),
    workers: int = typer.Option(1, "--workers"),
    run_count: int | None = typer.Option(None, "--run-count"),
    policy: list[str] = typer.Option([], "--policy"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    check_env: bool = typer.Option(False, "--check-env"),
    fixture_origin: str = typer.Option("http://127.0.0.1:8000", "--fixture-origin"),
    resume: bool = typer.Option(False, "--resume"),
    profile_output: Path | None = typer.Option(
        None,
        "--profile-output",
        help="Write per-run stage timings as JSON files under this directory.",
    ),
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
        resume=resume,
        profile_output=profile_output,
        no_synthesis=False,
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


@app.command()
def synthesize(
    project: Path,
    experiment: str = typer.Option(..., "--experiment"),
    output: Path = typer.Option(Path("reports"), "--output"),
) -> None:
    """Run report synthesis for finalized experiment evidence."""
    matrix = _resolve_matrix_or_exit(
        project,
        experiment,
        run_count=None,
        policies=(),
    )
    _read_json_or_exit(output / "experiment.json")
    try:
        result = _finalized_experiment_result(matrix, output)
        settings = _model_settings_or_exit(report_synthesis_enabled=True)
        attempt = asyncio.run(
            _run_report_synthesis(
                result=result,
                output=output,
                loaded=matrix.loaded,
                settings=settings,
            )
        )
        attempt_path = _persist_synthesis_attempt(
            attempt,
            result=result,
            output=output,
            loaded=matrix.loaded,
        )
        report_path = _render_completed_report(output=output)
    except Exception as error:
        _exit_with_error(f"synthesis failed: {type(error).__name__}: {error}")
    typer.echo(f"synthesis attempt: {attempt_path}")
    typer.echo(f"report generated: {report_path}")


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
    resume: bool,
    profile_output: Path | None,
    no_synthesis: bool,
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
        # Report synthesis is best-effort. Keep missing report-role settings
        # from preventing deterministic experiment execution and fallback rendering.
        settings = _model_settings_or_exit(
            report_synthesis_enabled=(
                check_env
                and matrix.loaded.runtime.report_synthesis.enabled
                and not no_synthesis
            )
        )
    _print_matrix(
        matrix,
        workers=workers,
        max_concurrent_calls=(
            settings.max_concurrent_calls if settings is not None else None
        ),
    )
    if dry_run:
        return
    if settings is None:
        _exit_with_error("model settings are required for execution")
    selected_matrix = matrix
    selected_specs = selected_matrix.specs
    try:
        if resume:
            matrix, checkpoint = _prepare_resumed_matrix(matrix, output)
            state = checkpoint.state
            typer.echo(
                "resume: "
                f"{len(state.finalized_run_ids)} finalized; "
                f"{len(state.interrupted_run_ids)} interrupted; "
                f"{len(state.pending_run_ids)} pending"
            )
        else:
            checkpoint = ExperimentCheckpointStore(
                output,
                tuple(spec.run_id for spec in matrix.specs),
                selected_prominence_provider_ids={
                    spec.run_id: spec.prominence_provider_id for spec in matrix.specs
                },
            )
            checkpoint.initialize()
    except CheckpointError as error:
        _exit_with_error(f"checkpoint error: {error}")
    try:
        result = asyncio.run(
            _execute_matrix(
                matrix,
                output=output,
                workers=workers,
                fixture_origin=fixture_origin,
                settings=_settings_with_fixture_redaction(settings, matrix.loaded),
                checkpoint=checkpoint,
                profile_output=profile_output,
            )
        )
    except Exception as error:
        _exit_with_error(f"run failed: {error}")
    _print_result_summary(result)
    summary_path = _write_experiment_summary(
        result,
        output=output,
        selected_specs=selected_specs,
    )
    if matrix.loaded.runtime.report_synthesis.enabled and not no_synthesis:
        synthesis_result = result
        try:
            if resume:
                synthesis_result = _finalized_experiment_result(selected_matrix, output)
            attempt = asyncio.run(
                _run_report_synthesis(
                    result=synthesis_result,
                    output=output,
                    loaded=selected_matrix.loaded,
                    settings=settings,
                )
            )
            _persist_synthesis_attempt(
                attempt,
                result=synthesis_result,
                output=output,
                loaded=selected_matrix.loaded,
            )
        except Exception as error:
            _persist_unavailable_synthesis(
                result=synthesis_result,
                output=output,
                loaded=selected_matrix.loaded,
            )
            typer.echo(
                "warning: report synthesis unavailable; "
                f"deterministic findings retained ({type(error).__name__})"
            )
    report_path = _render_completed_report(output=output)
    typer.echo(f"evaluation summary: {summary_path}")
    typer.echo(f"report generated: {report_path}")
    if profile_output is not None:
        typer.echo(f"profile output: {profile_output}")
    if (
        result.failures
        or _evaluation_failures(result.results)
        or _invalid_ux_samples(result.results)
    ):
        raise typer.Exit(1)


def _prepare_resumed_matrix(
    matrix: _ResolvedMatrix, output: Path
) -> tuple[_ResolvedMatrix, ExperimentCheckpointStore]:
    """Reconcile a selected matrix with integrity-valid finalized bundles."""

    store = ExperimentCheckpointStore(
        output,
        tuple(spec.run_id for spec in matrix.specs),
        selected_prominence_provider_ids={
            spec.run_id: spec.prominence_provider_id for spec in matrix.specs
        },
    )
    state = store.initialize(resume=True)
    finalized = set(state.finalized_run_ids)
    store.archive_interrupted_staging()
    return (
        replace(
            matrix,
            specs=tuple(spec for spec in matrix.specs if spec.run_id not in finalized),
        ),
        store,
    )


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
                config_digest=loaded.config_digest_for(experiment_id),
            )
        )
    except ValueError as error:
        _exit_with_error(f"cannot expand experiment {experiment_id!r}: {error}")
    return _ResolvedMatrix(loaded=loaded, definition=definition, specs=specs)


def _resolve_single_run(
    project_path: Path,
    *,
    scenario_id: str,
    version_id: str,
    persona_id: str,
    policy: str,
    seed: int,
    prominence_provider_id: str = "heuristic",
) -> _ResolvedMatrix:
    """Resolve semantic identifiers into one deterministic run specification."""

    loaded = load_project(project_path)
    scenario = next(
        (item for item in loaded.project.scenarios if item.id == scenario_id), None
    )
    if scenario is None:
        raise ValueError(f"unknown scenario {scenario_id!r}")
    versions = tuple(
        version
        for application in loaded.project.applications
        for version in application.versions
    )
    version = next(
        (
            item
            for item in versions
            if item.id == version_id or item.kind.value == version_id
        ),
        None,
    )
    if version is None:
        raise ValueError(f"unknown application version {version_id!r}")
    persona = next(
        (item for item in loaded.project.personas if item.id == persona_id), None
    )
    if persona is None:
        raise ValueError(f"unknown persona {persona_id!r}")
    try:
        selected_policy = ExperimentPolicy(policy)
    except ValueError as error:
        raise ValueError(f"unsupported policy {policy!r}") from error
    definition = ExperimentDefinition(
        id="single-run",
        name="Single selected run",
        scenario_ids=(scenario.id,),
        application_version_ids=(version.id,),
        persona_ids=(persona.id,),
        policies=(selected_policy,),
        seeds=(seed,),
        run_count=1,
        prominence_provider_ids=(prominence_provider_id,),
    )
    specs = expand_experiment(
        ExperimentContext(
            definition=definition,
            project=loaded.project,
            config_digest=loaded.config_digest,
        )
    )
    if len(specs) != 1:
        raise ValueError(f"single-run expansion produced {len(specs)} specs")
    return _ResolvedMatrix(loaded=loaded, definition=definition, specs=specs)


def _print_matrix(
    matrix: _ResolvedMatrix,
    *,
    workers: int,
    max_concurrent_calls: int | None = None,
) -> None:
    policy_names = tuple(policy.value for policy in matrix.definition.policies)
    provider_names = matrix.definition.prominence_provider_ids
    configured_seeds = matrix.definition.seeds or tuple(
        range(matrix.definition.run_count)
    )
    calls = sum(_MODEL_CALLS_BY_POLICY[spec.policy.value] for spec in matrix.specs)
    cells = Counter(
        (
            spec.scenario.id,
            spec.application_version.id,
            spec.persona.id,
            spec.policy.value,
            spec.prominence_provider_id,
        )
        for spec in matrix.specs
    )
    accounting_cells = Counter(
        (
            spec.scenario.id,
            spec.application_version.id,
            spec.persona.id,
            spec.policy.value,
            spec.prominence_provider_id,
            spec.model_trial,
        )
        for spec in matrix.specs
    )
    typer.echo(f"project: {matrix.loaded.project.id}")
    typer.echo(f"experiment: {matrix.definition.id}")
    typer.echo(f"policies: {', '.join(policy_names)}")
    typer.echo(f"prominence providers: {', '.join(provider_names)}")
    typer.echo(f"workers: {workers}")
    if max_concurrent_calls is not None:
        typer.echo(f"model call concurrency: {max_concurrent_calls}")
    typer.echo(f"configured seeds: {len(configured_seeds)}")
    typer.echo(f"run specs: {len(matrix.specs)}")
    eligible_cells = len(accounting_cells)
    unsuppressed_specs = eligible_cells * len(configured_seeds)
    suppressed = unsuppressed_specs - len(matrix.specs)
    typer.echo(f"deterministic seed repetitions suppressed: {suppressed}")
    typer.echo(f"model calls for one attention cycle: {calls}")
    typer.echo(
        "maximum logical model calls: "
        f"{sum(spec.scenario.budget.max_model_calls for spec in matrix.specs)}"
    )
    timeouts = {spec.scenario.budget.timeout_seconds for spec in matrix.specs}
    if timeouts == {None}:
        timeout_summary = "none"
    elif None in timeouts or len(timeouts) != 1:
        timeout_summary = "mixed"
    else:
        timeout_summary = f"{next(iter(timeouts)):g}s"
    typer.echo(f"overall run timeout: {timeout_summary}")
    typer.echo("matrix:")
    for (scenario, version, persona, policy, provider), count in sorted(cells.items()):
        typer.echo(
            f"- {scenario}/{version}/{persona}/{policy}: {count} runs "
            f"(prominence-provider={provider})"
        )


def _model_settings_or_exit(
    *, report_synthesis_enabled: bool = False
) -> OpenAICompatibleSettings:
    try:
        settings = OpenAICompatibleSettings.from_env(
            report_synthesis_enabled=report_synthesis_enabled
        )
    except ModelConfigurationError as error:
        typer.echo(f"model environment error: {error}")
        raise typer.Exit(1) from error
    if settings.mode == "codex":
        typer.echo(
            "model environment: configured "
            "(mode: codex; scent and cognitive models configured)"
        )
    else:
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
        *,
        fixture_control_enabled: bool = True,
    ) -> None:
        self._adapter = adapter
        self._fixture_origin = fixture_origin.rstrip("/")
        self._fixture_inputs = dict(fixture_inputs)
        self._http_client = http_client
        self._fixture_control_enabled = fixture_control_enabled
        self._profiler: RunProfiler | None = None

    def set_profiler(self, profiler: RunProfiler) -> None:
        self._profiler = profiler

    async def start_session(self, config: ObservationSessionConfig) -> SessionHandle:
        return await self._adapter.start_session(config)

    async def capture(self, session: SessionHandle) -> ObservationCapture:
        if self._profiler is None:
            capture = await self._adapter.capture(session)
            extraction = await capture_snapshot_with_diagnostics(
                self._adapter.page_for_testing(session),
                capture.viewport_id,
            )
            return replace(
                capture,
                screenshot=extraction.screenshot,
                snapshot=extraction.snapshot,
            )
        with self._profiler.measure("browser.capture"):
            capture = await self._adapter.capture(session)
        with self._profiler.measure("dom.extract"):
            extraction = await capture_snapshot_with_diagnostics(
                self._adapter.page_for_testing(session),
                capture.viewport_id,
                measure=self._profiler.measure,
            )
        return replace(
            capture,
            screenshot=extraction.screenshot,
            snapshot=extraction.snapshot,
        )

    async def execute(
        self, session: SessionHandle, action: PlatformAction
    ) -> PlatformActionResult:
        return await self._adapter.execute(session, action)

    async def reset(self, session: SessionHandle) -> None:
        if self._fixture_control_enabled:
            if self._http_client is not None:
                response = await self._http_client.post(
                    f"{self._fixture_origin}/__control/reset",
                    json={
                        "session_id": session.session_id,
                        "inputs": self._fixture_inputs,
                    },
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
            if self._fixture_control_enabled:
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

    def __init__(
        self,
        policy: object,
        *,
        config: AttentionPolicyConfig | None = None,
    ) -> None:
        self._policy = policy
        self.config = config or getattr(policy, "config", AttentionPolicyConfig())

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
        *,
        recovery_level: int = 0,
    ) -> ObservationSelection:
        method = getattr(self._policy, "next_observation")
        selected = method(
            state,
            snapshot,
            scores,
            coarse_scent,
            rng,
            recovery_level=recovery_level,
        )
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
        client: object | None = None,
    ) -> None:
        self._output = output
        self._settings = settings
        self._runtime = runtime
        self._client = client

    def start(self, spec: RunSpec) -> RunBundleWriter:
        resolve_prominence_provider_id(spec.prominence_provider_id)
        scent_enabled = spec.policy is ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT
        default_provider_id = (
            "codex-cli"
            if self._settings.mode == "codex"
            else "openai-compatible-structured"
        )
        default_provider_version = (
            "codex-cli" if self._settings.mode == "codex" else "openai-compatible-v1"
        )
        provider_id = str(getattr(self._client, "provider_id", default_provider_id))
        provider_version = str(
            getattr(self._client, "provider_version", default_provider_version)
        )
        prominence_version = self._runtime.prominence.version
        if spec.prominence_provider_id == "foveacast":
            prominence_version = PROMINENCE_PROVIDER_REGISTRY[
                spec.prominence_provider_id
            ]
        endpoint_origin = str(
            getattr(self._client, "endpoint_origin", self._settings.endpoint_origin)
        )
        model_manifests = [
            ProviderManifest(
                provider_id=provider_id,
                role="cognitive",
                model_id=self._settings.cognitive_model,
                endpoint_origin=endpoint_origin,
                version=provider_version,
                prompt_version="cognitive-v2",
                schema_version="cognitive-v1",
            )
        ]
        if scent_enabled:
            model_manifests.extend(
                ProviderManifest(
                    provider_id=provider_id,
                    role=role,
                    model_id=self._settings.scent_model,
                    endpoint_origin=endpoint_origin,
                    version=provider_version,
                    prompt_version=prompt_version,
                    schema_version=schema_version,
                )
                for role, prompt_version, schema_version in (
                    ("coarse-scent", "scent-coarse-v1", "scent-coarse-v1"),
                    ("full-scent", "scent-full-v1", "scent-full-v1"),
                )
            )
        if spec.prominence_provider_id == "foveacast":
            model_manifests.append(
                ProviderManifest(
                    provider_id="foveacast",
                    role="prominence",
                    model_id=self._runtime.saliency.model_set[0],
                    endpoint_origin="internal",
                    version="v0.2.0",
                )
            )
        manifest = BundleManifest.from_run_spec(
            spec,
            endpoint_origin=endpoint_origin,
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
                "cognitive": "cognitive-v2",
            },
            provider_versions={
                "observation": "fixture-web-v1",
                "models": provider_version,
                "prominence": prominence_version,
                "prominence-provider": spec.prominence_provider_id,
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
    checkpoint: ExperimentCheckpointStore | None = None,
    profile_output: Path | None = None,
) -> ExperimentResult:
    from playwright.async_api import async_playwright

    origin = _fixture_origin(fixture_origin)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        client_http = httpx.AsyncClient(timeout=settings.timeout_seconds)
        fixture_http = httpx.AsyncClient(timeout=30.0)
        model_call_limiter = asyncio.Semaphore(settings.max_concurrent_calls)
        adapter = PlaywrightSessionAdapter(
            browser=browser,
            allowed_origins=None,
            trace_directory=output / "traces",
        )
        try:

            def factory(spec: RunSpec) -> RunAgent:
                client = create_structured_model_client(
                    settings,
                    http_client=client_http,
                    call_limiter=model_call_limiter,
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
                    profile_output=profile_output,
                )

            def record_progress(
                spec: RunSpec, result: object | None, failure: ExperimentFailure | None
            ) -> None:
                if checkpoint is None:
                    return
                if failure is not None:
                    checkpoint.record_failure(spec.run_id, failure.error_type)
                elif result is not None and finalized_bundle_is_valid(
                    output,
                    spec.run_id,
                    expected_prominence_provider_id=spec.prominence_provider_id,
                ):
                    checkpoint.record_finalized(spec.run_id)
                else:
                    checkpoint.record_failure(spec.run_id, "InvalidFinalizedBundle")

            return await ExperimentRunner(factory).run(
                matrix.specs, workers=workers, on_complete=record_progress
            )
        finally:
            await adapter.close()
            await client_http.aclose()
            await fixture_http.aclose()
            await browser.close()


def _build_agent(
    spec: RunSpec,
    *,
    adapter: PlaywrightSessionAdapter,
    client: StructuredModelClient,
    output: Path,
    fixture_origin: str,
    settings: OpenAICompatibleSettings,
    runtime: RuntimeConfig,
    fixture_http_client: httpx.AsyncClient | None = None,
    profile_output: Path | None = None,
) -> RunAgent:
    resolve_prominence_provider_id(spec.prominence_provider_id)
    live_version = spec.application_version.kind is ApplicationVersionKind.LIVE
    fixture_inputs: Mapping[str, str] = (
        {} if live_version else spec.scenario.fixture_inputs.values
    )
    provider = _FixtureObservationProvider(
        adapter,
        fixture_origin,
        fixture_inputs,
        fixture_http_client,
        fixture_control_enabled=not live_version,
    )
    attention_config = replace(
        runtime.attention,
        temperature=spec.persona.attention_temperature,
    )
    policy = _AttentionPolicyAdapter(
        _attention_policy_for(spec.policy, attention_config),
        config=attention_config,
    )
    scent_enabled = spec.policy is ExperimentPolicy.PROGRESSIVE_PROMINENCE_SCENT
    coarse = (
        StructuredCoarseScentEvaluator(client, model=settings.scent_model)
        if scent_enabled
        else None
    )
    full = (
        StructuredFullScentEvaluator(client, model=settings.scent_model)
        if scent_enabled
        else None
    )
    cognitive = StructuredCognitiveAgent(
        client,
        model=settings.cognitive_model,
        fixture_keys=(
            () if live_version else tuple(sorted(spec.scenario.fixture_inputs.values))
        ),
    )
    fixture_state_client = (
        HttpFixtureStateClient(
            fixture_origin,
            http_client=fixture_http_client,
            timeout_seconds=30.0,
        )
        if not live_version
        else None
    )
    verifier = WebVerifier(
        spec.scenario.verifier,
        fixture_inputs=spec.scenario.fixture_inputs,
        fixture_state_client=fixture_state_client,
        observation_provider=cast(ObservationProvider, provider),
        snapshot_extractor=_snapshot_from_capture,
    )
    heuristic_prominence = HeuristicProminenceProvider(runtime.prominence)
    if spec.prominence_provider_id == "foveacast":
        model_set = runtime.saliency.model_set
        if len(model_set) != 1:
            raise ValueError(
                "Foveacast runtime currently requires exactly one configured model"
            )
        registry = _registry_for(model_set[0])
        redaction = RedactionPolicy.from_fixture_inputs(spec.scenario.fixture_inputs)
        prominence_provider: ProminenceProvider = cast(
            ProminenceProvider,
            FoveacastProminenceProvider(
                model_provider=lambda: FoveacastSaliencyProvider(
                    registry,
                    execution_provider_preference=(
                        runtime.saliency.execution_provider_preference
                    ),
                ),
                cache=lambda: SaliencyCache(output, redaction=redaction),
                cache_enabled=runtime.saliency.cache.enabled,
                cache_scope=runtime.saliency.cache.scope,
                model_set=model_set,
                precision=runtime.saliency.precision,
                execution_provider_preference=(
                    runtime.saliency.execution_provider_preference
                ),
                aggregation_config=runtime.saliency.aggregation,
                stage_selector=AttentionStageSelector(
                    version=runtime.saliency.stage_selector.version,
                    temperature=runtime.saliency.stage_selector.temperature,
                    mixtures=runtime.saliency.stage_selector.mixtures,
                ),
                heuristic_provider=heuristic_prominence,
                fallback_enabled=runtime.saliency.fallback.enabled,
                fallback_provider_id=runtime.saliency.fallback.provider_id,
            ),
        )
    else:
        prominence_provider = cast(ProminenceProvider, heuristic_prominence)
    return RunAgent(
        observation_provider=cast(ObservationProvider, provider),
        prominence_provider=prominence_provider,
        attention_policy=cast(AttentionPolicy, policy),
        cognitive_agent=cast(RunCognitiveAgent, cognitive),
        verifier=verifier,
        bundle_factory=_BundleFactory(output, settings, runtime, client=client),
        session_config_factory=lambda current_spec: _session_config(
            current_spec, output=output, fixture_origin=fixture_origin
        ),
        coarse_scent_evaluator=coarse,
        full_scent_evaluator=full,
        state_update_config=replace(
            runtime.state_updates,
            abandonment_threshold=spec.persona.abandonment_threshold,
        ),
        model_record_source=cast(ModelRecordSource, client),
        result_evaluator=lambda result: _evaluate_result(result, runtime),
        profile_path=(profile_output / f"{spec.run_id}.json")
        if profile_output is not None
        else None,
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


def _write_experiment_summary(
    result: ExperimentResult,
    *,
    output: Path,
    selected_specs: Sequence[RunSpec] | None,
) -> Path:
    completed: list[RunResult] = []
    for item in result.results:
        if not isinstance(item, RunResult):
            raise TypeError("experiment result contains non-RunResult value")
        completed.append(item)
    run_results = tuple(completed)
    if selected_specs is None:
        evaluable = tuple(item for item in run_results if item.metrics is not None)
        evaluation = evaluate_experiment_results(evaluable)
        run_metrics = evaluation.run_metrics
        grouping_specs = tuple(item.state.spec for item in run_results)
        cell_aggregates = _provider_cell_aggregates(run_metrics, grouping_specs)
        variant_comparisons = _compare_selected_variants(run_metrics, grouping_specs)
        findings_by_run: dict[str, object] = {
            item.run_id: item.findings or () for item in run_results
        }
    else:
        run_metrics, findings_by_run = _selected_finalized_evidence(
            output, selected_specs, run_results
        )
        cell_aggregates = _provider_cell_aggregates(run_metrics, selected_specs)
        variant_comparisons = _compare_selected_variants(run_metrics, selected_specs)
    failures = [
        *(_experiment_failure_record(item) for item in result.failures),
        *(
            _evaluation_failure_record(item)
            for item in _evaluation_failures(run_results)
        ),
    ]
    invalid_runs = [
        _invalid_ux_sample_record(item)
        for item in run_results
        if not item.ux_sample_valid
    ]
    summary = {
        "run_metrics": run_metrics,
        "cell_aggregates": cell_aggregates,
        "variant_comparisons": variant_comparisons,
        "findings": findings_by_run,
        "failures": failures,
        "invalid_runs": invalid_runs,
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
    return summary_path


def _render_completed_report(*, output: Path) -> Path:
    return render_experiment_report(output, output / "report.html")


def _synthesis_corpus(
    *,
    result: ExperimentResult,
    output: Path,
    loaded: LoadedProject,
) -> EvidenceCorpus:
    expectations = {
        expectation.key: expectation for expectation in loaded.runtime.expectations
    }
    return EvidenceCorpusBuilder().build(result, output, expectations)


async def _run_report_synthesis(
    *,
    result: ExperimentResult,
    output: Path,
    loaded: LoadedProject,
    settings: OpenAICompatibleSettings,
) -> SynthesisAttempt:
    corpus = _synthesis_corpus(result=result, output=output, loaded=loaded)
    http_client = httpx.AsyncClient(timeout=settings.timeout_seconds)
    try:
        client = create_structured_model_client(
            settings,
            http_client=http_client,
            call_limiter=asyncio.Semaphore(settings.max_concurrent_calls),
        )
        synthesis = loaded.runtime.report_synthesis
        service = ReportSynthesisService(
            analyst=ReportAnalyst(
                client, model=settings.model_for_role(ModelRole.REPORT_ANALYST)
            ),
            evidence_auditor=EvidenceAuditor(
                client,
                model=settings.model_for_role(ModelRole.REPORT_EVIDENCE_AUDITOR),
            ),
            pattern_reviewer=PatternReviewer(
                client,
                model=settings.model_for_role(ModelRole.REPORT_PATTERN_REVIEWER),
            ),
            adjudicator=ReportAdjudicator(
                client,
                model=settings.model_for_role(ModelRole.REPORT_ADJUDICATOR),
            ),
            principles=ux_principles(),
            max_retrieval_rounds=synthesis.max_retrieval_rounds,
            max_adjudication_revisions=synthesis.max_adjudication_revisions,
            max_final_verifications=synthesis.max_final_verifications,
            model_record_source=client,
        )
        return await service.synthesize(corpus)
    finally:
        await http_client.aclose()


def _finalized_experiment_result(
    matrix: _ResolvedMatrix,
    output: Path,
) -> ExperimentResult:
    output_root = Path(output)
    if not output_root.is_absolute():
        output_root = output_root.absolute()
    finalized_specs: list[RunSpec] = []
    references: list[_FinalizedRunReference] = []
    for spec in matrix.specs:
        bundle = output_root / "runs" / spec.run_id
        if not finalized_bundle_is_valid(
            output_root,
            spec.run_id,
            expected_prominence_provider_id=spec.prominence_provider_id,
        ):
            continue
        finalized_specs.append(spec)
        references.append(
            _FinalizedRunReference(run_id=spec.run_id, bundle_path=bundle)
        )
    if not finalized_specs:
        raise CheckpointError("no finalized runs available for synthesis")
    return ExperimentResult(
        specs=tuple(finalized_specs),
        results=tuple(references),
        failures=(),
    )


def _storage_attempt_id(
    attempt: SynthesisAttempt, store: SynthesisArtifactStore
) -> str:
    if not attempt.corpus_digest:
        raise ValueError("synthesis attempt has no corpus digest")
    if attempt.created_at:
        try:
            created = datetime.fromisoformat(attempt.created_at.replace("Z", "+00:00"))
        except ValueError:
            created = datetime.now(UTC)
    else:
        created = datetime.now(UTC)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    created_token = created.astimezone(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    creation_second = created.astimezone(UTC).replace(microsecond=0)
    existing_sequences = [
        sequence
        for item in store.attempts
        for existing_creation_second, sequence in (
            synthesis_attempt_position(item.attempt_id),
        )
        if existing_creation_second == creation_second
    ]
    sequence = max(existing_sequences, default=0) + 1
    return f"{created_token}-{attempt.corpus_digest[:12]}-{sequence}"


def _persist_synthesis_attempt(
    attempt: SynthesisAttempt,
    *,
    result: ExperimentResult,
    output: Path,
    loaded: LoadedProject,
) -> Path:
    corpus = _synthesis_corpus(result=result, output=output, loaded=loaded)
    store = SynthesisArtifactStore(output)
    collision: SynthesisArtifactError | None = None
    # ID selection happens before the store's publication lock; retry if another
    # synthesis writer publishes the same sequence first.
    for _ in range(16):
        persisted_attempt = replace(
            attempt,
            attempt_id=_storage_attempt_id(attempt, store),
        )
        try:
            return store.write_attempt(persisted_attempt, corpus)
        except SynthesisArtifactError as error:
            if str(error) not in {
                "attempt already exists; overwrite refused",
                "attempt sequence already exists for creation token",
            }:
                raise
            collision = error
    if collision is not None:
        raise collision
    raise AssertionError("synthesis attempt publication retry loop was empty")


def _persist_unavailable_synthesis(
    *,
    result: ExperimentResult,
    output: Path,
    loaded: LoadedProject,
) -> None:
    try:
        corpus = _synthesis_corpus(result=result, output=output, loaded=loaded)
        synthesis = loaded.runtime.report_synthesis
        attempt = asyncio.run(
            ReportSynthesisService(
                max_retrieval_rounds=synthesis.max_retrieval_rounds,
                max_adjudication_revisions=synthesis.max_adjudication_revisions,
                max_final_verifications=synthesis.max_final_verifications,
                principles=ux_principles(),
            ).synthesize(corpus)
        )
        _persist_synthesis_attempt(
            attempt,
            result=result,
            output=output,
            loaded=loaded,
        )
    except Exception:
        return


def _complete_experiment(
    result: ExperimentResult,
    *,
    output: Path,
    runtime: RuntimeConfig,
    selected_specs: Sequence[RunSpec] | None = None,
    render_report: bool = True,
) -> tuple[Path, Path]:
    del runtime
    summary_path = _write_experiment_summary(
        result,
        output=output,
        selected_specs=selected_specs,
    )
    report_path = output / "report.html"
    if render_report:
        report_path = _render_completed_report(output=output)
    return summary_path, report_path


def complete_experiment(
    result: ExperimentResult,
    *,
    output: Path,
    runtime: RuntimeConfig,
    selected_specs: Sequence[RunSpec] | None = None,
    render_report: bool = True,
) -> tuple[Path, Path]:
    """Persist production experiment evaluation and render its report."""

    return _complete_experiment(
        result,
        output=output,
        runtime=runtime,
        selected_specs=selected_specs,
        render_report=render_report,
    )


def _provider_cell_aggregates(
    metrics: Sequence[RunMetrics], specs: Sequence[RunSpec]
) -> tuple[dict[str, object], ...]:
    """Serialize cell aggregates without merging prominence provider axes."""

    specs_by_run = {spec.run_id: spec for spec in specs}
    groups: dict[str, list[RunMetrics]] = {}
    for metric in metrics:
        spec = specs_by_run.get(metric.run_id)
        if spec is not None and metric.comparison_valid:
            groups.setdefault(spec.prominence_provider_id, []).append(metric)
    rows: list[dict[str, object]] = []
    for provider_id in sorted(groups):
        for aggregate in aggregate_cells(groups[provider_id]):
            value = _json_data(aggregate)
            if not isinstance(value, dict):
                raise TypeError("cell aggregate did not serialize to an object")
            row = cast(dict[str, object], value)
            row["prominence_provider_id"] = provider_id
            rows.append(row)
    return tuple(rows)


def _selected_finalized_evidence(
    output: Path,
    selected_specs: Sequence[RunSpec],
    returned_results: Sequence[RunResult],
) -> tuple[tuple[RunMetrics, ...], dict[str, object]]:
    selected_by_run = {spec.run_id: spec for spec in selected_specs}
    returned_run_ids = {result.run_id for result in returned_results}
    metrics_by_run: dict[str, RunMetrics] = {}
    findings_by_run: dict[str, object] = {}
    for result in returned_results:
        spec = selected_by_run.get(result.run_id)
        if spec is None or result.metrics is None:
            continue
        if result.state.spec.prominence_provider_id != spec.prominence_provider_id:
            continue
        if not comparison_sample_is_valid(result, result.metrics):
            continue
        if not finalized_bundle_is_valid(
            output,
            spec.run_id,
            expected_prominence_provider_id=spec.prominence_provider_id,
        ):
            continue
        metrics_by_run[result.run_id] = result.metrics
        findings_by_run[result.run_id] = result.findings or ()
    adapter = TypeAdapter(RunMetrics)
    for spec in selected_specs:
        if spec.run_id in metrics_by_run or spec.run_id in returned_run_ids:
            continue
        if not finalized_bundle_is_valid(
            output,
            spec.run_id,
            expected_prominence_provider_id=spec.prominence_provider_id,
        ):
            continue
        try:
            persisted_snapshot = read_finalized_bundle(
                output / "runs" / spec.run_id,
                expected_run_id=spec.run_id,
                expected_prominence_provider_id=spec.prominence_provider_id,
            )
            persisted_object = persisted_snapshot.result
            manifest_object = persisted_snapshot.manifest
            raw_metrics = persisted_object.get("metrics")
            if isinstance(
                raw_metrics, Mapping
            ) and persisted_comparison_sample_is_valid(
                persisted_object,
                cast(Mapping[str, object], raw_metrics),
                manifest=manifest_object,
                expected_run_id=spec.run_id,
                expected_prominence_provider_id=spec.prominence_provider_id,
                timeline_events=persisted_snapshot.events,
            ):
                persisted_metrics = adapter.validate_python(raw_metrics)
                if persisted_metrics.comparison_valid:
                    metrics_by_run[spec.run_id] = persisted_metrics
            findings_by_run[spec.run_id] = persisted_object.get("findings") or ()
        except (OSError, UnicodeError, ValueError) as error:
            raise CheckpointError(
                f"invalid finalized evaluation for {spec.run_id}: {type(error).__name__}"
            ) from error
    ordered_metrics = tuple(
        metrics_by_run[spec.run_id]
        for spec in selected_specs
        if spec.run_id in metrics_by_run
    )
    return ordered_metrics, findings_by_run


def _compare_selected_variants(
    metrics: Sequence[RunMetrics], selected_specs: Sequence[RunSpec]
) -> tuple[object, ...]:
    specs_by_run = {spec.run_id: spec for spec in selected_specs}
    groups: dict[
        tuple[str, str, str, int, str], dict[ApplicationVersionKind, list[RunMetrics]]
    ] = {}
    for metric in metrics:
        if not metric.comparison_valid:
            continue
        spec = specs_by_run.get(metric.run_id)
        if spec is None:
            continue
        key = (
            metric.scenario_id,
            metric.persona_id,
            metric.policy,
            metric.model_trial,
            spec.prominence_provider_id,
        )
        groups.setdefault(key, {}).setdefault(spec.application_version.kind, []).append(
            metric
        )
    comparisons: list[object] = []
    for key in sorted(groups):
        baseline = groups[key].get(ApplicationVersionKind.DEFECTIVE)
        improved = groups[key].get(ApplicationVersionKind.IMPROVED)
        if (
            baseline
            and improved
            and {(item.seed, item.model_trial) for item in baseline}
            == {(item.seed, item.model_trial) for item in improved}
        ):
            provider_id = key[-1]
            comparison = _json_data(compare_variants(baseline, improved))
            if not isinstance(comparison, dict):
                raise TypeError("variant comparison did not serialize to an object")
            comparison_row = cast(dict[str, object], comparison)
            comparison_row["prominence_provider_id"] = provider_id
            for variant_name in ("baseline", "improved"):
                variant = comparison_row.get(variant_name)
                if isinstance(variant, dict):
                    cast(dict[str, object], variant)["prominence_provider_id"] = (
                        provider_id
                    )
            comparisons.append(comparison_row)
    return tuple(comparisons)


def _evaluation_failures(results: Sequence[object]) -> tuple[RunResult, ...]:
    return tuple(
        item
        for item in results
        if isinstance(item, RunResult) and item.evaluation_failure_reason is not None
    )


def _invalid_ux_samples(results: Sequence[object]) -> tuple[RunResult, ...]:
    return tuple(
        item
        for item in results
        if isinstance(item, RunResult) and not item.ux_sample_valid
    )


def _print_result_summary(result: ExperimentResult) -> None:
    run_results = tuple(item for item in result.results if isinstance(item, RunResult))
    invalid = _invalid_ux_samples(run_results)
    outcomes = Counter(item.outcome.kind for item in run_results)
    typer.echo(
        f"finalized runs: {len(run_results)}; execution failures: {len(result.failures)}"
    )
    typer.echo(
        f"UX samples: {len(run_results) - len(invalid)} valid; {len(invalid)} invalid"
    )
    if outcomes:
        typer.echo(
            "outcomes: "
            + ", ".join(f"{kind}={count}" for kind, count in sorted(outcomes.items()))
        )


def _invalid_ux_sample_record(result: RunResult) -> dict[str, object]:
    spec = result.state.spec
    return {
        "run_id": result.run_id,
        "outcome": result.outcome.kind,
        "reason": result.ux_sample_invalid_reason or "invalid UX sample",
        "scenario_id": spec.scenario.id,
        "application_version_id": spec.application_version.id,
        "persona_id": spec.persona.id,
        "policy": spec.policy.value,
        "seed": spec.seed,
        "model_trial": spec.model_trial,
        "prominence_provider_id": spec.prominence_provider_id,
    }


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
        "model_trial": spec.model_trial,
        "prominence_provider_id": spec.prominence_provider_id,
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
        "model_trial": spec.model_trial,
        "prominence_provider_id": spec.prominence_provider_id,
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
    application_version = spec.application_version
    if application_version.kind is ApplicationVersionKind.LIVE:
        if application_version.start_url is None:
            raise ValueError("live application version has no start URL")
        start_url = application_version.start_url
        live_origins = BrowserAllowedOrigins.for_live(
            start_url, application_version.allowed_origins
        )
        navigation_origins = tuple(sorted(live_origins.navigation_origins))
        resource_origins = tuple(sorted(live_origins.resource_origins))
        fixture_only = False
    else:
        version = application_version.kind.value
        start_url = f"{fixture_origin}/app/{quote(spec.run_id)}/{version}{page}"
        navigation_origins = (fixture_origin,)
        resource_origins = ()
        fixture_only = True
    return ObservationSessionConfig(
        session_id=spec.run_id,
        start_url=start_url,
        test_account_id=TestAccountId(f"test-{spec.run_id.removeprefix('run-')}"),
        viewport=ViewportSize(
            width=spec.scenario.viewport_width,
            height=spec.scenario.viewport_height,
        ),
        trace_path=output / "traces" / f"{spec.run_id}.zip",
        artifact_redaction=RedactionPolicy.from_fixture_inputs(
            spec.scenario.fixture_inputs
        ),
        navigation_settle_ms=application_version.navigation_settle_ms,
        action_settle_ms=application_version.action_settle_ms,
        navigation_origins=navigation_origins,
        resource_origins=resource_origins,
        fixture_only=fixture_only,
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
