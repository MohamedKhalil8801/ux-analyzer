"""Operator command surface for benchmark validation and execution."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx
import typer

if TYPE_CHECKING:
    from ux_analyzer.analysis.project_audit import CaptureMaterialsHook
import yaml
from pydantic import TypeAdapter

from ux_analyzer import __version__
from ux_analyzer.adapters.exploration_server import (
    ExplorationReviewServer,
    PersonaSelectionPayload,
    _suggestion_to_dict,
)
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
from ux_analyzer.application.exploration_crawler import (
    CrawlFrontier,
    ExplorationCrawler,
    PageSettlementPolicy,
)
from ux_analyzer.application.exploration_synthesizer import ExplorationSynthesizer
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
    Application,
    ApplicationVersion,
    ApplicationVersionKind,
    Budget,
    ExperimentDefinition,
    ExperimentPolicy,
    Persona,
    ScenarioEvaluationTarget,
    VisibleResultVerifierSpec,
    resolve_prominence_provider_id,
)
from ux_analyzer.domain.benchmark import (
    normalize_crawl_url as benchmark_normalize_crawl_url,
)
from ux_analyzer.domain.exploration import (
    CrawlCorpus,
    CrawlPage,
    ExplorationSpec,
    ScenarioSuggestion,
)
from ux_analyzer.domain.exploration import (
    normalize_crawl_url as exploration_normalize_crawl_url,
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
from ux_analyzer.reporting.renderer import (
    load_report_findings,
    render_experiment_report,
)
from ux_analyzer.saliency.model_registry import (
    DEFAULT_MODEL_ID,
    DEFAULT_PRECISION,
    ModelRegistry,
    ModelRegistryError,
    load_manifest,
)
from ux_analyzer.storage.exploration_artifacts import (
    ExplorationArtifactError,
    ExplorationArtifactStore,
    exploration_digest,
)
from ux_analyzer.storage.redesign_artifacts import (
    new_attempt_id as new_redesign_attempt_id,
)
from ux_analyzer.storage.run_bundle import (
    FilesystemRunBundleWriter,
    secure_create_exclusive_file,
    secure_open_directory,
    secure_replace_exclusive_file,
)
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

_MAX_EXPERIMENT_JSON_BYTES = 8 * 1024 * 1024
_EXPERIMENT_JSON_CHUNK_BYTES = 64 * 1024
_EXPERIMENT_JSON_WRITE_LOCK = Lock()
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


def _site_slug(url: str) -> str:
    """Sanitized hostname slug for nested default output folders."""

    host = urlsplit(url).hostname or "site"
    slug = re.sub(r"[^a-z0-9]+", "-", host.casefold()).strip("-")
    return slug or "site"


def _resolve_default_output(output: Path | None, project_id: str) -> Path:
    """Nest benchmark outputs under the git-ignored ``reports/`` parent.

    Each project gets its own folder so multiple experiments never collide
    in one checkpoint root and stray outputs never scatter across the repo
    root.
    """

    if output is not None:
        return output
    return Path("reports") / project_id


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
            report_synthesis_enabled=(check_env and _synthesis_gate(loaded))
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
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output directory [default: reports/<project-id>]",
    ),
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
    allow_origin: list[str] = typer.Option(
        [],
        "--allow-origin",
        help="Grant one extra browser resource origin (repeatable); skips the prompt.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Accept all referenced non-allowlisted origins without prompting.",
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
        allow_origin=allow_origin,
        yes=yes,
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
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output directory [default: reports/<project-id>]",
    ),
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
    output = _resolve_default_output(output, matrix.loaded.project.id)
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
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output directory [default: reports/<project-id>]",
    ),
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
    allow_origin: list[str] = typer.Option(
        [],
        "--allow-origin",
        help="Grant one extra browser resource origin (repeatable); skips the prompt.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Accept all referenced non-allowlisted origins without prompting.",
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
        allow_origin=allow_origin,
        yes=yes,
    )


@app.command()
def report(
    bundle_root: Path,
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Rendered report path [default: <bundle-root>/report.html]",
    ),
    serve: bool = typer.Option(
        False,
        "--serve",
        help=(
            "Serve the report over loopback HTTP after rendering. Live sidecar "
            "views (Page findings, Performance) refetch ux-audit.json / "
            "pagespeed.json at page load, which browsers block on file:// URLs."
        ),
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        min=1,
        max=65535,
        help="Port for --serve [default: ephemeral]",
    ),
    open_browser: bool = typer.Option(
        True,
        "--browser/--no-browser",
        help="Open the served report in the browser",
    ),
) -> None:
    """Regenerate static report from finalized run bundles."""
    resolved_output = output if output is not None else bundle_root / "report.html"
    try:
        rendered = render_experiment_report(bundle_root, resolved_output)
    except (FileNotFoundError, OSError, ValueError) as error:
        _exit_with_error(f"report failed: {error}")
    typer.echo(f"report generated: {rendered}")
    if serve:
        from ux_analyzer.reporting.serve import serve_report

        try:
            serve_report(
                resolved_output.parent,
                port=port,
                open_browser=open_browser,
            )
        except OSError as error:
            _exit_with_error(f"report serve failed: {error}")


@app.command()
def pagespeed(
    url: str = typer.Argument(..., help="URL to run PageSpeed Insights on"),
    strategy: list[str] = typer.Option(
        ["mobile", "desktop"], "--strategy", help="Lighthouse strategy (repeatable)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the derived report JSON instead of summary"
    ),
    cache_root: Path | None = typer.Option(
        None, "--cache-root", help="Directory for the raw-response disk cache"
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Ignore and do not update the disk cache"
    ),
    web_ui: bool = typer.Option(
        True,
        "--web-ui/--no-web-ui",
        help="Run one pagespeed.web.dev analysis per URL to capture the saved-report link",
    ),
) -> None:
    """Fetch the complete PageSpeed Insights report for any URL.

    Calls the same API behind pagespeed.web.dev
    (https://www.googleapis.com/pagespeedonline/v5/runPagespeed), matching
    its Lighthouse category scores, per-audit pass/fail results, and
    opportunity savings. Uses PSI_API_Key / GOOGLE_API_KEY from .env or the
    environment when present. With --web-ui (default) each URL is also
    analyzed once in a headless pagespeed.web.dev session to capture the
    stable saved-report link and the scores that report renders — that is
    a second, independent Lighthouse run, so its scores are labeled and
    shown separately from the API run's. The capture is cached per URL for
    7 days; set UXA_SKIP_PAGESPEED_WEB=1 to disable.
    """
    from ux_analyzer.analysis.pagespeed import (
        enrich_pagespeed_web_links,
        pagespeed_api_key,
        pagespeed_report_sync,
    )

    try:
        report = pagespeed_report_sync(
            (url,),
            key=pagespeed_api_key(),
            cache_root=cache_root,
            strategies=tuple(dict.fromkeys(strategy)),
            use_cache=not no_cache,
            store_cache=not no_cache,
        )
        report = enrich_pagespeed_web_links(
            report,
            cache_root=cache_root,
            resolve=web_ui,
        )
    except Exception as error:
        _exit_with_error(f"pagespeed failed: {type(error).__name__}: {error}")
    if json_output:
        typer.echo(json.dumps(report, indent=2))
        return
    for url_report in report["urls"]:
        typer.echo(f"{url_report['url']}")
        web_link = url_report.get("pagespeed_web_url")
        saved = bool(url_report.get("pagespeed_web_saved"))
        if web_link:
            typer.echo(
                f"  pagespeed.web.dev: {web_link}"
                + (" (saved report)" if saved else " (runs a fresh analysis)")
            )
        saved_scores = url_report.get("saved_report_scores") or {}
        if saved and saved_scores:
            rendered = ", ".join(
                f"{name} {score}" for name, score in saved_scores.items()
            )
            captured_at = url_report.get("saved_report_captured_at") or ""
            typer.echo(
                f"  saved report scores: {rendered}"
                + (f" (captured {captured_at[:10]})" if captured_at else "")
            )
        for strategy_name in url_report["strategies"]:
            entry = url_report["strategies"][strategy_name]
            if entry.get("status") != "ok":
                typer.secho(
                    f"  {strategy_name}: unavailable ({entry.get('error', 'unknown')})",
                    fg="red",
                )
                continue
            categories = ", ".join(
                f"{c['title']} {c['score_percent']}" for c in entry["categories"]
            )
            failed = entry["audits"]["totals"]["failed"]
            opportunities = entry["opportunities"]
            total_savings_ms = sum(o["savings_ms"] or 0 for o in opportunities)
            total_savings_bytes = sum(o["savings_bytes"] or 0 for o in opportunities)
            typer.secho(
                f"  {strategy_name}: {categories}",
                fg="green"
                if all(
                    c["score_percent"] and c["score_percent"] >= 90
                    for c in entry["categories"]
                )
                else "yellow",
            )
            typer.echo(
                f"    audits: {failed} failed, {entry['audits']['totals']['passed']} passed; "
                f"{len(opportunities)} opportunity(ies) "
                f"(~{total_savings_ms} ms / {total_savings_bytes} bytes est. savings)"
            )
            for opportunity in opportunities:
                typer.echo(
                    f"      - {opportunity['title']}: "
                    f"{opportunity['display_value'] or 'n/a'} "
                    f"(~{opportunity['savings_ms'] or 0} ms, "
                    f"{opportunity['savings_bytes'] or 0} bytes, "
                    f"{len(opportunity['items'])} item(s))"
                )


@app.command()
def slop(
    source: str = typer.Argument(..., help="URL or local HTML file to score"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON instead of pretty output"
    ),
    copy: bool = typer.Option(
        False, "--copy", help="Also score the copy axis (9 patterns)"
    ),
) -> None:
    """Score any page against the 27-rule AI-design-slop fingerprint.

    Renders the page in a headless browser, extracts a computed-style
    snapshot plus page text, then reports the 0-100 score, tier
    (Clean / Mild / Heavy), and every triggered pattern with its evidence.
    """
    from ux_analyzer.analysis.slop.pipeline import analyze_slop
    from ux_analyzer.analysis.visual.extract import extract_snapshot
    from ux_analyzer.analysis.visual.snapshot import snapshot_from_dict

    try:
        raw = extract_snapshot(source)
        meta = raw.get("meta", {})
        report = analyze_slop(
            snapshot_from_dict(raw),
            viewport_w=int(meta.get("viewport", {}).get("w", 1280)),
            viewport_h=int(meta.get("viewport", {}).get("h", 800)),
            doc_height=int(meta.get("docHeight", 0) or 0),
            scroll_y=int(meta.get("scrollY", 0) or 0),
            text_context=meta.get("textContext"),
            surface=meta.get("surface"),
        )
    except Exception as error:
        _exit_with_error(f"slop scan failed: {type(error).__name__}: {error}")
    report = {**report, "url": source}
    if json_output:
        typer.echo(json.dumps(report, indent=2))
        return
    tier_color = {"Clean": "green", "Mild": "yellow", "Heavy": "red"}.get(
        report["tier"], "white"
    )
    typer.secho(f"{source}", bold=True)
    typer.secho(
        f"{report['tier']}  ·  score {report['score']}/100  ·  "
        f"{report['patternsFlagged']}/{report['patternsTotal']} patterns triggered"
        f"  ·  grade {report['grade']}",
        fg=tier_color,
        bold=True,
    )
    typer.echo("Triggered:")
    for p in report["patterns"]:
        if not p["triggered"]:
            continue
        typer.secho(f"  x {_ascii(p['label'])}  (+{p['weight']})", fg="red")
        _print_slop_evidence(p["evidence"])
    if copy and report.get("copy"):
        copy_summary = report["copy"]
        typer.echo("")
        typer.secho(
            f"copy: {copy_summary['tier']}  score {copy_summary['score']}/100  ·  "
            f"{copy_summary['patternsFlagged']}/{copy_summary['patternsTotal']} flagged",
            bold=True,
        )
        for p in copy_summary.get("patterns", []):
            if not p["triggered"]:
                continue
            typer.secho(f"  x {_ascii(p['label'])}  (+{p['weight']})", fg="red")
            _print_slop_evidence(p["evidence"])
    if report.get("axesScored") and len(report["axesScored"]) > 1:
        typer.echo(
            f"unified: {report.get('unifiedScore')}/100 · {report.get('unifiedTier')}"
        )


def _ascii(text: str) -> str:
    """Console-safe text for legacy code pages (cp1252 etc.)."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _print_slop_evidence(evidence: object) -> None:
    if not isinstance(evidence, dict):
        return
    for key, value in evidence.items():
        if key in ("triggered", "error"):
            continue
        if isinstance(value, str) and value:
            typer.echo(f"      {key}: {_ascii(value[:160])}")
        elif isinstance(value, (int, float, bool)):
            typer.echo(f"      {key}: {value}")
        elif isinstance(value, list) and value:
            typer.echo(
                f"      {key}: {_ascii(json.dumps(value, ensure_ascii=False)[:220])}"
            )


@app.command()
def synthesize(
    project: Path,
    experiment: str = typer.Option(..., "--experiment"),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output directory [default: reports/<project-id>]",
    ),
) -> None:
    """Run report synthesis for finalized experiment evidence."""
    matrix = _resolve_matrix_or_exit(
        project,
        experiment,
        run_count=None,
        policies=(),
    )
    output = _resolve_default_output(output, matrix.loaded.project.id)
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


@app.command()
def explore(
    project: Path | None = typer.Argument(
        None,
        help="Base project YAML (optional positional). Provides personas/providers.",
    ),
    starting_url: list[str] = typer.Option(
        [],
        "--starting-url",
        help="Starting URL (repeatable, HTTPS required). Flag wins over YAML exploration section.",
        show_default=False,
    ),
    depth: int | None = typer.Option(
        None, "--depth", help="Crawl depth 0-5 (flag wins over YAML)"
    ),
    max_pages: int | None = typer.Option(None, "--max-pages", help="Max pages 1-200"),
    max_scenarios: int | None = typer.Option(
        None, "--max-scenarios", help="Max scenarios 1-20"
    ),
    auto_accept: bool = typer.Option(
        False, "--auto-accept", help="Skip UI and accept all suggestions"
    ),
    review_port: int | None = typer.Option(
        None, "--review-port", help="Review UI port 1-65535"
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help=(
            "Output directory "
            "[default: .uxa-output/explore/<site-slug from first starting URL>]"
        ),
    ),
    project_opt: Path | None = typer.Option(
        None, "--project", help="Base project YAML (alternative to positional)"
    ),
    run: bool = typer.Option(
        False, "--run", help="Immediately run generated experiment"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview crawl matrix and token estimate without browser/model",
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not auto-open browser for review UI"
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Reuse checkpointed crawl corpus and suggestions from an interrupted run",
    ),
) -> None:
    """Discover scenarios via smart crawl + cognitive synthesis + human curation.

    Loads base project to reuse personas/providers if --project given or positional project supplied,
    else bootstraps minimal BenchmarkProject with one Application(live) per unique start origin.
    Builds ExplorationSpec from flags (flag wins over YAML exploration section), runs
    ExplorationCrawler -> CrawlCorpus, runs ExplorationSynthesizer -> suggestions,
    then either auto-accepts or launches ExplorationReviewServer until curated.
    Persists via ExplorationArtifactStore and materializes project.fragment.yaml / full project.yaml
    with new scenarios+experiments (one experiment exploration-run covering curated scenarios x selected personas x default policies).
    """
    # Resolve base project path: --project wins, else positional
    base_path: Path | None = project_opt if project_opt is not None else project
    loaded: LoadedProject | None = None
    if base_path is not None:
        # Use _load_project_or_exit to validate and provide actionable error
        loaded = _load_project_or_exit(base_path)
    # Determine raw starting URLs: flag wins, else YAML exploration section, else error
    raw_starts: list[str] = []
    if starting_url:
        raw_starts = list(starting_url)
    elif loaded is not None and getattr(loaded, "exploration", None) is not None:
        expl = loaded.exploration  # type: ignore[union-attr]
        # expl may be ExplorationModel
        try:
            raw_starts = list(expl.start_urls)  # type: ignore[union-attr]
        except Exception:
            raw_starts = []
    # Validate starting URLs presence
    if not raw_starts:
        _exit_with_error(
            "starting-url is required (provide --starting-url or set exploration.start_urls in project)"
        )
    # Flag validation: depth, max_pages, max_scenarios, review_port
    # Use explicit validation for helpful messages; also rely on ExplorationSpec for duplicate etc.
    # Depth bounds
    depth_val: int | None = depth
    if depth_val is not None and not (0 <= depth_val <= 5):
        _exit_with_error("depth must be between 0 and 5")
    max_pages_val: int | None = max_pages
    if max_pages_val is not None and not (1 <= max_pages_val <= 200):
        _exit_with_error("max_pages must be between 1 and 200")
    max_scenarios_val: int | None = max_scenarios
    if max_scenarios_val is not None and not (1 <= max_scenarios_val <= 20):
        _exit_with_error("max_scenarios must be between 1 and 20")
    if review_port is not None and not (1 <= review_port <= 65535):
        _exit_with_error("review-port must be between 1 and 65535")
    # Resolve effective values: flag wins over YAML else defaults
    # For depth/max_pages/max_scenarios we need to consider YAML defaults
    yaml_depth = None
    yaml_max_pages = None
    yaml_max_scenarios = None
    yaml_settle_ms = None
    if loaded is not None and getattr(loaded, "exploration", None) is not None:
        expl = loaded.exploration  # type: ignore[union-attr]
        try:
            yaml_depth = int(getattr(expl, "depth"))  # type: ignore[union-attr]
        except Exception:
            yaml_depth = None
        try:
            yaml_max_pages = int(getattr(expl, "max_pages"))  # type: ignore[union-attr]
        except Exception:
            yaml_max_pages = None
        try:
            yaml_max_scenarios = int(getattr(expl, "max_scenarios"))  # type: ignore[union-attr]
        except Exception:
            yaml_max_scenarios = None
        try:
            yaml_settle_ms = int(getattr(expl, "settle_ms"))  # type: ignore[union-attr]
        except Exception:
            yaml_settle_ms = None
    effective_depth = (
        depth_val
        if depth_val is not None
        else (yaml_depth if yaml_depth is not None else 2)
    )
    effective_max_pages = (
        max_pages_val
        if max_pages_val is not None
        else (yaml_max_pages if yaml_max_pages is not None else 50)
    )
    effective_max_scenarios = (
        max_scenarios_val
        if max_scenarios_val is not None
        else (yaml_max_scenarios if yaml_max_scenarios is not None else 8)
    )
    effective_settle_ms = yaml_settle_ms if yaml_settle_ms is not None else 10000
    # Validate effective bounds (already did flag, but also ensure final)
    if not (0 <= effective_depth <= 5):
        _exit_with_error("depth must be between 0 and 5")
    if not (1 <= effective_max_pages <= 200):
        _exit_with_error("max_pages must be between 1 and 200")
    if not (1 <= effective_max_scenarios <= 20):
        _exit_with_error("max_scenarios must be between 1 and 20")
    # Normalize and validate starting URLs (HTTPS, duplicate, etc.)
    # Use exploration_normalize_crawl_url for HTTPS enforcement
    normalized_starts: list[str] = []
    for raw in raw_starts:
        try:
            n = exploration_normalize_crawl_url(raw)
        except ValueError as ve:
            _exit_with_error(f"invalid starting-url {raw!r}: {ve}")
        except Exception as ve:
            _exit_with_error(f"invalid starting-url {raw!r}: {ve}")
        normalized_starts.append(n)
    if len(normalized_starts) != len(set(normalized_starts)):
        _exit_with_error(
            "starting-url must be unique after normalization (duplicate detected)"
        )
    if effective_depth == 0 and effective_max_pages < len(normalized_starts):
        _exit_with_error("max_pages must be >= number of start URLs when depth is 0")
    # Nest exploration workspaces under one git-ignored parent so per-site
    # folders never scatter across the repository root.
    output = output or (
        Path(".uxa-output") / "explore" / _site_slug(normalized_starts[0])
    )
    # Dry-run: print crawl matrix + synthesis token estimate without browser/model
    if dry_run:
        # Validate model not required, just print
        typer.echo("explore matrix:")
        typer.echo(f" starts: {len(normalized_starts)}")
        for u in normalized_starts:
            typer.echo(f"  - {u}")
        typer.echo(f" depth: {effective_depth}")
        typer.echo(f" max_pages: {effective_max_pages}")
        typer.echo(f" max_scenarios: {effective_max_scenarios}")
        typer.echo(f" settle_ms: {effective_settle_ms}")
        # Estimates below are heuristic bounds derived from configuration,
        # not measurements; the real page count and token usage depend on
        # the crawled site and the model.
        est_pages = min(
            effective_max_pages,
            max(1, len(normalized_starts) * (effective_depth + 1) * 3),
        )
        typer.echo(f" estimated_pages: {est_pages} (upper-bound estimate)")
        # Rough synthesis prompt size: ~800 tokens per page plus fixed overhead.
        est_tokens = est_pages * 800 + 500
        typer.echo(
            f" synthesis token estimate: ~{est_tokens} tokens (rough estimate,"
            " not a measurement)"
        )
        # A corpus digest can only exist after a real crawl; dry-run exposes
        # only a config fingerprint derived from these settings.
        config_fingerprint = exploration_digest(
            {
                "starts": normalized_starts,
                "depth": effective_depth,
                "max_pages": effective_max_pages,
            }
        )
        typer.echo(
            f" config fingerprint: {config_fingerprint[:12]} (settings hash,"
            " not a corpus digest)"
        )
        typer.echo(f" output: {output}")
        if auto_accept:
            typer.echo(" auto_accept: enabled (UI skipped)")
        else:
            typer.echo(" auto_accept: disabled (would launch review UI)")
        return
    # Non-dry-run requires model env
    try:
        settings = _model_settings_or_exit(report_synthesis_enabled=False)
    except SystemExit:
        raise
    except Exception as error:
        _exit_with_error(f"model environment error: {error}")
    # Prepare output directory early for validation
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        _exit_with_error(f"cannot create output directory {output}: {error}")

    # Async orchestration: crawl -> synthesize -> curate -> persist -> materialize -> optionally run
    async def _explore_async() -> None:
        # Bootstrap or reuse personas/applications
        # Determine existing personas and applications
        # Always compute bootstrap live apps for exploration origins to guarantee live target
        bootstrap_live_apps = _explore_bootstrap_applications(tuple(normalized_starts))
        if loaded is not None:
            existing_personas = tuple(loaded.project.personas)
            # Combine existing + bootstrap live apps (avoid id collision)
            existing_app_ids = {a.id for a in loaded.project.applications}
            merged_apps: list[Application] = list(loaded.project.applications)
            for b_app in bootstrap_live_apps:
                if b_app.id not in existing_app_ids:
                    merged_apps.append(b_app)
                else:
                    # collide -> create variant id
                    variant_id = f"{b_app.id}-explore"
                    if variant_id not in existing_app_ids:
                        variant = Application(
                            id=variant_id,
                            name=b_app.name,
                            versions=tuple(
                                ApplicationVersion(
                                    id=f"{variant_id}-live",
                                    kind=ApplicationVersionKind.LIVE,
                                    label=v.label,
                                    start_url=v.start_url,
                                    allowed_origins=tuple(v.allowed_origins),
                                )
                                for v in b_app.versions
                            ),
                        )
                        merged_apps.append(variant)
            existing_applications = tuple(merged_apps)
            # Collect existing application_version_ids for live versions
            persona_ids_for_exp = (
                [p.id for p in existing_personas] if existing_personas else []
            )
            if not persona_ids_for_exp:
                # fallback if project has no personas (should not happen)
                persona_ids_for_exp = ["default-explorer"]
        else:
            # Bootstrap minimal
            existing_applications = bootstrap_live_apps
            # default persona
            default_persona = Persona(
                id="default-explorer",
                name="Default Explorer",
                working_memory_capacity=4,
                initial_confidence=0.55,
                initial_frustration=0.10,
                abandonment_threshold=0.75,
                attention_temperature=1.0,
            )
            existing_personas = (default_persona,)
            persona_ids_for_exp = ["default-explorer"]
        # Build ExplorationSpec
        # Derive allowed origins from normalized starts (unique origins)
        allowed_origins_set: set[str] = set()
        for nurl in normalized_starts:
            try:
                origin = _explore_origin_from_url(nurl)
                allowed_origins_set.add(origin)
            except Exception:
                continue
        allowed_origins_tuple = tuple(sorted(allowed_origins_set))
        try:
            spec = ExplorationSpec(
                start_urls=tuple(normalized_starts),
                depth=effective_depth,
                max_pages=effective_max_pages,
                max_scenarios=effective_max_scenarios,
                settle_ms=effective_settle_ms,
                allowed_origins=allowed_origins_tuple,
            )
        except Exception as ve:
            _exit_with_error(f"invalid exploration spec: {ve}")
        # Print crawl matrix
        typer.echo("explore matrix:")
        typer.echo(
            f" starts: {len(normalized_starts)} depth={spec.depth} max_pages={spec.max_pages} max_scenarios={spec.max_scenarios}"
        )
        for u in spec.start_urls:
            typer.echo(f"  - {u}")
        # Run crawler / synthesizer with checkpoint-resume support.
        # A matching checkpoint (interrupted previous run) skips the expensive
        # crawl and, when present, the synthesis step too.
        checkpoint = _explore_checkpoint_load(output, spec)
        if checkpoint is None and resume:
            if _explore_checkpoint_exists(output):
                _exit_with_error(
                    "resume requested but the existing checkpoint does not match "
                    "current exploration settings (starting URLs, depth, "
                    "max-pages, max-scenarios, or settle-ms)"
                )
        policy = PageSettlementPolicy(settle_ms=spec.settle_ms)
        corpus: CrawlCorpus
        suggestions: tuple[ScenarioSuggestion, ...] | None

        async def _save_frontier_progress(frontier: CrawlFrontier) -> None:
            # Per-page crash-recovery: each completed page is persisted
            # before the next one starts. Failures are swallowed inside
            # the checkpoint writer so a flaky disk cannot kill the crawl.
            _explore_checkpoint_save_progress(output, spec, frontier)

        if checkpoint is not None and (
            checkpoint.crawl_complete or checkpoint.suggestions is not None
        ):
            corpus = checkpoint.corpus
            suggestions = checkpoint.suggestions
            typer.echo(
                f"resume: reusing checkpointed crawl corpus ({len(corpus.pages)} pages); crawl skipped"
            )
            if suggestions is not None:
                typer.echo(
                    f"resume: reusing checkpointed suggestions ({len(suggestions)} scenarios); synthesis skipped"
                )
        elif checkpoint is not None and checkpoint.frontier is not None:
            saved_frontier = checkpoint.frontier
            typer.echo(
                f"crawling: resuming interrupted crawl "
                f"({len(saved_frontier.pages)} pages done,"
                f" {len(saved_frontier.queued)} queued)"
            )
            try:
                corpus = await _explore_run_crawler(
                    spec,
                    policy,
                    on_frontier=_save_frontier_progress,
                    resume_from=saved_frontier,
                )
            except Exception as error:
                _exit_with_error(f"crawl failed: {type(error).__name__}: {error}")
            _explore_checkpoint_save(output, spec, corpus, None, crawl_complete=True)
            suggestions = None
        else:
            typer.echo(f"crawling: depth={spec.depth} max_pages={spec.max_pages}")
            try:
                corpus = await _explore_run_crawler(
                    spec, policy, on_frontier=_save_frontier_progress
                )
            except Exception as error:
                _exit_with_error(f"crawl failed: {type(error).__name__}: {error}")
            _explore_checkpoint_save(output, spec, corpus, None, crawl_complete=True)
            suggestions = None
        typer.echo(
            f" crawl corpus: {len(corpus.pages)} pages digest={corpus.corpus_digest[:12]}"
        )
        # Run synthesizer
        if suggestions is None:
            typer.echo(f"synthesizing: max_scenarios={spec.max_scenarios}")
            try:
                suggestions = await _explore_run_synthesizer(
                    corpus, spec.max_scenarios, settings
                )
            except Exception as error:
                # Persist real evidence with an unavailable status; never fake
                # suggestions to keep the pipeline moving.
                _persist_unavailable_exploration_attempt(
                    output=output,
                    corpus=corpus,
                    personas=existing_personas,
                    spec=spec,
                    model=settings.cognitive_model,
                    reason=f"{type(error).__name__}: {error}",
                )
                _exit_with_error(f"synthesis failed: {type(error).__name__}: {error}")
            _explore_checkpoint_save(output, spec, corpus, suggestions)
        assert suggestions is not None
        typer.echo(f" suggestions: {len(suggestions)} scenarios")
        for s in suggestions:
            typer.echo(f"  - {s.id}: {s.goal[:60]}")
        if not suggestions:
            typer.echo(
                " warning: synthesis produced 0 scenarios — the crawl had "
                f"{len(corpus.pages)} page(s) with sparse visible labels; "
                "try --depth 1 for broader coverage or add a custom scenario "
                "in the review UI.",
                err=True,
            )
        # Determine curated set
        curated: tuple[ScenarioSuggestion | dict[str, Any], ...]
        persona_selection_result: PersonaSelectionPayload | None = None
        if auto_accept:
            curated = suggestions
            typer.echo(" auto_accept: using all suggestions as curated")
            # Build persona selection for auto_accept: use existing personas
            persona_selection_result = PersonaSelectionPayload(
                mode="existing",
                persona_ids=persona_ids_for_exp,
            )
        else:
            typer.echo(
                f" launching review UI on port {review_port or 'auto'} (no_browser={no_browser})"
            )
            # Existing personas for UI
            suggested_personas: tuple[Any, ...] = ()
            # If synthesizer produced persona suggestions, they'd be here; currently empty
            server = ExplorationReviewServer(
                corpus=corpus,
                suggestions=suggestions,
                existing_personas=existing_personas,
                suggested_personas=suggested_personas,
                host="127.0.0.1",
                port=review_port or 0,
                auto_accept=False,
                no_browser=no_browser,
            )
            # Serve until curated
            result = await server.serve_forever()
            if result is None or "curated" not in result:
                _exit_with_error("review UI did not return curated set (aborted)")
            curated = tuple(result.get("curated", ()))
            raw_persona_selection = result.get("persona_selection")
            persona_selection_result = (
                PersonaSelectionPayload.model_validate(raw_persona_selection)
                if isinstance(raw_persona_selection, dict)
                else None
            )
            typer.echo(f" curated: {len(curated)} scenarios via review UI")
        # Normalize curated to list for store (handle both ScenarioSuggestion and dict)
        if not curated:
            _exit_with_error(
                "curated set is empty; nothing to materialize or run "
                "(accept at least one scenario in the review UI or via --auto-accept)"
            )
        # Persist via store
        store = ExplorationArtifactStore(output)
        try:
            attempt_path = store.write_attempt(
                corpus,
                suggestions,
                curated,
                persona_set=existing_personas,
                spec=spec,
                model=settings.cognitive_model,
                prompt_version="exploration-synthesis-v1",
                status="succeeded",
            )
        except FileExistsError as fe:
            _exit_with_error(f"exploration attempt already exists: {fe}")
        except ExplorationArtifactError as e:
            _exit_with_error(f"exploration artifact error: {e}")
        except Exception as e:
            _exit_with_error(f"exploration persist failed: {e}")
        typer.echo(f" exploration artifact: {attempt_path}")
        # Verify the published attempt is discoverable through the index.
        try:
            published_index = store.get_index()
        except ExplorationArtifactError as error:
            _exit_with_error(f"exploration index verification failed: {error}")
        raw_attempts = published_index.get("attempts", [])
        attempt_records: list[object] = (
            cast("list[object]", raw_attempts) if isinstance(raw_attempts, list) else []
        )
        listed_attempt_ids: set[str] = set()
        for attempt_entry in attempt_records:
            if not isinstance(attempt_entry, dict):
                continue
            record = cast(dict[str, object], attempt_entry)
            listed_attempt_ids.add(str(record.get("attempt_id")))
        if attempt_path.name not in listed_attempt_ids:
            _exit_with_error(
                "exploration index verification failed: published attempt "
                f"{attempt_path.name} is missing from the exploration index"
            )
        # Materialize project files
        try:
            generated_paths = _explore_materialize_project(
                output=output,
                base_path=base_path,
                loaded=loaded,
                corpus=corpus,
                suggestions=suggestions,
                curated=curated,
                existing_personas=existing_personas,
                existing_applications=existing_applications,
                persona_selection=persona_selection_result,
                spec=spec,
                attempt_path=attempt_path,
            )
        except Exception as e:
            _exit_with_error(f"materialize project failed: {type(e).__name__}: {e}")
        for p in generated_paths:
            typer.echo(f" generated: {p}")
        # The run completed curation, persistence, and materialization; the
        # crash-recovery checkpoint has served its purpose.
        _explore_checkpoint_clear(output)
        # Digest stability check (for tests): print digest
        typer.echo(
            f" digest stable: {corpus.corpus_digest[:12]} corpus {exploration_digest(corpus)[:12]}"
        )
        # Optionally run
        if run:
            if not generated_paths:
                _exit_with_error("no generated project to run")
            # Prefer the full project.yaml for run
            full_project = None
            for cand in generated_paths:
                if cand.name == "project.yaml" and cand.parent == output:
                    full_project = cand
                    break
            if full_project is None:
                full_project = generated_paths[0]
            typer.echo(
                f" running generated experiment: {full_project} --experiment exploration-run"
            )
            # Invoke experiment runner synchronously (reuse _run_experiment_command)
            _run_experiment_command(
                project=full_project,
                experiment_id="exploration-run",
                output=output / "exploration-run-output",
                workers=1,
                run_count=None,
                policies=(),
                dry_run=False,
                check_env=False,
                fixture_origin="http://127.0.0.1:8000",
                resume=False,
                profile_output=None,
                no_synthesis=False,
            )
        else:
            # Print next step
            # Prefer full project path
            hint_path = None
            for cand in generated_paths:
                if cand.name == "project.yaml":
                    hint_path = cand
                    break
            if hint_path is None and generated_paths:
                hint_path = generated_paths[0]
            if hint_path is not None:
                typer.echo(
                    f"next step: uxa run {hint_path} --experiment exploration-run"
                )

    try:
        asyncio.run(_explore_async())
    except SystemExit:
        raise
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        typer.echo(
            "explore interrupted; crawl checkpoint retained for resume "
            "(rerun the same command to continue)"
        )
        raise typer.Exit(130) from None
    except Exception as error:
        _exit_with_error(f"explore failed: {type(error).__name__}: {error}")


def _explore_origin_from_url(url: str) -> str:
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    host = parsed.hostname.lower() if parsed.hostname else ""  # type: ignore[union-attr]
    port = parsed.port
    if port is None or port == 443:
        return f"https://{host}"
    return f"https://{host}:{port}"


def _explore_bootstrap_applications(
    start_urls: tuple[str, ...],
) -> tuple[Application, ...]:
    origin_to_first_url: dict[str, str] = {}
    for url in start_urls:
        origin = _explore_origin_from_url(url)
        if origin not in origin_to_first_url:
            origin_to_first_url[origin] = url
    apps: list[Application] = []
    sorted_origins = sorted(origin_to_first_url.items())
    for idx, (origin, first_url) in enumerate(sorted_origins):
        app_id = (
            "exploration-app"
            if len(sorted_origins) == 1
            else f"exploration-app-{idx + 1}"
        )
        # Live vercel/next.js sites commonly load fonts and scripts from CDNs.
        # Include the start origin plus a minimal resource allowlist so the
        # run is not safety-blocked on the first external stylesheet.
        _default_resource_origins = (
            "https://fonts.googleapis.com",
            "https://fonts.gstatic.com",
        )
        # Keep origin first for determinism; dedup via dict preserves order.
        resource_origins = tuple(dict.fromkeys((origin, *_default_resource_origins)))
        version = ApplicationVersion(
            id=f"{app_id}-live",
            kind=ApplicationVersionKind.LIVE,
            label="Live",
            start_url=first_url,
            allowed_origins=resource_origins,
        )
        app = Application(
            id=app_id, name=f"Exploration App {idx + 1}", versions=(version,)
        )
        apps.append(app)
    return tuple(apps)


_EXPLORATION_CHECKPOINT_SCHEMA_VERSION = "exploration-checkpoint-v1"


@dataclass(frozen=True)
class _ExplorationCheckpoint:
    """Recovered crawl/synthesis progress from an interrupted explore run.

    ``crawl_complete`` is False when the crawl itself was interrupted; in
    that case ``frontier`` carries the saved position (completed pages,
    visited set, pending queue) to continue from.
    """

    corpus: CrawlCorpus
    suggestions: tuple[ScenarioSuggestion, ...] | None
    crawl_complete: bool = True
    frontier: CrawlFrontier | None = None


def _explore_checkpoint_root(output: Path) -> Path:
    return Path(output) / "exploration" / "checkpoint"


def _explore_spec_fingerprint(spec: ExplorationSpec) -> str:
    payload = {
        "start_urls": list(spec.start_urls),
        "depth": spec.depth,
        "max_pages": spec.max_pages,
        "max_scenarios": spec.max_scenarios,
        "settle_ms": spec.settle_ms,
        "allowed_origins": list(spec.allowed_origins),
    }
    return exploration_digest(payload)


def _explore_checkpoint_exists(output: Path) -> bool:
    root = _explore_checkpoint_root(output)
    return (root / "state.json").is_file() and (root / "payload.json").is_file()


def _atomic_json_write(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _explore_checkpoint_save(
    output: Path,
    spec: ExplorationSpec,
    corpus: CrawlCorpus,
    suggestions: tuple[ScenarioSuggestion, ...] | None,
    *,
    frontier: CrawlFrontier | None = None,
    crawl_complete: bool = True,
) -> None:
    """Persist crawl/synthesis progress as pending state for resume.

    Best-effort: checkpoint failures never fail an otherwise healthy run.
    Payload is written before state so a crash mid-write leaves no valid
    checkpoint. Called once per completed page during the crawl (with
    ``crawl_complete=False`` and the live frontier) and again after the
    full crawl / synthesis completes.
    """

    root = _explore_checkpoint_root(output)
    state = {
        "schema_version": _EXPLORATION_CHECKPOINT_SCHEMA_VERSION,
        "status": "pending",
        "fingerprint": _explore_spec_fingerprint(spec),
        "has_suggestions": suggestions is not None,
        "crawl_complete": crawl_complete,
        "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    payload: dict[str, object] = {
        "corpus": {
            "pages": [_json_data(page) for page in corpus.pages],
            "link_graph": {
                key: list(targets) for key, targets in corpus.link_graph.items()
            },
            "started_at": corpus.started_at,
            "corpus_digest": corpus.corpus_digest,
        },
        "suggestions": (
            [_json_data(suggestion) for suggestion in suggestions]
            if suggestions is not None
            else []
        ),
    }
    if frontier is not None:
        payload["frontier"] = {
            "queued": [[url, depth] for url, depth in frontier.queued],
            "visited": sorted(frontier.visited),
        }
    try:
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(root / "payload.json", payload)
        _atomic_json_write(root / "state.json", state)
    except (OSError, TypeError, ValueError):
        return


def _explore_checkpoint_save_progress(
    output: Path, spec: ExplorationSpec, frontier: CrawlFrontier
) -> None:
    """Persist one completed crawl page incrementally (best-effort)."""

    try:
        corpus = CrawlCorpus(
            pages=frontier.pages,
            link_graph=dict(frontier.link_graph),
            started_at=frontier.started_at,
        )
    except (TypeError, ValueError):
        return
    _explore_checkpoint_save(
        output, spec, corpus, None, frontier=frontier, crawl_complete=False
    )


def _frontier_from_checkpoint(
    value: object, corpus: CrawlCorpus
) -> CrawlFrontier | None:
    """Rebuild a saved frontier; None when absent or malformed."""

    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    payload = cast(dict[str, object], value)
    queued_raw = payload.get("queued")
    visited_raw = payload.get("visited")
    if not isinstance(queued_raw, list) or not isinstance(visited_raw, list):
        return None
    queued: list[tuple[str, int]] = []
    for entry in cast(list[object], queued_raw):
        if not isinstance(entry, (list, tuple)):
            return None
        pair = cast("list[object] | tuple[object, ...]", entry)
        if len(pair) != 2:
            return None
        url, depth = pair[0], pair[1]
        if not isinstance(url, str) or not isinstance(depth, int):
            return None
        queued.append((url, depth))
    visited: list[str] = []
    for item in cast(list[object], visited_raw):
        if not isinstance(item, str):
            return None
        visited.append(item)
    try:
        return CrawlFrontier(
            pages=corpus.pages,
            link_graph=dict(corpus.link_graph),
            visited=tuple(visited),
            queued=tuple(queued),
            started_at=corpus.started_at,
        )
    except (TypeError, ValueError):
        return None


def _explore_checkpoint_load(
    output: Path, spec: ExplorationSpec
) -> _ExplorationCheckpoint | None:
    """Return the checkpointed progress when it matches this spec exactly."""

    root = _explore_checkpoint_root(output)
    if not _explore_checkpoint_exists(output):
        return None
    try:
        state_value: object = json.loads(
            (root / "state.json").read_text(encoding="utf-8")
        )
        payload_value: object = json.loads(
            (root / "payload.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(state_value, dict) or not isinstance(payload_value, dict):
        return None
    state = cast(dict[str, object], state_value)
    payload = cast(dict[str, object], payload_value)
    if state.get("schema_version") != _EXPLORATION_CHECKPOINT_SCHEMA_VERSION:
        return None
    if state.get("fingerprint") != _explore_spec_fingerprint(spec):
        return None
    # Checkpoints written before per-page checkpointing lack crawl_complete;
    # they were only saved after a finished crawl, so default to complete.
    crawl_complete = state.get("crawl_complete") is not False
    corpus = _corpus_from_checkpoint(payload.get("corpus"))
    if corpus is None:
        return None
    frontier: CrawlFrontier | None = None
    if not crawl_complete and payload.get("frontier") is not None:
        frontier = _frontier_from_checkpoint(payload.get("frontier"), corpus)
    restored_suggestions: tuple[ScenarioSuggestion, ...] | None = None
    if state.get("has_suggestions") is True:
        raw_suggestions = payload.get("suggestions")
        if not isinstance(raw_suggestions, list):
            return None
        suggestion_entries = cast(list[object], raw_suggestions)
        rebuilt = [_suggestion_from_checkpoint(item) for item in suggestion_entries]
        if any(item is None for item in rebuilt):
            return None
        restored_suggestions = tuple(item for item in rebuilt if item is not None)
    return _ExplorationCheckpoint(
        corpus=corpus,
        suggestions=restored_suggestions,
        crawl_complete=crawl_complete,
        frontier=frontier,
    )


def _corpus_from_checkpoint(value: object) -> CrawlCorpus | None:
    if not isinstance(value, dict):
        return None
    payload = cast(dict[str, object], value)
    pages_raw = payload.get("pages")
    link_graph_raw = payload.get("link_graph")
    started_at = payload.get("started_at")
    if not isinstance(pages_raw, list) or not isinstance(link_graph_raw, dict):
        return None
    if not isinstance(started_at, str):
        return None
    pages: list[CrawlPage] = []
    for entry in cast(list[object], pages_raw):
        if not isinstance(entry, dict):
            return None
        page_payload = cast(dict[str, object], entry)
        viewport_id = page_payload.get("viewport_id")
        screenshot_digest = page_payload.get("screenshot_digest")
        headings_raw = page_payload.get("headings", ())
        links_raw = page_payload.get("discovered_links", ())
        elements_raw = page_payload.get("visible_elements", ())
        regions_raw = page_payload.get("region_labels", ())
        if not isinstance(headings_raw, list) or not isinstance(links_raw, list):
            return None
        if not isinstance(elements_raw, list) or not isinstance(regions_raw, list):
            return None
        try:
            pages.append(
                CrawlPage(
                    url=str(page_payload["url"]),
                    normalized_url=str(page_payload["normalized_url"]),
                    origin=str(page_payload["origin"]),
                    depth=int(cast(int, page_payload["depth"])),
                    title=str(page_payload["title"]),
                    headings=tuple(str(h) for h in cast(list[object], headings_raw)),
                    viewport_id=None if viewport_id is None else str(viewport_id),
                    screenshot_digest=(
                        None if screenshot_digest is None else str(screenshot_digest)
                    ),
                    discovered_links=tuple(
                        str(link) for link in cast(list[object], links_raw)
                    ),
                    visible_elements=tuple(
                        str(item) for item in cast(list[object], elements_raw)
                    ),
                    region_labels=tuple(
                        str(item) for item in cast(list[object], regions_raw)
                    ),
                )
            )
        except (KeyError, TypeError, ValueError):
            return None
    graph: dict[str, tuple[str, ...]] = {}
    for key, targets in cast(dict[object, object], link_graph_raw).items():
        if not isinstance(key, str) or not isinstance(targets, list):
            return None
        graph[key] = tuple(str(target) for target in cast(list[object], targets))
    declared_digest = payload.get("corpus_digest")
    try:
        return CrawlCorpus(
            pages=tuple(pages),
            link_graph=graph,
            started_at=started_at,
            corpus_digest=declared_digest if isinstance(declared_digest, str) else "",
        )
    except (TypeError, ValueError):
        return None


def _suggestion_from_checkpoint(value: object) -> ScenarioSuggestion | None:
    if not isinstance(value, dict):
        return None
    payload = cast(dict[str, object], value)
    verifier_raw = payload.get("verifier")
    target_raw = payload.get("evaluation_target")
    budget_raw = payload.get("budget")
    if not all(
        isinstance(part, dict) for part in (verifier_raw, target_raw, budget_raw)
    ):
        return None
    verifier = cast(dict[str, object], verifier_raw)
    target = cast(dict[str, object], target_raw)
    budget = cast(dict[str, object], budget_raw)
    labels_raw = target.get("labels_by_version")
    coverage_raw = payload.get("coverage", ())
    if not isinstance(labels_raw, dict):
        return None
    if coverage_raw is not None and not isinstance(coverage_raw, list):
        return None
    timeout_raw = budget.get("timeout_seconds")
    stall_raw = budget.get("stall_timeout_seconds")
    verifier_role = verifier.get("role")
    target_role = target.get("role")
    region_label = target.get("region_label")
    all_of_raw = verifier.get("all_of", ())
    try:
        built_budget = Budget(
            max_steps=int(cast(int, budget["max_steps"])),
            max_observations=int(cast(int, budget["max_observations"])),
            max_interactions=int(cast(int, budget["max_interactions"])),
            timeout_seconds=(
                None
                if timeout_raw is None
                else float(cast("int | float | str", timeout_raw))
            ),
            stall_timeout_seconds=(
                None
                if stall_raw is None
                else float(cast("int | float | str", stall_raw))
            ),
            max_model_calls=int(cast(int, budget.get("max_model_calls", 64))),
        )
        built_verifier = VisibleResultVerifierSpec(
            type=str(verifier.get("type", "visible-result")),
            text=str(verifier.get("text", "")),
            role=None if verifier_role is None else str(verifier_role),
            all_of=tuple(str(item) for item in cast(list[object], all_of_raw or ())),
        )
        built_target = ScenarioEvaluationTarget(
            labels_by_version={
                str(version): str(label)
                for version, label in cast(dict[object, object], labels_raw).items()
            },
            role=None if target_role is None else str(target_role),
            region_label=None if region_label is None else str(region_label),
        )
        return ScenarioSuggestion(
            id=str(payload["id"]),
            name=str(payload["name"]),
            goal=str(payload["goal"]),
            start_url=str(payload["start_url"]),
            verifier=built_verifier,
            evaluation_target=built_target,
            budget=built_budget,
            rationale=str(payload["rationale"]),
            coverage=tuple(
                str(item) for item in cast(list[object], coverage_raw or ())
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _explore_checkpoint_clear(output: Path) -> None:
    shutil.rmtree(_explore_checkpoint_root(output), ignore_errors=True)


def _persist_unavailable_exploration_attempt(
    *,
    output: Path,
    corpus: CrawlCorpus,
    personas: tuple[Any, ...],
    spec: ExplorationSpec,
    model: str,
    reason: str,
) -> None:
    """Best-effort record that synthesis could not produce suggestions.

    Persists the real crawl evidence with status ``unavailable`` plus a
    limitation note; persistence failures are swallowed so the loud exit
    reason below always reaches the operator.
    """

    try:
        store = ExplorationArtifactStore(output)
        store.write_attempt(
            corpus,
            suggestions=(),
            curated=(),
            persona_set=personas,
            spec=spec,
            model=model,
            prompt_version="exploration-synthesis-v1",
            status="unavailable",
            limitation=f"scenario synthesis unavailable: {reason}",
        )
    except Exception:
        return


async def _explore_run_crawler(
    spec: ExplorationSpec,
    policy: PageSettlementPolicy,
    *,
    on_frontier: Callable[[CrawlFrontier], Awaitable[None]] | None = None,
    resume_from: CrawlFrontier | None = None,
) -> CrawlCorpus:
    """Crawl via Playwright; any failure propagates (no fabricated corpus).

    ``on_frontier`` fires after every completed page so the caller can
    persist crash-recovery checkpoints incrementally; ``resume_from``
    continues an interrupted crawl from its saved position.
    """

    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            crawler = ExplorationCrawler(page=page, policy=policy)
            corpus = await crawler.crawl(
                spec, on_frontier=on_frontier, resume_from=resume_from
            )
            await page.close()
            await context.close()
            return corpus
        finally:
            await browser.close()


async def _explore_run_synthesizer(
    corpus: CrawlCorpus, max_scenarios: int, settings: OpenAICompatibleSettings
) -> tuple[ScenarioSuggestion, ...]:
    """Synthesize suggestions via the model client; failures propagate.

    An empty suggestion list from the model is a valid outcome; missing
    model connectivity or schema failures raise so the caller can persist
    an ``unavailable`` attempt instead of fabricating scenarios.
    """

    import httpx as _httpx

    http_client = _httpx.AsyncClient(timeout=settings.timeout_seconds)
    try:
        client = create_structured_model_client(
            settings,
            http_client=http_client,
            call_limiter=asyncio.Semaphore(settings.max_concurrent_calls),
        )
        synthesizer = ExplorationSynthesizer(
            client=client, model=settings.cognitive_model
        )
        result = await synthesizer.suggest(corpus, max_scenarios=max_scenarios)
        # Bubble operational failures instead of silently returning 0.
        # The synthesizer distinguishes "ok with zero" (rare but valid)
        # from "unavailable/invalid with zero" (provider 500, schema drift).
        # Only the latter must be surfaced as an error so the user sees why
        # the review would be empty.
        if not result.suggestions and result.status != "ok":
            reason = (
                "; ".join(result.limitations) if result.limitations else result.status
            )
            raise RuntimeError(reason)
        return tuple(result.suggestions)
    finally:
        await http_client.aclose()


def _write_yaml_document(path: Path, payload: object, *, sort_keys: bool) -> Path:
    """Canonical YAML writer for materialized exploration project files.

    Single serialization path (safe_dump, explicit key order policy) so all
    generated YAML files share one format; write errors propagate to the
    caller instead of being swallowed by fallback chains.
    """

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, sort_keys=sort_keys, allow_unicode=False),
            encoding="utf-8",
        )
    except OSError as error:
        raise OSError(f"cannot write {path}: {error}") from error
    return path


def _explore_materialize_project(
    *,
    output: Path,
    base_path: Path | None,
    loaded: LoadedProject | None,
    corpus: CrawlCorpus,
    suggestions: tuple[ScenarioSuggestion, ...],
    curated: Sequence[ScenarioSuggestion | dict[str, Any]],
    existing_personas: tuple[Persona, ...],
    existing_applications: tuple[Application, ...],
    persona_selection: PersonaSelectionPayload | None,
    spec: ExplorationSpec,
    attempt_path: Path,
) -> list[Path]:
    # Normalize curated to the canonical scenario-dict shape (the review
    # server's ScenarioPayload dump); model suggestions convert through the
    # same canonical schema. This is the single boundary conversion.
    curated_list: list[dict[str, Any]] = [
        _suggestion_to_dict(item) if isinstance(item, ScenarioSuggestion) else item
        for item in curated
    ]
    # Determine persona ids for experiment and custom personas
    persona_ids_for_exp: list[str] = []
    custom_personas_to_add: list[dict[str, Any]] = []
    if persona_selection is not None:
        if persona_selection.mode == "custom":
            custom_persona = persona_selection.custom_persona
            if custom_persona is not None:
                custom_dict = custom_persona.model_dump()
                custom_personas_to_add = [custom_dict]
                persona_ids_for_exp = [
                    str(custom_dict["id"]).strip() or "custom-persona"
                ]
        elif persona_selection.mode in {"existing", "suggested"}:
            persona_ids_for_exp = [
                pid.strip() for pid in persona_selection.persona_ids if pid.strip()
            ]
            if not persona_ids_for_exp:
                persona_ids_for_exp = [p.id for p in existing_personas]
    else:
        # auto_accept or no selection: use existing personas
        for p in existing_personas:
            pid = p.id.strip()
            if pid:
                persona_ids_for_exp.append(pid)
        if not persona_ids_for_exp and existing_personas:
            persona_ids_for_exp = [existing_personas[0].id]
        if not persona_ids_for_exp:
            persona_ids_for_exp = ["default-explorer"]
            custom_personas_to_add = [
                {
                    "id": "default-explorer",
                    "name": "Default Explorer",
                    "working_memory_capacity": 4,
                    "initial_confidence": 0.55,
                    "initial_frustration": 0.10,
                    "abandonment_threshold": 0.75,
                    "attention_temperature": 1.0,
                }
            ]
    # Resolve applications mapping for each curated scenario: find app version id by origin
    # Build origin -> app version id map
    app_version_by_origin: dict[str, str] = {}
    app_version_ids_all: set[str] = set()
    for app in existing_applications:
        for ver in app.versions:
            if not ver.id or ver.start_url is None:
                continue
            try:
                origin = _explore_origin_from_url(
                    benchmark_normalize_crawl_url(ver.start_url)
                )
            except ValueError:
                continue
            app_version_by_origin[origin] = ver.id
            app_version_ids_all.add(ver.id)
    # Fallback if no mapping: use first app's first version with a start URL
    if not app_version_by_origin and existing_applications:
        first_ver = existing_applications[0].versions[0]
        if first_ver.start_url is not None:
            origin = _explore_origin_from_url(
                benchmark_normalize_crawl_url(first_ver.start_url)
            )
            app_version_by_origin[origin] = first_ver.id
            app_version_ids_all.add(first_ver.id)
    # For each curated, determine app_version_id
    curated_full_scenarios: list[dict[str, Any]] = []
    used_app_version_ids: set[str] = set()
    for item in curated_list:
        start_url = str(item.get("start_url", "")).strip()
        # Resolve origin
        try:
            origin = _explore_origin_from_url(
                exploration_normalize_crawl_url(start_url)
            )
        except ValueError:
            try:
                origin = _explore_origin_from_url(
                    benchmark_normalize_crawl_url(start_url)
                )
            except ValueError:
                origin = ""
        app_ver_id = app_version_by_origin.get(origin)
        if app_ver_id is None:
            # fallback to first available
            if app_version_ids_all:
                app_ver_id = min(app_version_ids_all)
            elif existing_applications:
                app_ver_id = existing_applications[0].versions[0].id
            else:
                app_ver_id = "exploration-app-live"
        used_app_version_ids.add(app_ver_id)
        # Convert to full scenario dict
        full = _explore_curated_to_full_scenario(
            item, app_ver_id, tuple(persona_ids_for_exp)
        )
        curated_full_scenarios.append(full)
    # Build experiment
    experiment: dict[str, Any] = {
        "id": "exploration-run",
        "name": "Exploration Run",
        "scenario_ids": [s["id"] for s in curated_full_scenarios],
        "application_version_ids": sorted(used_app_version_ids)
        if used_app_version_ids
        else [
            next(iter(sorted(app_version_ids_all)))
            if app_version_ids_all
            else "exploration-app-live"
        ],
        "persona_ids": persona_ids_for_exp,
        "policies": ["full-list"],
        "run_count": 1,
    }
    generated_paths: list[Path] = []
    # Ensure exploration directory exists
    exploration_dir = output / "exploration"
    exploration_dir.mkdir(parents=True, exist_ok=True)
    # Write fragment file at exploration/project.fragment.yaml (top-level fragment) and at attempt dir already exists
    fragment_payload: dict[str, Any] = {
        "id": "exploration-fragment",
        "scenarios": curated_full_scenarios,
        "experiments": [experiment],
    }
    if custom_personas_to_add:
        fragment_payload["personas"] = custom_personas_to_add
    # Write top-level fragment
    fragment_path = _write_yaml_document(
        exploration_dir / "project.fragment.yaml", fragment_payload, sort_keys=True
    )
    generated_paths.append(fragment_path)
    # Also write exploration/generated.yaml for compatibility
    # For generated.yaml, produce full project if bootstrap else merged
    if loaded is not None and base_path is not None:
        # Merge with base project: read base YAML and append. The project was
        # already parsed and validated by the loader at command start, so a
        # failure here is an I/O or encoding problem and must be loud.
        try:
            base_raw_value: object = yaml.safe_load(
                base_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, yaml.YAMLError):
            _exit_with_error(f"cannot re-read base project {base_path} for merge")
        base_raw: dict[str, Any] = (
            cast("dict[str, Any]", base_raw_value)
            if isinstance(base_raw_value, dict)
            else {}
        )
        # Ensure lists
        base_scenarios: list[Any] = list(base_raw.get("scenarios", []) or [])
        base_experiments: list[Any] = list(base_raw.get("experiments", []) or [])
        base_personas: list[Any] = list(base_raw.get("personas", []) or [])
        # Append new scenarios (keep original)
        merged_scenarios: list[Any] = base_scenarios + curated_full_scenarios
        # Append experiment (avoid duplicate id)
        existing_exp_ids: set[Any] = {
            cast("dict[str, Any]", e).get("id")
            for e in base_experiments
            if isinstance(e, dict)
        }
        merged_experiments: list[Any]
        if experiment["id"] not in existing_exp_ids:
            merged_experiments = base_experiments + [experiment]
        else:
            merged_experiments = base_experiments
            # replace? keep existing and add with new id variant
            experiment_alt = dict(experiment)
            experiment_alt["id"] = "exploration-run-2"
            merged_experiments = base_experiments + [experiment_alt]
            experiment = experiment_alt
            # Update fragment to reflect id?
            fragment_payload["experiments"] = [experiment]
            _write_yaml_document(fragment_path, fragment_payload, sort_keys=True)
        # Append custom personas if any not already present
        existing_persona_ids: set[Any] = {
            cast("dict[str, Any]", p).get("id")
            for p in base_personas
            if isinstance(p, dict)
        }
        for cp in custom_personas_to_add:
            if cp.get("id") not in existing_persona_ids:
                base_personas.append(cp)
        merged: dict[str, Any] = dict(base_raw)
        merged["scenarios"] = merged_scenarios
        merged["experiments"] = merged_experiments
        if custom_personas_to_add:
            merged["personas"] = base_personas
        # Append exploration live applications missing from base so new scenarios resolve
        raw_apps_value: Any = base_raw.get("applications", []) or []
        base_apps_raw: list[dict[str, Any]] = [
            cast("dict[str, Any]", entry)
            for entry in raw_apps_value
            if isinstance(entry, dict)
        ]
        base_app_ids: set[Any] = {a.get("id") for a in base_apps_raw}
        extra_apps: list[dict[str, Any]] = []
        for app in existing_applications:
            app_id = app.id.strip()
            if not app_id or app_id in base_app_ids:
                continue
            vers_payload_extra: list[dict[str, Any]] = []
            for ver in app.versions:
                version_entry: dict[str, Any] = {
                    "id": ver.id,
                    "kind": ver.kind.value,
                    "label": ver.label,
                    "start_url": ver.start_url,
                }
                if ver.allowed_origins:
                    version_entry["allowed_origins"] = list(ver.allowed_origins)
                vers_payload_extra.append(version_entry)
            extra_apps.append(
                {
                    "id": app_id,
                    "name": app.name,
                    "versions": vers_payload_extra,
                }
            )
        if extra_apps:
            merged["applications"] = base_apps_raw + extra_apps
        # Write full project.yaml at output/project.yaml plus compatibility
        # copies (exploration/generated.yaml, exploration/project.yaml).
        full_project_path = _write_yaml_document(
            output / "project.yaml", merged, sort_keys=False
        )
        generated_paths.append(full_project_path)
        generated_paths.append(
            _write_yaml_document(
                exploration_dir / "generated.yaml", merged, sort_keys=False
            )
        )
        generated_paths.append(
            _write_yaml_document(
                exploration_dir / "project.yaml", merged, sort_keys=False
            )
        )
    else:
        # Bootstrap: create full project from scratch
        # Build applications payload for YAML
        apps_payload: list[dict[str, Any]] = []
        for app in existing_applications:
            vers_payload: list[dict[str, Any]] = []
            for ver in app.versions:
                origin = None
                try:
                    if ver.start_url:
                        origin = _explore_origin_from_url(
                            benchmark_normalize_crawl_url(ver.start_url)
                        )
                except Exception:
                    origin = None
                payload: dict[str, Any] = {
                    "id": ver.id,
                    "kind": ver.kind.value,
                    "label": ver.label,
                    "start_url": ver.start_url,
                }
                if ver.allowed_origins:
                    # Preserve the bootstrap resource allowlist (start origin
                    # plus CDN resources the crawl itself relied on). Dropping
                    # these makes every run safety-block on first subresource.
                    payload["allowed_origins"] = list(ver.allowed_origins)
                elif origin:
                    payload["allowed_origins"] = [origin]
                vers_payload.append(payload)
            apps_payload.append(
                {
                    "id": app.id,
                    "name": app.name,
                    "versions": vers_payload,
                }
            )
        personas_payload: list[dict[str, Any]] = [
            {
                "id": p.id,
                "name": p.name,
                "working_memory_capacity": p.working_memory_capacity,
                "initial_confidence": p.initial_confidence,
                "initial_frustration": p.initial_frustration,
                "abandonment_threshold": p.abandonment_threshold,
                "attention_temperature": p.attention_temperature,
            }
            for p in existing_personas
        ]
        # Include custom personas if any
        for cp in custom_personas_to_add:
            if cp["id"] not in {pp["id"] for pp in personas_payload}:
                personas_payload.append(cp)
        full_project = {
            "id": "exploration-generated",
            "name": "Exploration Generated",
            "applications": apps_payload,
            "scenarios": curated_full_scenarios,
            "personas": personas_payload,
            "experiments": [experiment],
        }
        generated_paths.append(
            _write_yaml_document(output / "project.yaml", full_project, sort_keys=False)
        )
        generated_paths.append(
            _write_yaml_document(
                exploration_dir / "generated.yaml", full_project, sort_keys=False
            )
        )
        generated_paths.append(
            _write_yaml_document(
                exploration_dir / "project.yaml", full_project, sort_keys=False
            )
        )
        # Also write generated.yaml at output root for convenience
        generated_paths.append(
            _write_yaml_document(
                output / "generated.yaml", full_project, sort_keys=False
            )
        )
    # Materialize fragment at output root as well; failures are loud.
    root_fragment_path = _write_yaml_document(
        output / "project.fragment.yaml", fragment_payload, sort_keys=True
    )
    if root_fragment_path not in generated_paths:
        generated_paths.append(root_fragment_path)
    return generated_paths


def _budget_int_from_text(value: Any, default: int) -> int:
    """``int(v) if str(v).strip() else default`` (blank/missing falls back)."""

    return int(value) if str(value).strip() else default


def _budget_int_if_set(value: Any, default: int) -> int:
    """``int(v) if v else default`` (falsy values, including 0, fall back)."""

    return int(value) if value else default


def _budget_float_seconds(value: Any, default: float) -> float:
    """``float(v) if v not in (None, '') else default``, mapping 0 to default."""

    if value is None or value == "":
        return default
    seconds = float(value)
    return default if seconds == 0 else seconds


def _budget_float_seconds_or_none(value: Any) -> float | None:
    """Parsed seconds or None; runs without a wall cap rely on stall cutoffs."""

    if value is None or value == "":
        return None
    seconds = float(value)
    return None if seconds == 0 else seconds


def _explore_curated_to_full_scenario(
    item: dict[str, Any], app_version_id: str, eligible_personas: tuple[str, ...]
) -> dict[str, Any]:
    sc_id = str(item.get("id", "")).strip() or f"explore-{uuid4().hex[:6]}"
    name = str(item.get("name", sc_id)).strip() or sc_id
    goal = str(item.get("goal", "")).strip() or name
    raw_verifier = item.get("verifier", {})
    verifier: dict[str, Any]
    if isinstance(raw_verifier, dict):
        verifier = cast("dict[str, Any]", raw_verifier)
    else:
        verifier = {"type": "visible-result", "text": str(raw_verifier)}
    if verifier.get("type") != "visible-result":
        verifier["type"] = "visible-result"
    if not str(verifier.get("text", "")).strip():
        verifier["text"] = name or "visible result"
    if "all_of" not in verifier or verifier["all_of"] is None:
        verifier["all_of"] = []
    # evaluation_target
    et_raw = item.get("evaluation_target", {})
    et: dict[str, Any] = (
        cast("dict[str, Any]", et_raw) if isinstance(et_raw, dict) else {}
    )
    labels_by_version: dict[str, str] = {}
    role_val: str | None = None
    region_label: str | None = None
    if isinstance(et.get("labels_by_version"), dict) and et["labels_by_version"]:
        labels_by_version = {
            str(k): str(v)
            for k, v in cast("dict[Any, Any]", et["labels_by_version"]).items()
            if str(v).strip()
        }
    elif isinstance(et.get("label"), str) and et["label"].strip():
        labels_by_version = {
            app_version_id: et["label"].strip(),
            "live": et["label"].strip(),
        }
    elif isinstance(et.get("labels"), dict):
        labels_by_version = {
            str(k): str(v) for k, v in cast("dict[Any, Any]", et["labels"]).items()
        }
    # role
    if isinstance(et.get("role"), str) and et["role"].strip():
        role_val = et["role"].strip()
    if isinstance(et.get("region_label"), str) and et["region_label"].strip():
        region_label = et["region_label"].strip()
    if not labels_by_version:
        labels_by_version = {app_version_id: name, "live": name}
    if app_version_id not in labels_by_version:
        first = next(iter(labels_by_version.values()))
        labels_by_version[app_version_id] = first
    if "live" not in labels_by_version:
        labels_by_version["live"] = next(iter(labels_by_version.values()))
    evaluation_target: dict[str, Any] = {"labels_by_version": labels_by_version}
    if role_val:
        evaluation_target["role"] = role_val
    if region_label:
        evaluation_target["region_label"] = region_label
    # budget
    budget_raw = item.get("budget", {})
    budget: dict[str, Any] = (
        cast("dict[str, Any]", budget_raw) if isinstance(budget_raw, dict) else {}
    )
    budget_norm: dict[str, Any] = {
        "max_steps": _budget_int_from_text(budget.get("max_steps", ""), 20),
        "max_observations": _budget_int_if_set(budget.get("max_observations"), 18),
        "max_interactions": _budget_int_if_set(budget.get("max_interactions"), 8),
        # Explorer-generated scenarios run progress-based: no wall-clock cap.
        "timeout_seconds": _budget_float_seconds_or_none(budget.get("timeout_seconds")),
        "stall_timeout_seconds": _budget_float_seconds(
            budget.get("stall_timeout_seconds"), 90
        ),
        "max_model_calls": _budget_int_if_set(budget.get("max_model_calls"), 32),
    }
    return {
        "id": sc_id,
        "name": name,
        "goal": goal,
        "application_version_ids": [app_version_id],
        "start_state": "dashboard",
        "fixture_inputs": {},
        "budget": budget_norm,
        "verifier": verifier,
        "safeguards": [],
        "eligible_persona_ids": list(eligible_personas)
        if eligible_personas
        else ["default-explorer"],
        "expected_evidence": ["target-discovery", "verified-completion"],
        "evaluation_target": evaluation_target,
        "viewport": {"width": 1280, "height": 800},
    }


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


def _parse_issue_skill(flag: str) -> tuple[str, str]:
    finding_id, separator, set_name = flag.partition("=")
    if not separator or not finding_id.strip() or not set_name.strip():
        raise typer.BadParameter(
            "--issue-skill must be FINDING_ID=SET (the skill set for one issue)"
        )
    return finding_id, set_name


@app.command()
def export(
    report: Path = typer.Option(
        ...,
        "--report",
        exists=True,
        file_okay=False,
        help="Experiment output directory containing report.html",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Package directory (defaults under the git-ignored .uxa-output/ dir)",
    ),
    all_issues: bool = typer.Option(False, "--all"),
    finding: list[str] = typer.Option([], "--finding", help="Finding ID"),
    exclude: list[str] = typer.Option([], "--exclude", help="Finding ID"),
    skill_set: list[str] = typer.Option([], "--skill-set", help="Set for all issues"),
    issue_skill: list[str] = typer.Option(
        [], "--issue-skill", help="FINDING_ID=SET per-issue override"
    ),
    skill_sets_file: Path | None = typer.Option(None, "--skill-sets"),
    skills_note: str | None = typer.Option(None, "--skills-note"),
    notes: Path | None = typer.Option(
        None, "--notes", help="Reproduction notes file (embedded verbatim)"
    ),
) -> None:
    """Export selected issues as an LLM-optimized fix package."""
    from ux_analyzer.export.catalog import build_catalog, parse_issue_flags
    from ux_analyzer.export.render import ExportContext
    from ux_analyzer.export.skills import (
        load_skill_sets,
        resolve_assignments,
        resolve_skill_sets_path,
    )
    from ux_analyzer.export.writer import ExportError, write_export

    view = load_report_findings(report)
    if not view["findings"]:
        raise typer.BadParameter("nothing to export: the report contains no issues")
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    catalog = build_catalog(view)
    sets = load_skill_sets(resolve_skill_sets_path(skill_sets_file))
    try:
        if interactive:
            from ux_analyzer.export.interactive import run_interactive_export

            result = run_interactive_export(catalog, sets)
            if result is None:
                raise typer.Exit(code=1)
            selected = result.issues
            default_name = result.default_skill_set
            per_issue = result.per_issue_skills
        else:
            selected = parse_issue_flags(
                all_issues=all_issues,
                findings=finding,
                exclude=exclude,
                catalog=catalog,
            )
            default_name = (
                skill_set[0]
                if skill_set
                else next((s.name for s in sets if s.is_default), None)
            )
            per_issue = dict(_parse_issue_skill(flag) for flag in issue_skill)
            resolve_assignments(sets, default_name, per_issue)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    assignments = {
        issue.finding_id: default_name for issue in selected if default_name is not None
    }
    assignments.update(per_issue)
    notes_text: str | None = None
    if notes is not None:
        if not notes.is_file():
            raise typer.BadParameter(f"notes file does not exist: {notes}")
        notes_text = notes.read_text(encoding="utf-8")
    now = datetime.now(UTC)
    package_dir = (
        out
        if out is not None
        # Generated artifacts live under the git-ignored .uxa-output/ dir;
        # explicit --out still wins for custom locations.
        else Path(".uxa-output") / f"fix-export-{now:%Y%m%d-%H%M%S}"
    )
    context = ExportContext(
        report_path=report,
        exported_at=now.isoformat(),
        tool_version=__version__,
        synthesis_status=str(view["synthesis_status"]),
        using_fallback=bool(view["using_fallback"]),
        attempt_id=view["attempt_id"],
        issues=tuple(selected),
        assignments=assignments,
        skill_sets=sets,
        skills_note=skills_note,
        reproduction_notes=notes_text,
    )
    try:
        result = write_export(package_dir, context)
    except ExportError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"export package: {result.package_dir}")
    typer.echo(f"issues: {result.issue_count}")
    typer.echo(f"assets: {result.asset_count}")
    for issue in selected:
        if issue.has_unresolved_evidence:
            typer.echo(
                f"warning: issue '{issue.finding_id}' has evidence unavailable; "
                "it is exported with 'evidence unavailable' markers.",
                err=True,
            )


def _referenced_origins_sync_cli(url: str) -> frozenset[str]:
    from ux_analyzer.analysis.referenced_origins import referenced_origins_sync

    return referenced_origins_sync(url)


def _negotiate_live_origin_gaps(
    loaded: LoadedProject,
    *,
    allow_origin: Sequence[str],
    yes: bool,
) -> dict[str, frozenset[str]]:
    """Discover referenced foreign origins and ask once before runs start.

    Non-interactive sessions (piped stdio, CI, ``--workers`` batches) keep the
    fail-closed default: gaps are reported but nothing is granted implicitly.
    """

    gaps = _collect_origin_gaps(loaded)
    if not allow_origin and not gaps:
        return {}
    if allow_origin:
        live_version_ids = {
            version.id
            for application in loaded.project.applications
            for version in application.versions
            if version.kind is ApplicationVersionKind.LIVE
        }
        flagged: dict[str, frozenset[str]] = {
            version_id: frozenset(allow_origin)
            for version_id in sorted(live_version_ids)
        }
        if not gaps:
            return flagged
        merged_gaps = {
            version_id: gap - set(allow_origin) for version_id, gap in gaps.items()
        }
        selected = _select_extra_origins(
            {key: value for key, value in merged_gaps.items() if value},
            auto_yes=yes,
            prompt=input,
            echo=typer.echo,
        )
        merged: dict[str, frozenset[str]] = dict(flagged)
        for version_id, origins in selected.items():
            merged[version_id] = frozenset(
                {*merged.get(version_id, frozenset()), *origins}
            )
        return merged
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
        blocked = sorted({origin for gap in gaps.values() for origin in gap})
        typer.echo(
            "warning: live targets reference non-allowlisted origins "
            f"({', '.join(blocked)}); these requests will safety-block. "
            "Re-run interactively or pass --yes / --allow-origin.",
            err=True,
        )
        return {}
    return _select_extra_origins(gaps, auto_yes=yes, prompt=input, echo=typer.echo)


def _run_experiment_command(
    *,
    project: Path,
    experiment_id: str,
    output: Path | None,
    workers: int,
    run_count: int | None,
    policies: Sequence[str],
    dry_run: bool,
    check_env: bool,
    fixture_origin: str,
    resume: bool,
    profile_output: Path | None,
    no_synthesis: bool,
    allow_origin: Sequence[str] = (),
    yes: bool = False,
) -> None:
    if workers <= 0:
        _exit_with_error("workers must be greater than zero")
    extra_resource_origins: Mapping[str, Sequence[str]] = {}
    if not dry_run:
        preflight_loaded = _load_project_or_exit(project)
        extra_resource_origins = _negotiate_live_origin_gaps(
            preflight_loaded,
            allow_origin=allow_origin,
            yes=yes,
        )
    matrix = _resolve_matrix_or_exit(
        project,
        experiment_id,
        run_count=run_count,
        policies=policies,
        extra_resource_origins=extra_resource_origins or None,
    )
    output = _resolve_default_output(output, matrix.loaded.project.id)
    settings: OpenAICompatibleSettings | None = None
    if check_env or not dry_run:
        # Report synthesis is best-effort. Keep missing report-role settings
        # from preventing deterministic experiment execution and fallback rendering.
        settings = _model_settings_or_exit(
            report_synthesis_enabled=(
                check_env and _synthesis_gate(matrix.loaded) and not no_synthesis
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
    _write_ux_audit(output=output, results=result.results)
    _write_pagespeed(output=output, results=result.results)
    _run_auto_redesign(output=output, results=result.results)
    if _synthesis_gate(matrix.loaded) and not no_synthesis:
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
    extra_resource_origins: Mapping[str, Sequence[str]] | None = None,
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
    if extra_resource_origins:
        specs = _with_extra_resource_origins(specs, extra_resource_origins)
    return _ResolvedMatrix(loaded=loaded, definition=definition, specs=specs)


def _with_extra_resource_origins(
    specs: Sequence[RunSpec],
    grants: Mapping[str, Sequence[str]],
) -> tuple[RunSpec, ...]:
    """Grant session-only resource origins without changing the YAML digest."""

    patched: list[RunSpec] = []
    for spec in specs:
        grant = grants.get(spec.application_version.id)
        if not grant:
            patched.append(spec)
            continue
        merged = tuple(
            dict.fromkeys((*spec.application_version.allowed_origins, *grant))
        )
        if merged == spec.application_version.allowed_origins:
            patched.append(spec)
            continue
        patched.append(
            replace(
                spec,
                application_version=replace(
                    spec.application_version, allowed_origins=merged
                ),
            )
        )
    return tuple(patched)


def _collect_origin_gaps(
    loaded: LoadedProject,
    *,
    fetcher: Callable[[str], frozenset[str]] | None = None,
) -> dict[str, frozenset[str]]:
    """Map live version IDs to referenced origins their allowlist would block."""

    resolve = fetcher or _referenced_origins_sync_cli
    fetched_by_url: dict[str, frozenset[str]] = {}
    gaps: dict[str, frozenset[str]] = {}
    for application in loaded.project.applications:
        for version in application.versions:
            if version.kind is not ApplicationVersionKind.LIVE:
                continue
            if version.start_url is None:
                continue
            start_origin = _explore_origin_from_url(version.start_url)
            if version.start_url not in fetched_by_url:
                fetched_by_url[version.start_url] = resolve(version.start_url)
            allowed = {start_origin, *version.allowed_origins}
            missing = fetched_by_url[version.start_url] - allowed
            if missing:
                gaps[version.id] = frozenset(sorted(missing))
    return gaps


def _select_extra_origins(
    gaps: Mapping[str, frozenset[str]],
    *,
    auto_yes: bool,
    prompt: Callable[[str], str] = input,
    echo: Callable[[object], None] = print,
) -> dict[str, frozenset[str]]:
    """Ask the operator once which referenced origins the browser may load."""

    if auto_yes:
        return {version_id: set(gap) for version_id, gap in gaps.items()}
    echo("live targets reference origins outside the configured allowlist:")
    ordered: dict[str, list[str]] = {}
    for version_id in sorted(gaps):
        ordered[version_id] = sorted(gaps[version_id])
        for origin in ordered[version_id]:
            echo(f"  [{version_id}] {origin}")
    echo("blocked requests abort runs with outcome 'safety-blocked'.")
    for _attempt in range(3):
        answer = (
            prompt(
                "allow these origins for this run? [a]ll / [s]elect numbers / [n]one: "
            )
            .strip()
            .casefold()
        )
        if answer in {"a", "all", "y", "yes"}:
            return {version_id: set(gap) for version_id, gap in gaps.items()}
        if answer in {"n", "none", ""}:
            return {}
        tokens = answer.replace(",", " ").split()
        try:
            indexes = {int(token) for token in tokens}
        except ValueError:
            echo("enter 'a', 's', 'n', or space-separated numbers.")
            continue
        selected: dict[str, frozenset[str]] = {}
        flat: list[tuple[str, str]] = [
            (version_id, origin)
            for version_id in sorted(gaps)
            for origin in sorted(gaps[version_id])
        ]
        for index in indexes:
            if 1 <= index <= len(flat):
                version_id, origin = flat[index - 1]
                selected.setdefault(version_id, set()).add(origin)
        if indexes and len(selected) == 0:
            echo(f"numbers must be between 1 and {len(flat)}.")
            continue
        return {
            version_id: frozenset(origins) for version_id, origins in selected.items()
        }
    echo("no selection after three attempts; continuing without extra origins.")
    return {}


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
    stalls = {spec.scenario.budget.stall_timeout_seconds for spec in matrix.specs}
    if stalls == {None}:
        stall_summary = "none"
    elif None in stalls or len(stalls) != 1:
        stall_summary = "mixed"
    else:
        stall_summary = f"{next(iter(stalls)):g}s"
    typer.echo(f"stall timeout (no-progress cutoff): {stall_summary}")
    typer.echo("matrix:")
    for (scenario, version, persona, policy, provider), count in sorted(cells.items()):
        typer.echo(
            f"- {scenario}/{version}/{persona}/{policy}: {count} runs "
            f"(prominence-provider={provider})"
        )


def _synthesis_gate(loaded: LoadedProject) -> bool:
    """Decide whether report synthesis runs for this project.

    ``UXA_REPORT_SYNTHESIS_ENABLED`` overrides the YAML gate when set to a
    non-empty value (truthy values enable, falsey disable); empty or unset
    keeps current behavior — the project's
    ``evaluation.report_synthesis.enabled`` decides. Treating an empty value
    as unset matters because ``load_dotenv`` turns a ``VAR=`` line in
    ``.env.example`` into an empty string, which must not silently flip the
    gate.
    """

    raw = os.environ.get("UXA_REPORT_SYNTHESIS_ENABLED", "").strip()
    if raw:
        return raw.lower() not in {"0", "false", "no", "off"}
    return loaded.runtime.report_synthesis.enabled


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
                prompt_version="cognitive-v3",
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
                "cognitive": "cognitive-v3",
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
    summary_path = output / "experiment.json"
    _atomic_write_experiment_json(summary_path, summary)
    return summary_path


class _BoundedCanonicalJson:
    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError(f"experiment summary exceeds {max_bytes} bytes")
        self._max_bytes = max_bytes
        self._content_limit = max_bytes - 1
        self._size = 0
        self._chunks: list[bytes] = []
        self._pending = bytearray()

    def encode(self, value: object) -> tuple[bytes, ...]:
        self._emit_value(value)
        self._pending.extend(b"\n")
        if self._pending:
            self._chunks.append(bytes(self._pending))
            self._pending.clear()
        return tuple(self._chunks)

    def _reserve(self, length: int) -> None:
        if self._size + length > self._content_limit:
            raise ValueError(f"experiment summary exceeds {self._max_bytes} bytes")
        self._size += length

    def _emit_ascii(self, value: str) -> None:
        self._emit_ascii_range(value, 0, len(value))

    def _emit_ascii_range(self, value: str, start: int, end: int) -> None:
        self._reserve(end - start)
        while start < end:
            available = _EXPERIMENT_JSON_CHUNK_BYTES - len(self._pending)
            chunk_end = min(start + available, end)
            self._pending.extend(value[start:chunk_end].encode("ascii"))
            start = chunk_end
            if len(self._pending) == _EXPERIMENT_JSON_CHUNK_BYTES:
                self._chunks.append(bytes(self._pending))
                self._pending.clear()

    def _emit_string(self, value: str) -> None:
        self._emit_ascii('"')
        run_start = 0
        for index, character in enumerate(value):
            codepoint = ord(character)
            escaped: str | None = None
            if character == '"':
                escaped = '\\"'
            elif character == "\\":
                escaped = "\\\\"
            elif character == "\b":
                escaped = "\\b"
            elif character == "\f":
                escaped = "\\f"
            elif character == "\n":
                escaped = "\\n"
            elif character == "\r":
                escaped = "\\r"
            elif character == "\t":
                escaped = "\\t"
            elif codepoint < 0x20:
                escaped = f"\\u{codepoint:04x}"
            elif codepoint > 0x7E:
                if codepoint <= 0xFFFF:
                    escaped = f"\\u{codepoint:04x}"
                else:
                    scalar = codepoint - 0x10000
                    high = 0xD800 | (scalar >> 10)
                    low = 0xDC00 | (scalar & 0x3FF)
                    escaped = f"\\u{high:04x}\\u{low:04x}"
            if escaped is None:
                continue
            if run_start < index:
                self._emit_ascii_range(value, run_start, index)
            self._emit_ascii(escaped)
            run_start = index + 1
        if run_start < len(value):
            self._emit_ascii_range(value, run_start, len(value))
        self._emit_ascii('"')

    def _emit_value(self, value: object) -> None:
        if value is None:
            self._emit_ascii("null")
        elif value is True:
            self._emit_ascii("true")
        elif value is False:
            self._emit_ascii("false")
        elif isinstance(value, str):
            self._emit_string(value)
        elif isinstance(value, int):
            self._emit_ascii(int.__repr__(value))
        elif isinstance(value, float):
            if math.isnan(value):
                self._emit_ascii("NaN")
            elif math.isinf(value):
                self._emit_ascii("Infinity" if value > 0 else "-Infinity")
            else:
                self._emit_ascii(float.__repr__(value))
        elif isinstance(value, list):
            sequence = cast(list[object], value)
            self._emit_ascii("[")
            for index, item in enumerate(sequence):
                if index:
                    self._emit_ascii(",")
                self._emit_value(item)
            self._emit_ascii("]")
        elif isinstance(value, dict):
            mapping = cast(dict[str, object], value)
            keys = sorted(mapping)
            self._emit_ascii("{")
            for index, key in enumerate(keys):
                if index:
                    self._emit_ascii(",")
                self._emit_string(key)
                self._emit_ascii(":")
                self._emit_value(mapping[key])
            self._emit_ascii("}")
        else:
            raise TypeError(f"cannot serialize canonical JSON value {type(value)!r}")


def _canonical_experiment_json_chunks(
    value: object, *, max_bytes: int
) -> tuple[bytes, ...]:
    return _BoundedCanonicalJson(max_bytes).encode(_json_data(value))


def _atomic_write_experiment_json(path: Path, value: object) -> None:
    with _EXPERIMENT_JSON_WRITE_LOCK:
        with secure_open_directory(
            path.parent, "experiment summary directory", create=True
        ) as parent:
            chunks = _canonical_experiment_json_chunks(
                value, max_bytes=_MAX_EXPERIMENT_JSON_BYTES
            )

            temporary_name = f".{path.stem}.{uuid4().hex}.tmp"
            with secure_create_exclusive_file(
                parent, temporary_name, "experiment summary temporary file"
            ) as temporary:
                offset = 0
                for chunk in chunks:
                    while offset < len(chunk):
                        written = os.write(temporary.descriptor, chunk[offset:])
                        if written <= 0:
                            raise OSError("failed to write experiment summary")
                        offset += written
                    offset = 0
                os.fsync(temporary.descriptor)
                secure_replace_exclusive_file(
                    parent,
                    temporary,
                    path.name,
                    "experiment summary publication",
                    replace_existing=True,
                )


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
    _write_ux_audit(output=output, results=result.results)
    _write_pagespeed(output=output, results=result.results)
    _run_auto_redesign(output=output, results=result.results)
    report_path = output / "report.html"
    if render_report:
        report_path = _render_completed_report(output=output)
    return summary_path, report_path


def _ux_audit_start_urls(results: Sequence[object]) -> tuple[str, ...]:
    """Collect unique application start URLs from closed run results."""

    urls: list[str] = []
    for result in results:
        try:
            url = result.state.spec.application_version.start_url  # type: ignore[attr-defined]
        except AttributeError:
            continue
        if isinstance(url, str) and url:
            urls.append(url)
    return tuple(dict.fromkeys(urls))


def _exploration_corpus_for_page_list(
    output: Path,
) -> Mapping[str, object] | None:
    """Crawl corpus from the newest finalized exploration attempt, if any.

    Best-effort by contract: any error reading the artifact yields ``None``
    so the shared page list falls back to start URLs alone.
    """

    try:
        store = ExplorationArtifactStore(output)
        index = store.get_index()
        records = index.get("attempts")
        finalized = ()
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            finalized = tuple(
                cast(Mapping[str, object], record)
                for record in cast(Sequence[object], records)
                if isinstance(record, Mapping)
                and str(cast(Mapping[str, object], record).get("status", ""))
                in {"succeeded", "partial", "completed"}
            )
        if not finalized:
            return None
        attempt_id = str(finalized[-1].get("attempt_id", ""))
        if not attempt_id:
            return None
        loaded = store.load_attempt(attempt_id)
        corpus_value = loaded.get("corpus")
        if isinstance(corpus_value, Mapping):
            return cast(Mapping[str, object], corpus_value)
    except Exception:  # noqa: BLE001 - page-list widening is best-effort
        return None
    return None


def _audit_page_urls(
    *,
    output: Path,
    results: Sequence[object],
    cap: int,
) -> tuple[str, ...]:
    """Shared deterministic page list for the audit (ADR 0007).

    Start URLs first, then BFS discovery order from a finalized exploration
    attempt's crawl corpus when one exists (page findings may then cover more
    URLs than start URLs — the renderer renders whatever the audit contains).
    The list stays exactly the start URLs when no finalized attempt exists.
    Resolution is best-effort: an unreadable exploration artifact leaves the
    start URLs unchanged instead of failing the audit.
    """

    from ux_analyzer.analysis.page_capture import resolve_redesign_page_list

    start_urls = _ux_audit_start_urls(results)
    if not start_urls:
        return ()
    corpus = _exploration_corpus_for_page_list(output)
    return resolve_redesign_page_list(corpus, start_urls, cap=cap)


def _ux_audit_sync(
    urls: Sequence[str], *, capture_hook: CaptureMaterialsHook | None = None
) -> dict[str, Any]:
    """Indirection so tests can stub the live audit without network."""

    from ux_analyzer.analysis.project_audit import audit_urls_sync

    return audit_urls_sync(urls, capture_hook=capture_hook)


def _write_ux_audit(output: Path, results: Sequence[object]) -> Path | None:
    """Persist the live-page UX audit beside experiment evidence.

    Best-effort: audit failures never fail an otherwise healthy run; they are
    recorded inside ``ux-audit.json`` as bounded error entries instead.
    """

    from ux_analyzer.analysis.page_capture import (
        PAGE_CAPTURE_FILENAME,
        PAGE_CAPTURE_SCHEMA,
        audit_capture_hook,
        max_pages_from_env,
    )
    from ux_analyzer.analysis.project_audit import AUDIT_FILENAME

    captures: dict[str, dict[str, object]] = {}
    capture_hook = audit_capture_hook(captures)
    urls = _audit_page_urls(output=output, results=results, cap=max_pages_from_env())
    if not urls:
        return None
    try:
        report = _ux_audit_sync(urls, capture_hook=capture_hook)
    except Exception as error:  # noqa: BLE001 - audit must not fail a run
        typer.echo(
            f"warning: live-page audit unavailable: {type(error).__name__}: {error}",
            err=True,
        )
        return None
    captured_pages = [
        entry["payload"]
        for entry in captures.values()
        if entry.get("status") == "captured"
    ]
    if captured_pages:
        document = {
            "schema": PAGE_CAPTURE_SCHEMA,
            "pages": captured_pages,
        }
        try:
            output.mkdir(parents=True, exist_ok=True)
            _atomic_json_write(output / PAGE_CAPTURE_FILENAME, document)
        except (OSError, TypeError, ValueError) as error:
            typer.echo(
                f"warning: could not persist {PAGE_CAPTURE_FILENAME}: {error}",
                err=True,
            )
    destination = output / AUDIT_FILENAME
    try:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(destination, report)
    except (OSError, TypeError, ValueError) as error:
        typer.echo(f"warning: could not persist {AUDIT_FILENAME}: {error}", err=True)
        return None
    total = report.get("total_issues", 0)
    audited_urls = len(report.get("urls", ()))
    typer.echo(
        f"live-page audit: {total} issue(s) across {audited_urls} URL(s); "
        f"see {AUDIT_FILENAME} and report.html"
    )
    return destination


def _write_pagespeed(output: Path, results: Sequence[object]) -> Path | None:
    """Persist PageSpeed Insights reports beside experiment evidence.

    Best-effort like the live-page audit: the API's own JSON is cached
    under ``<output>/pagespeed-cache`` and the derived, bounded report goes
    to ``pagespeed.json`` so ``report.html`` can render the complete
    pagespeed.web.dev-style result. Failures are recorded as error entries
    or a warning; they never fail an otherwise healthy run.

    The report shows exactly one analysis per URL and strategy — the API
    run — so its numbers and any link it renders cannot contradict each
    other. The headless pagespeed.web.dev saved-report capture is opt-in
    via ``UXA_PAGESPEED_WEB_LINKS=1`` because it triggers a second,
    independent Lighthouse analysis whose scores inevitably differ from
    the recorded API run.
    """
    if os.environ.get("UXA_SKIP_PAGESPEED", "") not in {"", "0", "false", "False"}:
        return None
    from ux_analyzer.analysis.pagespeed import (
        PAGESPEED_FILENAME,
        enrich_pagespeed_web_links,
        pagespeed_api_key,
        pagespeed_report_sync,
    )

    urls = _ux_audit_start_urls(results)
    if not urls:
        return None
    try:
        report = pagespeed_report_sync(
            urls,
            key=pagespeed_api_key(),
            cache_root=output,
        )
        report = enrich_pagespeed_web_links(
            report,
            cache_root=output,
            resolve=os.environ.get("UXA_PAGESPEED_WEB_LINKS", "")
            in {"1", "true", "True"},
        )
    except Exception as error:  # noqa: BLE001 - pagespeed must not fail a run
        typer.echo(
            f"warning: PageSpeed Insights unavailable: {type(error).__name__}: {error}",
            err=True,
        )
        return None
    destination = output / PAGESPEED_FILENAME
    try:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(destination, report)
    except (OSError, TypeError, ValueError) as error:
        typer.echo(
            f"warning: could not persist {PAGESPEED_FILENAME}: {error}", err=True
        )
        return None
    typer.echo(
        f"pagespeed: {report.get('ok_strategy_count', 0)} report(s) for "
        f"{report.get('url_count', 0)} URL(s); see {PAGESPEED_FILENAME} and report.html"
    )
    return destination


def _capture_page(url: str, *, max_page_height: int | None = None) -> dict[str, object]:
    """Indirection so tests can stub the standalone capture without a browser."""

    from ux_analyzer.analysis.page_capture import capture_page

    return capture_page(url, max_page_height=max_page_height)


async def _run_redesign_pass(
    captures: Mapping[str, Mapping[str, object]],
    *,
    audience: str,
    settings: OpenAICompatibleSettings,
):
    """Run one redesign pass through the two ADR 0007 model roles."""

    from dataclasses import asdict, replace

    from ux_analyzer.application.redesign import run_redesign_pass
    from ux_analyzer.providers.redesign import (
        RedesignCriticMerger,
        RedesignProposer,
    )
    from ux_analyzer.providers.redesign_principles import (
        REDESIGN_PRINCIPLE_PACK_VERSION,
        redesign_principle_ids,
        redesign_principle_pack,
    )

    http_client = httpx.AsyncClient(timeout=settings.timeout_seconds)
    try:
        client = create_structured_model_client(
            settings,
            http_client=http_client,
            call_limiter=asyncio.Semaphore(settings.max_concurrent_calls),
        )
        outcome = await run_redesign_pass(
            captures,
            audience=audience,
            proposer=RedesignProposer(
                client,
                model=settings.model_for_role(ModelRole.REDESIGN_PROPOSER),
            ),
            critic=RedesignCriticMerger(
                client,
                model=settings.model_for_role(ModelRole.REDESIGN_CRITIC_MERGER),
            ),
            attempt_id=new_redesign_attempt_id(),
            principle_pack=[asdict(item) for item in redesign_principle_pack()],
            principle_ids=redesign_principle_ids(),
            principle_pack_version=REDESIGN_PRINCIPLE_PACK_VERSION,
        )
        records = tuple(getattr(client, "records", ()) or ())
        if records:
            # Attach sanitized transport records so the publisher can persist
            # them next to the payload (failure debugging without payloads).
            outcome = replace(outcome, model_call_records=records)
        return outcome
    finally:
        await http_client.aclose()


def _redesign_settings_or_exit() -> OpenAICompatibleSettings:
    """Model settings for the redesign roles; missing names exit loudly."""

    try:
        return OpenAICompatibleSettings.from_env()
    except ModelConfigurationError as error:
        typer.echo(f"model environment error: {error}")
        raise typer.Exit(1) from error


def _redesign_captures(
    *,
    output: Path,
    pages: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    """Load captures from the persisted sidecar, re-capturing what is missing.

    Fresh sidecar pages never trigger a browser; only pages absent from the
    sidecar (or captured before effective tap-target geometry landed) are
    captured with a dedicated pass, and the sidecar is updated so the next
    run stays on the shared single-pass path (ADR 0007).
    """

    from ux_analyzer.analysis.page_capture import (
        PAGE_CAPTURE_SCHEMA,
        capture_reports_effective_tap_boxes,
        load_page_capture,
        max_page_height_from_env,
        sidecar_captures_by_url,
    )

    captures = sidecar_captures_by_url(load_page_capture(output) or {})
    # Pages captured before effective tap-target measurement (no tap_box on
    # interactive entries) cannot validate hit-target claims against the
    # real tappable surface, so they are stale for the redesign consumer.
    missing = tuple(
        url
        for url in pages
        if url not in captures or not capture_reports_effective_tap_boxes(captures[url])
    )
    if missing:
        # Same env semantics as the capture module: unset, unparseable, or
        # non-positive values fall back to the default cap (never crash).
        max_page_height = max_page_height_from_env()
        for url in missing:
            captures[url] = _capture_page(url, max_page_height=max_page_height)
        document = {
            "schema": PAGE_CAPTURE_SCHEMA,
            "pages": [captures[url] for url in pages if url in captures],
        }
        try:
            output.mkdir(parents=True, exist_ok=True)
            _atomic_json_write(output / "page-capture.json", document)
        except (OSError, TypeError, ValueError) as error:
            typer.echo(
                f"warning: could not persist page-capture.json: {error}",
                err=True,
            )
    return captures


def _persist_redesign_outcome(outcome: object, *, output: Path) -> None:
    """Publish one pass outcome; publication failures surface as warnings."""

    from ux_analyzer.storage.redesign_artifacts import (
        RedesignArtifactError,
        RedesignAttemptStore,
    )

    store = RedesignAttemptStore(output)
    try:
        store.publish(outcome.attempt, captures_digest=outcome.captures_digest)  # type: ignore[attr-defined]
    except RedesignArtifactError as error:
        typer.echo(f"warning: redesign attempt not persisted: {error}", err=True)
        return
    records = getattr(outcome, "model_call_records", ())
    attempt_id = str(getattr(outcome.attempt, "attempt_id", ""))  # type: ignore[attr-defined]
    if not records or not attempt_id:
        return
    # Sanitized transport audit records next to the payload: role, model,
    # attempts, latency, retries, and failure diagnostics — no request or
    # response payloads — so a terminal model failure stays debuggable.
    destination = output / "redesign" / attempt_id
    try:
        _atomic_json_write(
            destination / "model-calls.json",
            {
                "schema": "redesign-model-calls-v1",
                "model_calls": [
                    {
                        "role": record.role.value,
                        "model": record.model,
                        "endpoint_origin": record.endpoint_origin,
                        "schema_version": record.schema_version,
                        "attempts": record.attempts,
                        "latency_ms": record.latency_ms,
                        "token_usage": {
                            "prompt_tokens": record.token_usage.prompt_tokens,
                            "completion_tokens": record.token_usage.completion_tokens,
                            "total_tokens": record.token_usage.total_tokens,
                        },
                        "retries": [
                            {
                                "reason": retry.reason,
                                "attempt": retry.attempt,
                                "status_code": retry.status_code,
                                "delay_seconds": retry.delay_seconds,
                            }
                            for retry in record.retries
                        ],
                        "failure": _redesign_failure_record(record.response),
                    }
                    for record in records
                ],
            },
        )
    except (OSError, TypeError, ValueError) as error:
        typer.echo(
            f"warning: could not persist redesign model-calls.json: {error}",
            err=True,
        )


def _redesign_failure_record(response: object) -> dict[str, object] | None:
    """Safe failure metadata from a terminal transport record response."""

    if not isinstance(response, Mapping):
        return None
    mapping = cast(Mapping[str, object], response)
    failure = mapping.get("failure")
    if not isinstance(failure, str):
        return None
    record: dict[str, object] = {"reason": failure}
    provider = mapping.get("provider")
    if isinstance(provider, Mapping):
        provider_mapping = cast(Mapping[str, object], provider)
        for key in ("status_code", "error_code", "error_type", "request_id"):
            value = provider_mapping.get(key)
            if isinstance(value, (str, int)):
                record[key] = value
    diagnostics = mapping.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        record["diagnostics"] = cast(Mapping[str, object], diagnostics)
    return record


def _run_auto_redesign(
    *,
    output: Path,
    results: Sequence[object],
) -> None:
    """Best-effort redesign attempt after a completed experiment.

    Gated by ``UXA_REDESIGN_ENABLED`` (default off). Resolves the shared
    page list, reuses the audit's persisted ``page-capture.json`` sidecar
    pages, and runs one redesign pass; every failure path mirrors the
    audit's best-effort warnings and never fails the run.
    """

    if os.environ.get("UXA_REDESIGN_ENABLED", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    try:
        settings = _redesign_settings_or_exit()
    except typer.Exit:
        typer.echo(
            "warning: redesign enabled but model environment incomplete; skipping",
            err=True,
        )
        return
    from ux_analyzer.analysis.page_capture import max_pages_from_env

    pages = _audit_page_urls(output=output, results=results, cap=max_pages_from_env())
    if not pages:
        return
    try:
        captures = _redesign_captures(output=output, pages=pages)
        if not captures:
            typer.echo("warning: redesign skipped; no page captures", err=True)
            return
        outcome = asyncio.run(
            _run_redesign_pass(captures, audience="", settings=settings)
        )
        _persist_redesign_outcome(outcome, output=output)
        status = outcome.attempt.status.value  # type: ignore[attr-defined]
        typer.echo(f"redesign attempt: {status}; see redesign/ and report.html")
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 - redesign must not fail a run
        typer.echo(
            f"warning: redesign unavailable: {type(error).__name__}: {error}",
            err=True,
        )


@app.command("redesign")
def redesign_command(
    output: Path = typer.Argument(..., help="Experiment output directory"),
    pages: list[str] = typer.Option(
        [],
        "--pages",
        help="Explicit page URLs (repeatable); overrides the resolved page list.",
    ),
    extra_pages: list[str] | None = typer.Argument(
        None,
        help="Optional bare page URLs appended to --pages (``--pages URL ...``).",
    ),
    audience: str = typer.Option(
        "",
        "--audience",
        help="Optional operator context; the model still infers and states its own audience.",
    ),
    max_pages: int | None = typer.Option(
        None,
        "--max-pages",
        help="Page-list cap [default: UXA_REDESIGN_MAX_PAGES or 10].",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Resolve and print the page list without running models.",
    ),
) -> None:
    """Propose creative redesign improvements from persisted page captures.

    Reads the shared ``page-capture.json`` sidecar when fresh; only missing
    pages get a dedicated capture pass. Publishes an immutable attempt under
    ``<output>/redesign/<attempt-id>/``.
    """

    from ux_analyzer.analysis.page_capture import (
        fresh_capture_urls,
        max_pages_from_env,
        normalize_capture_url,
        resolve_redesign_page_list,
    )
    from ux_analyzer.domain.redesign import RedesignAttemptStatus
    from ux_analyzer.storage.redesign_artifacts import RedesignAttemptStore

    cap = max_pages if max_pages is not None else max_pages_from_env()
    if pages or extra_pages:
        # Explicit URLs take the same canonical form as the resolved list
        # (fragment stripped, host lowercased, default port elided) so they
        # hit the persisted sidecar instead of re-capturing a near-duplicate
        # key; the cap applies so --max-pages is honored here too.
        normalized: list[str] = []
        for raw_url in [*pages, *(extra_pages or ())]:
            try:
                normalized.append(normalize_capture_url(raw_url))
            except ValueError as error:
                _exit_with_error(f"invalid page URL {raw_url!r}: {error}")
        resolved_pages = tuple(dict.fromkeys(normalized))[:cap]
    else:
        exploration_corpus = _exploration_corpus_for_page_list(output)
        resolved_pages = resolve_redesign_page_list(
            exploration_corpus,
            fresh_capture_urls(output),
            cap=cap,
        )
        if not resolved_pages:
            resolved_pages = fresh_capture_urls(output)[:cap]
    if not resolved_pages:
        _exit_with_error(
            "no page captures available: provide --pages or run an experiment "
            "or exploration first"
        )
    if dry_run:
        typer.echo(f"redesign dry-run: {len(resolved_pages)} page(s)")
        for url in resolved_pages:
            typer.echo(f"- {url}")
        return
    settings = _redesign_settings_or_exit()
    try:
        captures = _redesign_captures(output=output, pages=resolved_pages)
    except Exception as error:  # noqa: BLE001 - capture failures are loud but bounded
        _exit_with_error(
            f"redesign failed: capture unavailable: {type(error).__name__}: {error}"
        )
    if not captures:
        _exit_with_error("redesign failed: no page captures available")
    try:
        outcome = asyncio.run(
            _run_redesign_pass(captures, audience=audience, settings=settings)
        )
    except Exception as error:  # noqa: BLE001 - surfaced as attempt status
        _exit_with_error(f"redesign failed: {type(error).__name__}: {error}")
    _persist_redesign_outcome(outcome, output=output)
    status = outcome.attempt.status
    typer.echo(f"redesign attempt: {status.value}")
    if status is RedesignAttemptStatus.REJECTED:
        for reason in outcome.attempt.rejection_reasons:
            typer.echo(f"- rejected: {reason}", err=True)
    if status is RedesignAttemptStatus.UNAVAILABLE:
        typer.echo(f"- unavailable: {outcome.attempt.unavailable_reason}", err=True)
    store = RedesignAttemptStore(output)
    selection = store.newest_valid_attempt()
    for note in selection.skipped:
        typer.echo(f"warning: {note}", err=True)


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


def _json_data(value: object, *, _active: set[int] | None = None) -> object:
    active: set[int] = set() if _active is None else _active
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_data(value.value, _active=active)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        identity = id(mapping)
        if identity in active:
            raise ValueError("circular reference in experiment value")
        active.add(identity)
        try:
            return {
                str(key): _json_data(item, _active=active)
                for key, item in mapping.items()
            }
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = cast(Sequence[object], value)
        identity = id(sequence)
        if identity in active:
            raise ValueError("circular reference in experiment value")
        active.add(identity)
        try:
            return [_json_data(item, _active=active) for item in sequence]
        finally:
            active.remove(identity)
    if is_dataclass(value):
        identity = id(value)
        if identity in active:
            raise ValueError("circular reference in experiment value")
        active.add(identity)
        try:
            return {
                item.name: _json_data(getattr(value, item.name), _active=active)
                for item in fields(value)
            }
        finally:
            active.remove(identity)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        identity = id(value)
        if identity in active:
            raise ValueError("circular reference in experiment value")
        active.add(identity)
        try:
            return _json_data(model_dump(mode="json"), _active=active)
        finally:
            active.remove(identity)
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
