"""Build self-contained HTML replay pages from immutable run bundles."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import mimetypes
import re
from collections import defaultdict
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from PIL import Image

from ux_analyzer.application.checkpoint import finalized_bundle_failures
from ux_analyzer.application.evaluation import (
    persisted_comparison_sample_is_valid,
)
from ux_analyzer.ports.artifacts import (
    SaliencyArtifactKind,
    canonicalize_saliency_artifact_content,
    parse_saliency_json_content,
    required_saliency_artifact_paths,
    validate_saliency_artifact_path,
    validate_timeline_event_order,
)
from ux_analyzer.storage.run_bundle import (
    secure_assert_ancestors,
    secure_is_link_or_reparse,
    secure_read_bytes,
    validate_saliency_heatmap_content,
    validate_saliency_native_map_content,
)

DEFAULT_SINGLE_FILE_THRESHOLD = 2_000_000
_MAX_REPORT_JSON_BYTES = 8 * 1024 * 1024
_MAX_REPORT_TIMELINE_BYTES = 16 * 1024 * 1024
_MAX_SALIENCY_ARTIFACT_BYTES = 8 * 1024 * 1024
_MAX_SOURCE_SCREENSHOT_BYTES = 16 * 1024 * 1024
_REQUIRED_BUNDLE_FILES = frozenset({"manifest.json", "timeline.jsonl", "result.json"})
_PRIVATE_KEYS = frozenset(
    {
        "api_key",
        "destination_url",
        "execution_reference",
        "handler_name",
        "hidden_label",
        "password",
        "provider_id",
        "selector",
        "test_id",
        "token",
    }
)
_METRIC_IDENTITY_KEYS = frozenset(
    {
        "abandoned",
        "application_version_id",
        "config_digest",
        "persona_id",
        "policy",
        "run_id",
        "scenario_id",
        "seed",
        "model_trial",
        "prominence_provider_id",
    }
)
_SALIENCY_DURATIONS = ("1s", "3s", "7s")


def _json_object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains duplicate fields")
        result[key] = value
    return result


_SALIENCY_ARTIFACT_FILENAMES = frozenset(
    {
        "1s.npz",
        "3s.npz",
        "7s.npz",
        "1s-heatmap.png",
        "3s-heatmap.png",
        "7s-heatmap.png",
        "profiles.json",
        "metadata.json",
    }
)
_SAFE_PUBLIC_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SENSITIVE_IDENTIFIER_MARKERS = (
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_REDACTED_IDENTIFIER = "[REDACTED]"
_SALIENCY_PROFILE_KEYS = frozenset(
    {
        "viewport_id",
        "element_id",
        "immediate",
        "early",
        "eventual",
        "general",
        "aggregates",
        "aggregation_version",
        "prediction_provenance",
    }
)
_SALIENCY_ESTIMATE_KEYS = frozenset({"kind", "score", "source"})
_SALIENCY_AGGREGATE_KEYS = frozenset(
    {
        "viewport_id",
        "element_id",
        "duration",
        "density",
        "robust_peak",
        "raw_mass",
        "mass_share",
        "clipped_area",
        "visibility_fraction",
        "occlusion_fraction",
        "raw_score",
        "adjusted_score",
    }
)
_SALIENCY_PROVIDER_IDS = frozenset(
    {"heuristic", "foveacast", "foveacast-prominence", "heuristic-prominence"}
)
_CANONICAL_PROVIDER_IDS = {
    "heuristic": "heuristic",
    "heuristic-prominence": "heuristic",
    "foveacast": "foveacast",
    "foveacast-prominence": "foveacast",
}


class SaliencyReplayUnavailable(ValueError):
    """Raised when checksum-covered saliency evidence fails schema checks."""


def render_experiment_report(
    bundle_root: Path,
    output_path: Path,
    *,
    max_single_file_bytes: int | None = None,
    single_file_threshold: int | None = None,
) -> Path:
    """Render bundle root as one HTML file or an index plus run pages."""

    root = Path(bundle_root)
    destination = Path(output_path)
    if not root.exists() or not root.is_dir() or secure_is_link_or_reparse(root):
        raise FileNotFoundError(f"bundle root does not exist: {root}")
    if max_single_file_bytes is not None and single_file_threshold is not None:
        raise ValueError("provide one report size threshold")
    threshold = (
        max_single_file_bytes
        if max_single_file_bytes is not None
        else single_file_threshold
    )
    threshold = DEFAULT_SINGLE_FILE_THRESHOLD if threshold is None else threshold
    if threshold <= 0:
        raise ValueError("report size threshold must be greater than zero")

    experiment = _load_experiment(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _estimated_full_report_bytes(experiment) <= threshold:
        full_context = _report_context(experiment)
        single_html = _render_html(full_context, "Attention-guided experiment replay")
        if len(single_html.encode("utf-8")) <= threshold:
            destination.write_text(single_html, encoding="utf-8")
            return destination

    run_directory = destination.parent / f"{destination.stem}-runs"
    run_directory.mkdir(parents=True, exist_ok=True)
    run_page_names = _run_page_names(experiment["runs"])
    run_links = {
        run_id: f"{run_directory.name}/{page_name}"
        for run_id, page_name in run_page_names.items()
    }
    index_context = _report_context(
        experiment,
        include_run_payload=False,
        run_links=run_links,
    )
    destination.write_text(
        _render_html(index_context, "Attention-guided experiment replay"),
        encoding="utf-8",
    )
    for run in experiment["runs"]:
        run_context = _report_context(
            {**experiment, "runs": [run]},
            run_links={run["run_id"]: ""},
        )
        run_html = _render_html(
            run_context,
            f"Run replay: {run['run_id']}",
        )
        if len(run_html.encode("utf-8")) > threshold:
            run_html = _oversized_run_html(run["run_id"], threshold)
        (run_directory / run_page_names[run["run_id"]]).write_text(
            run_html, encoding="utf-8"
        )
    return destination


def _estimated_full_report_bytes(experiment: dict[str, Any]) -> int:
    template_root = Path(__file__).parent
    shell_bytes = sum(
        path.stat().st_size
        for path in (
            template_root / "templates" / "experiment.html.j2",
            template_root / "static" / "report.css",
            template_root / "static" / "report.js",
        )
    )
    return shell_bytes + len(_safe_json(experiment).encode("utf-8"))


def _oversized_run_html(run_id: str, threshold: int) -> str:
    escaped_run_id = html.escape(run_id)
    detailed = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>Run replay omitted: {escaped_run_id}</title>"
        f"<h1>Run {escaped_run_id}</h1>"
        "<p>Detailed replay omitted because run page exceeds configured size limit.</p>"
        "<p>Run remains listed in experiment index with trust and failure details.</p>"
        "</html>"
    )
    if len(detailed.encode("utf-8")) <= threshold:
        return detailed
    concise = "<!doctype html><title>Replay omitted</title><p>Run page exceeds size limit.</p>"
    if len(concise.encode("utf-8")) > threshold:
        raise ValueError("report size threshold is too small for overflow notice")
    return concise


def _load_experiment(root: Path) -> dict[str, Any]:
    summary = _read_object(root / "experiment.json", required=False)
    run_directories = _run_directories(root)
    runs = [_load_run(path) for path in run_directories]
    failures = _failure_rows(summary)
    by_run_id = {run["run_id"]: run for run in runs}
    for failure in failures:
        existing = by_run_id.get(failure["run_id"])
        if existing is None:
            created = _failed_run(failure)
            runs.append(created)
            by_run_id[created["run_id"]] = created
        else:
            _merge_failure(existing, failure)
    if not runs:
        raise ValueError(f"no run or failure evidence found under {root}")
    ordered_runs = tuple(sorted(runs, key=lambda item: item["run_id"]))
    gate_rows = _gate_rows(summary, ordered_runs)
    return {
        "runs": ordered_runs,
        "run_rows": _run_overview_rows(ordered_runs, gate_rows),
        "comparison_rows": _comparison_rows(ordered_runs),
        "gate_rows": gate_rows,
        "failure_rows": [run for run in ordered_runs if run["failed"]],
        "evidence_summary": _evidence_summary(ordered_runs),
        "focused_acceptance": _focused_acceptance(summary),
        "limitations": _unique(
            limitation for run in ordered_runs for limitation in run["limitations"]
        ),
    }


def _run_directories(root: Path) -> tuple[Path, ...]:
    if secure_is_link_or_reparse(root):
        return ()
    if any(
        (root / name).exists() for name in (*_REQUIRED_BUNDLE_FILES, "crash.marker")
    ):
        return (root,)
    candidates: list[Path] = []
    runs_root = root / "runs"
    if runs_root.is_dir() and not secure_is_link_or_reparse(runs_root):
        candidates.extend(
            path
            for path in runs_root.iterdir()
            if path.is_dir() and not secure_is_link_or_reparse(path)
        )
    staging_root = root / ".staging"
    if staging_root.is_dir() and not secure_is_link_or_reparse(staging_root):
        candidates.extend(
            path
            for path in staging_root.iterdir()
            if path.is_dir()
            and not secure_is_link_or_reparse(path)
            and ((path / "crash.marker").is_file() or (path / ".active").is_file())
        )
    return tuple(sorted(candidates, key=lambda path: path.name))


def _load_run(path: Path) -> dict[str, Any]:
    manifest, manifest_failures = _read_object_safely(path / "manifest.json")
    result, result_failures = _read_object_safely(path / "result.json")
    crash, crash_failures = _read_object_safely(path / "crash.marker", required=False)
    events, timeline_failures = _read_jsonl_safely(path / "timeline.jsonl")
    if not events:
        state = _mapping(result.get("state"))
        events = _list_of_mappings(state.get("events"))
    state = _mapping(result.get("state"))
    spec = _mapping(result.get("spec")) or _mapping(state.get("spec"))
    integrity_failures = [
        *manifest_failures,
        *result_failures,
        *crash_failures,
        *timeline_failures,
    ]
    active = (path / ".active").is_file()
    if active:
        integrity_failures.append("active bundle marker present; run is partial")
    if not manifest:
        integrity_failures.append("manifest.json contains no manifest data")
    if not result:
        integrity_failures.append("result.json contains no run result data")
    if not events:
        integrity_failures.append("timeline.jsonl contains no events")
    embedded_run_ids = {
        value
        for value in (manifest.get("run_id"), result.get("run_id"))
        if isinstance(value, str)
    }
    if embedded_run_ids and embedded_run_ids != {path.name}:
        integrity_failures.append("bundle directory does not match embedded run ID")
    finalized_candidate = (
        path.parent.name == "runs" or (path / "checksums.sha256").is_file()
    )
    if finalized_candidate:
        integrity_failures.extend(_bundle_integrity_failures(path))
    if result and not any(_kind(event) == "run-terminated" for event in events):
        integrity_failures.append("missing terminal run event")
    integrity_failures = _unique(integrity_failures)
    raw_metrics = _extract_metrics(result)
    prominence_provider_id, provider_failures = _provider_identity_failures(
        manifest, result, state, raw_metrics
    )
    probe_snapshots = _snapshots(path, events, include_source=False)
    saliency = _saliency_replay(
        path,
        events,
        probe_snapshots,
        expected_provider_id=prominence_provider_id,
    )
    _, saliency_provider_failures = _provider_identity_failures(
        manifest,
        result,
        state,
        raw_metrics,
        saliency=saliency,
    )
    provider_failures.extend(saliency_provider_failures)
    integrity_failures = _unique([*integrity_failures, *provider_failures])
    trusted = bool(result and not crash and not integrity_failures)
    metrics = raw_metrics if trusted else {}
    snapshots = _snapshots(path, events, include_source=trusted)
    if trusted:
        saliency = _saliency_replay(
            path,
            events,
            snapshots,
            expected_provider_id=prominence_provider_id,
        )
    attention = _mapping(state.get("attention"))
    noticed = set(_strings(attention.get("noticed_ids")))
    inspected = set(_strings(attention.get("inspected_ids")))
    observations = [
        _observation(event)
        for event in events
        if _kind(event) == "observation-recorded"
    ]
    for observation in observations:
        noticed.update(
            element["id"] for element in observation["newly_revealed_elements"]
        )
    for event in events:
        if _kind(event) == "action-executed":
            action = _mapping(event.get("action"))
            if action.get("kind") in {"inspect", "inspect-element"}:
                element_id = action.get("element_id")
                if isinstance(element_id, str):
                    inspected.add(element_id)
    for snapshot in snapshots:
        for element in snapshot["elements"]:
            element["noticed"] = element["id"] in noticed
            element["inspected"] = element["id"] in inspected

    supported_evidence, unsupported_limitations = (
        _evidence(result, metrics) if trusted else ([], [])
    )
    findings, finding_limitations = _findings(result) if trusted else ([], [])
    limitations = _unique(
        [
            "Outputs describe simulated benchmark evidence, not real-user completion or satisfaction.",
            *(
                [
                    "Post-action feedback could not be evaluated because no post-action snapshot was recorded."
                ]
                if metrics and metrics.get("feedback_observed") is None
                else []
            ),
            *(_strings(result.get("limitations"))),
            *unsupported_limitations,
            *finding_limitations,
            *integrity_failures,
            *(
                [
                    "Bundle is incomplete or crashed; evidence is untrusted and excluded from scorecards."
                ]
                if not trusted
                else []
            ),
        ]
    )
    verification = _verification(events, result)
    outcome = _outcome(events, result)
    scenario = _mapping(spec.get("scenario"))
    version = _mapping(spec.get("application_version"))
    persona = _mapping(spec.get("persona"))
    version_id = _first_string(
        metrics.get("application_version_id"), version.get("id"), "unknown"
    )
    scenario_id = _first_string(
        metrics.get("scenario_id"), scenario.get("id"), "unknown"
    )
    persona_id = _first_string(metrics.get("persona_id"), persona.get("id"), "unknown")
    policy = _first_string(metrics.get("policy"), spec.get("policy"), "unknown")
    run_id = _first_string(manifest.get("run_id"), result.get("run_id"), path.name)
    seed = int(_number(manifest.get("seed"), 0))
    model_trial = int(_number(manifest.get("model_trial"), 0))
    reproducibility = _text(metrics.get("reproducibility"), "seeded")
    reproducibility_label = (
        f"model-dependent (attention seed {seed}, model trial {model_trial})"
        if reproducibility == "model-dependent"
        else reproducibility
    )
    public_events = [_public_event(event) for event in events]
    terminal_reason = _first_optional_string(
        result.get("terminal_reason"),
        crash.get("reason"),
        _mapping(result.get("outcome")).get("reason"),
    )
    evaluation_failure_reason = _optional_text(result.get("evaluation_failure_reason"))
    explicit_validity = result.get("ux_sample_valid")
    ux_sample_valid = (
        explicit_validity
        if isinstance(explicit_validity, bool)
        else outcome in {"verified-success", "agent-abandoned", "budget-exhausted"}
        and not evaluation_failure_reason
    )
    ux_sample_invalid_reason = _optional_text(result.get("ux_sample_invalid_reason"))
    saliency_fallbacks = _saliency_fallbacks(events)
    learned_fallback = _is_learned_provider(prominence_provider_id) and bool(
        saliency_fallbacks
        or (ux_sample_invalid_reason or "").startswith("saliency-fallback:")
        or any(
            event.get("cache_state") == "fallback"
            for event in events
            if _kind(event) == "prominence-recorded"
        )
    )
    if learned_fallback:
        ux_sample_valid = False
        if ux_sample_invalid_reason is None:
            ux_sample_invalid_reason = "saliency-fallback: " + _text(
                saliency_fallbacks[0].get("reason"), "learned output unavailable"
            )
    learned_replay_invalid = _is_learned_provider(prominence_provider_id) and any(
        group.get("replay_available") is False for group in saliency
    )
    if learned_replay_invalid:
        ux_sample_valid = False
        if ux_sample_invalid_reason is None:
            ux_sample_invalid_reason = (
                "saliency-replay-unavailable: evidence failed validation"
            )
    failure_reason = "; ".join(
        item
        for item in (
            terminal_reason,
            evaluation_failure_reason,
            ux_sample_invalid_reason,
            *integrity_failures,
        )
        if item
    )
    terminal_state = (
        "crashed"
        if crash
        else "partial"
        if active
        else "untrusted"
        if integrity_failures
        else "finalized"
        if result or (path / "checksums.sha256").is_file()
        else "partial"
    )
    comparison_valid = bool(
        trusted
        and persisted_comparison_sample_is_valid(
            result,
            raw_metrics,
            manifest=manifest,
            expected_run_id=run_id,
            expected_prominence_provider_id=prominence_provider_id,
        )
        and not learned_fallback
        and not learned_replay_invalid
    )
    stage = _run_stage(result, crash, outcome, integrity_failures)
    actions = _actions(events)
    model_calls = _model_calls(events)
    user_effort = _user_effort(actions, observations, metrics)
    analysis_cost = _analysis_cost(model_calls)
    return {
        "run_id": run_id,
        "bundle_path": f"{path.parent.name}/{path.name}",
        "integrity_status": "trusted" if trusted else "failed",
        "seed": seed,
        "model_trial": model_trial,
        "prominence_provider_id": prominence_provider_id,
        "reproducibility": reproducibility,
        "reproducibility_label": reproducibility_label,
        "scenario_id": scenario_id,
        "scenario_label": _first_string(scenario.get("name"), scenario_id.title()),
        "goal": _first_string(scenario.get("goal"), "Goal unavailable"),
        "version_id": version_id,
        "version_label": _first_string(version.get("label"), version_id.title()),
        "persona_id": persona_id,
        "persona_label": _first_string(persona.get("name"), persona_id.title()),
        "policy": policy,
        "outcome": outcome,
        "verified": bool(verification.get("verified", False)),
        "claimed": bool(result.get("agent_claimed_success", False)),
        "terminal_state": terminal_state,
        "stage": stage,
        "terminal_reason": terminal_reason,
        "evaluation_failure_reason": evaluation_failure_reason,
        "ux_sample_valid": ux_sample_valid,
        "ux_sample_invalid_reason": ux_sample_invalid_reason,
        "comparison_valid": comparison_valid,
        "failure_reason": failure_reason,
        "status_class": _status_class(outcome, stage, terminal_state, trusted),
        "failed": not trusted or stage != "complete",
        "trusted": trusted,
        "integrity_failures": integrity_failures,
        "timeline": public_events,
        "snapshots": snapshots,
        "observations": observations,
        "selections": _selections(events),
        "prominence": _prominence(events),
        "saliency": saliency,
        "saliency_stage_timeline": _saliency_stage_timeline(events),
        "saliency_runtime": _saliency_runtime(saliency, events),
        "saliency_fallbacks": saliency_fallbacks,
        "scent_records": _scent_records(events),
        "decisions": _decisions(events),
        "actions": actions,
        "verification": verification,
        "memory": _memory(attention),
        "manifests": _manifests(manifest, result, state),
        "model_calls": model_calls,
        "user_effort": user_effort,
        "analysis_cost": analysis_cost,
        "evidence": supported_evidence,
        "findings": findings,
        "metrics": _metric_rows(metrics, verification, outcome) if trusted else [],
        "limitations": limitations,
    }


def _report_context(
    experiment: dict[str, Any],
    *,
    include_run_payload: bool = True,
    run_links: dict[str, str] | None = None,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for source in _list_of_mappings(experiment["runs"]):
        run = (
            source
            if include_run_payload
            else {
                key: source[key]
                for key in (
                    "run_id",
                    "bundle_path",
                    "integrity_status",
                    "seed",
                    "model_trial",
                    "prominence_provider_id",
                    "reproducibility_label",
                    "scenario_id",
                    "scenario_label",
                    "version_id",
                    "version_label",
                    "persona_id",
                    "persona_label",
                    "policy",
                    "outcome",
                    "verified",
                    "claimed",
                    "terminal_state",
                    "stage",
                    "terminal_reason",
                    "evaluation_failure_reason",
                    "failure_reason",
                    "comparison_valid",
                    "status_class",
                    "failed",
                    "trusted",
                    "metrics",
                    "user_effort",
                    "analysis_cost",
                )
            }
        )
        if run_links is not None:
            run = {**run, "run_page": run_links.get(source["run_id"], "")}
        runs.append(run)
    initial_width = 0
    for run in _list_of_mappings(experiment["runs"]):
        if run["snapshots"]:
            initial_width = run["snapshots"][0]["viewport"]["width"]
            break
    failure_rows = [
        {
            key: run[key]
            for key in (
                "run_id",
                "scenario_id",
                "model_trial",
                "prominence_provider_id",
                "version_id",
                "persona_id",
                "policy",
                "stage",
                "terminal_state",
                "failure_reason",
            )
        }
        for run in _list_of_mappings(experiment["failure_rows"])
    ]
    limitations = (
        experiment["limitations"]
        if include_run_payload
        else ["Run-specific limitations are available on self-contained run pages."]
    )
    return {
        "runs": runs,
        "run_rows": experiment["run_rows"],
        "comparison_rows": experiment["comparison_rows"],
        "gate_rows": experiment["gate_rows"],
        "failure_rows": failure_rows,
        "evidence_summary": experiment["evidence_summary"],
        "focused_acceptance": experiment["focused_acceptance"],
        "limitations": limitations,
        "initial_viewport_width": initial_width,
        "report_json": _safe_json(
            {
                "runs": runs,
                "run_rows": experiment["run_rows"],
                "comparison_rows": experiment["comparison_rows"],
                "gate_rows": experiment["gate_rows"],
                "failure_rows": failure_rows,
                "evidence_summary": experiment["evidence_summary"],
                "focused_acceptance": experiment["focused_acceptance"],
                "limitations": limitations,
            }
        ),
    }


def _render_html(context: dict[str, Any], title: str) -> str:
    template_root = Path(__file__).parent
    environment = Environment(
        loader=FileSystemLoader(str(template_root / "templates")),
        autoescape=select_autoescape(("html", "xml")),
        undefined=StrictUndefined,
    )
    template = environment.get_template("experiment.html.j2")
    css = (template_root / "static" / "report.css").read_text(encoding="utf-8")
    javascript = (template_root / "static" / "report.js").read_text(encoding="utf-8")
    return template.render(
        title=title,
        css=css,
        javascript=javascript,
        **context,
    )


def _read_object(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file() or secure_is_link_or_reparse(path):
        if required:
            raise FileNotFoundError(path)
        return {}
    content = secure_read_bytes(path, "report JSON")
    if len(content) > _MAX_REPORT_JSON_BYTES:
        raise ValueError(f"report JSON exceeds {_MAX_REPORT_JSON_BYTES} bytes")
    value = json.loads(
        content.decode("utf-8"), object_pairs_hook=_json_object_without_duplicates
    )
    return _mapping(value)


def _read_object_safely(
    path: Path, *, required: bool = True
) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file() or secure_is_link_or_reparse(path):
        failure = f"missing required bundle file: {path.name}" if required else ""
        return {}, [failure] if failure else []
    try:
        content = secure_read_bytes(path, "report JSON")
        if len(content) > _MAX_REPORT_JSON_BYTES:
            return {}, [f"report JSON exceeds size limit: {path.name}"]
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (OSError, RuntimeError, UnicodeError, ValueError, json.JSONDecodeError):
        return {}, [f"invalid JSON bundle file: {path.name}"]
    if not isinstance(value, dict):
        return {}, [f"bundle file must contain object: {path.name}"]
    return cast(dict[str, Any], value), []


def _read_jsonl_safely(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file() or secure_is_link_or_reparse(path):
        return [], [f"missing required bundle file: {path.name}"]
    try:
        content = secure_read_bytes(path, "report timeline")
        if len(content) > _MAX_REPORT_TIMELINE_BYTES:
            return [], ["report timeline exceeds size limit"]
        lines = content.decode("utf-8").splitlines()
    except (OSError, RuntimeError, UnicodeError):
        return [], [f"unreadable bundle file: {path.name}"]
    events: list[dict[str, Any]] = []
    failures: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_json_object_without_duplicates)
        except (ValueError, json.JSONDecodeError):
            failures.append(f"invalid timeline JSON at line {line_number}")
            continue
        if not isinstance(value, dict):
            failures.append(f"timeline entry is not object at line {line_number}")
            continue
        events.append(cast(dict[str, Any], value))
    failures.extend(validate_timeline_event_order(events))
    return events, _unique(failures)


def _bundle_integrity_failures(path: Path) -> list[str]:
    return finalized_bundle_failures(path)


def _snapshots(
    path: Path, events: list[dict[str, Any]], *, include_source: bool
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for event in events:
        if _kind(event) != "viewport-captured":
            continue
        snapshot = _mapping(event.get("snapshot"))
        if not snapshot:
            continue
        public = _public_snapshot(path, snapshot, event, include_source=include_source)
        if public["id"] and all(item["id"] != public["id"] for item in snapshots):
            snapshots.append(public)
    return snapshots


def _public_snapshot(
    run_path: Path,
    snapshot: dict[str, Any],
    event: dict[str, Any],
    *,
    include_source: bool,
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = []
    for raw in _list_of_mappings(snapshot.get("elements")):
        bounds = _bounds(raw.get("bounds"))
        if not raw.get("id") or bounds is None:
            continue
        elements.append(
            {
                "id": _text(raw.get("id")),
                "role": _text(raw.get("role"), "other"),
                "label": _text(raw.get("label"), "Unlabelled element"),
                "bounds": bounds,
                "visibility_fraction": _bounded(raw.get("visibility_fraction")),
                "occlusion_fraction": _optional_bounded(raw.get("occlusion_fraction")),
                "local_contrast": _optional_bounded(raw.get("local_contrast")),
                "actionable": bool(raw.get("actionable", False)),
                "disabled": bool(raw.get("disabled", False)),
                "region_id": _optional_text(raw.get("region_id")),
                "noticed": False,
                "inspected": False,
            }
        )
    regions = [
        {"id": _text(region.get("id")), "label": _text(region.get("label"))}
        for region in _list_of_mappings(snapshot.get("regions"))
        if region.get("id")
    ]
    artifact = _optional_text(snapshot.get("screenshot_artifact"))
    screenshot, screenshot_redacted = (
        _source_artifact_data_uri(run_path, artifact)
        if include_source
        else (None, False)
    )
    viewport = _viewport(snapshot, event, elements)
    return {
        "id": _text(snapshot.get("id")),
        "viewport": viewport,
        "screenshot": screenshot,
        "screenshot_redacted": screenshot_redacted,
        "artifact": artifact,
        "elements": elements,
        "regions": regions,
    }


def _viewport(
    snapshot: dict[str, Any],
    event: dict[str, Any],
    elements: list[dict[str, Any]],
) -> dict[str, int]:
    candidates = (
        _mapping(snapshot.get("viewport")),
        _mapping(snapshot.get("viewport_size")),
        _mapping(event.get("viewport")),
        {
            "width": event.get("viewport_width"),
            "height": event.get("viewport_height"),
        },
    )
    for candidate in candidates:
        width = _number(candidate.get("width"), 0)
        height = _number(candidate.get("height"), 0)
        if width > 0 and height > 0:
            return {"width": int(width), "height": int(height)}
    max_x = max(
        (element["bounds"]["x"] + element["bounds"]["width"] for element in elements),
        default=1,
    )
    max_y = max(
        (element["bounds"]["y"] + element["bounds"]["height"] for element in elements),
        default=1,
    )
    return {"width": max(1, math.ceil(max_x)), "height": max(1, math.ceil(max_y))}


def _source_artifact_data_uri(
    run_path: Path, artifact: str | None
) -> tuple[str | None, bool]:
    if not artifact:
        return None, False
    relative = PurePosixPath(artifact.replace("\\", "/"))
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        return None, False
    if not _is_allowed_screenshot_artifact(relative):
        return None, False
    candidate = _secure_bundle_file(run_path, relative)
    if candidate is None:
        return None, False
    try:
        content = secure_read_bytes(candidate, "source screenshot")
    except (OSError, RuntimeError):
        return None, False
    if len(content) > _MAX_SOURCE_SCREENSHOT_BYTES:
        return None, False
    if _looks_redacted_source_png(content):
        return None, True
    if content.startswith(b"\x89PNG"):
        try:
            with Image.open(BytesIO(content)) as image:
                image.verify()
        except (OSError, ValueError):
            return None, False
    elif not content.startswith((b"\xff\xd8", b"GIF8")):
        return None, False
    mime = mimetypes.guess_type(candidate.name)[0] or _image_mime(content)
    return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}", False


def _is_allowed_screenshot_artifact(path: PurePosixPath) -> bool:
    """Accept only run-owned screenshot locations, never generated evidence."""

    if len(path.parts) == 2 and path.parts[0] != "artifacts":
        return False
    if len(path.parts) > 2:
        return False
    return path.suffix.casefold() in {".gif", ".jpeg", ".jpg", ".png"}


def _looks_redacted_source_png(content: bytes) -> bool:
    if not content.startswith(b"\x89PNG"):
        return False
    try:
        with Image.open(BytesIO(content)) as image:
            if image.width <= 0 or image.height <= 0:
                return False
            rgba = image.convert("RGBA")
            colors = rgba.getcolors(maxcolors=2)
            return colors == [(rgba.width * rgba.height, (0, 0, 0, 255))]
    except (OSError, ValueError):
        return False


def _image_mime(content: bytes) -> str:
    if content.startswith(b"\x89PNG"):
        return "image/png"
    if content.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if content.startswith(b"GIF8"):
        return "image/gif"
    return "application/octet-stream"


def _saliency_artifact_data_uri(run_path: Path, artifact: str) -> str | None:
    """Read one allowlisted heatmap without escaping bundle containment."""

    try:
        normalized = validate_saliency_artifact_path(artifact)
    except ValueError:
        return None
    if normalized.name not in {
        "1s-heatmap.png",
        "3s-heatmap.png",
        "7s-heatmap.png",
    }:
        return None
    candidate = _secure_bundle_file(run_path, normalized)
    if candidate is None:
        return None
    try:
        content = secure_read_bytes(candidate, "saliency heatmap")
    except (OSError, RuntimeError):
        return None
    if len(content) > _MAX_SALIENCY_ARTIFACT_BYTES:
        return None
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            if image.format != "PNG" or image.mode != "L":
                return None
    except (OSError, ValueError):
        return None
    return f"data:image/png;base64,{base64.b64encode(content).decode('ascii')}"


def _secure_bundle_file(root: Path, relative: PurePosixPath) -> Path | None:
    """Resolve bundle file only through real, contained directories."""

    try:
        secure_assert_ancestors(root, "report bundle")
        current = root
        if secure_is_link_or_reparse(current) or not current.is_dir():
            return None
        for index, part in enumerate(relative.parts):
            current = current / part
            if secure_is_link_or_reparse(current):
                return None
            if (
                index < len(relative.parts) - 1
                and current.exists()
                and not current.is_dir()
            ):
                return None
        if not current.is_file():
            return None
        return current
    except (OSError, RuntimeError, ValueError):
        return None


def _saliency_replay(
    run_path: Path,
    events: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    *,
    expected_provider_id: str = "unavailable",
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for event in events:
        kind = _kind(event)
        if not kind.startswith("saliency-") and kind != "prominence-recorded":
            continue
        if kind == "prominence-recorded" and not any(
            event.get(field_name) is not None
            for field_name in (
                "artifact_namespace",
                "active_provider_id",
                "source_event_id",
            )
        ):
            continue
        raw_namespace = _optional_text(
            event.get("artifact_namespace")
        ) or _optional_text(event.get("viewport_id"))
        namespace = _validated_saliency_namespace(raw_namespace)
        if not namespace:
            continue
        group = groups.setdefault(
            namespace,
            {
                "viewport_id": _safe_identifier(
                    event.get("source_viewport_id"), namespace
                ),
                "artifact_namespace": _safe_identifier(namespace, "unavailable"),
                "provider_id": "unavailable",
                "active_provider_id": "unavailable",
                "search_stage": "unavailable",
                "cache_state": "unavailable",
                "selected_mixture": [],
                "warnings": [],
                "model_checksums": [],
                "execution_provider": "unavailable",
                "timings_ms": [],
                "stage_history": [],
                "profile_event_ids": [],
                "operational_event_ids": [],
                "source_event_ids": [],
                "artifact_ids": [],
            },
        )
        _merge_saliency_event(group, event)

    replay: list[dict[str, Any]] = []
    for namespace in sorted(groups):
        group = groups[namespace]
        try:
            profiles = _read_saliency_profiles(
                run_path / "saliency" / namespace / "profiles.json",
                namespace,
            )
            metadata = _read_saliency_metadata(
                run_path / "saliency" / namespace / "metadata.json",
                namespace,
            )
            source_viewport_id = _text(group.get("viewport_id"), namespace)
            source_snapshot = next(
                (item for item in snapshots if item["id"] == source_viewport_id),
                None,
            )
            _validate_saliency_replay_linkage(
                run_path,
                events,
                group,
                profiles,
                metadata,
                namespace,
                source_snapshot=source_snapshot,
                expected_provider_id=expected_provider_id,
            )
            _validate_saliency_profile_lineage(profiles, source_snapshot)
        except SaliencyReplayUnavailable as error:
            replay.append(
                {
                    **group,
                    "profiles": [],
                    "metadata": {},
                    "entries": [],
                    "replay_available": False,
                    "replay_error": f"Saliency replay unavailable: {error}",
                    "overlay_available": False,
                    "overlay_message": None,
                }
            )
            continue
        entries: list[dict[str, Any]] = []
        for duration in _SALIENCY_DURATIONS:
            artifact = f"saliency/{namespace}/{duration}-heatmap.png"
            prediction: dict[str, Any] = next(
                (
                    item
                    for item in _list_of_mappings(metadata.get("predictions"))
                    if item.get("duration") == duration
                ),
                cast(dict[str, Any], {}),
            )
            prediction_metadata = _mapping(prediction.get("metadata"))
            aggregates = _aggregates_for_duration(profiles, duration)
            entries.append(
                {
                    "duration": duration,
                    "heatmap": _saliency_artifact_data_uri(run_path, artifact),
                    "heatmap_path": artifact,
                    "overlay_available": bool(
                        source_snapshot and source_snapshot.get("screenshot")
                    ),
                    "profiles": profiles,
                    "ranked_elements": _ranked_elements(
                        aggregates, source_snapshot, duration
                    ),
                    "aggregation_components": aggregates,
                    "inference_duration_ms": _number(
                        prediction_metadata.get("inference_duration_ms"),
                        0,
                    ),
                    "provider_id": _safe_identifier(
                        prediction_metadata.get("provider_id"),
                        _text(group.get("provider_id"), "unavailable"),
                    ),
                    "model_id": _safe_identifier(
                        prediction_metadata.get("model_id"), "unavailable"
                    ),
                    "model_version": _safe_identifier(
                        prediction_metadata.get("model_version"), "unavailable"
                    ),
                    "model_checksum": _safe_checksum(
                        prediction_metadata.get("model_checksum")
                    ),
                    "execution_provider": _safe_identifier(
                        prediction_metadata.get("execution_provider"),
                        _text(group.get("execution_provider"), "unavailable"),
                    ),
                    "warnings": _unique(
                        [
                            *_strings(group.get("warnings")),
                            *_strings(prediction_metadata.get("warnings")),
                        ]
                    ),
                }
            )
        replay.append(
            {
                **group,
                "profiles": profiles,
                "metadata": metadata,
                "entries": entries,
                "replay_available": True,
                "replay_error": None,
                "overlay_available": bool(
                    source_snapshot and source_snapshot.get("screenshot")
                ),
                "overlay_message": (
                    "Overlay unavailable due redaction. Heatmap-only artifact retained."
                    if source_snapshot and source_snapshot.get("screenshot_redacted")
                    else None
                ),
            }
        )
    return replay


def _validate_saliency_profile_lineage(
    profiles: list[dict[str, Any]], snapshot: dict[str, Any] | None
) -> None:
    """Keep model-estimate profiles joined to recorded interface elements."""

    if snapshot is None:
        raise SaliencyReplayUnavailable("saliency source viewport is unavailable")
    element_ids = {element["id"] for element in snapshot.get("elements", [])}
    if any(profile["element_id"] not in element_ids for profile in profiles):
        raise SaliencyReplayUnavailable(
            "saliency profile element is not present in source snapshot"
        )


def _merge_saliency_event(group: dict[str, Any], event: dict[str, Any]) -> None:
    for field_name in (
        "provider_id",
        "active_provider_id",
        "search_stage",
        "cache_state",
        "execution_provider",
    ):
        event_value = event.get(field_name)
        if field_name == "search_stage" and event_value is None:
            event_value = event.get("stage")
        if event_value is not None:
            group[field_name] = _safe_identifier(event_value, "unavailable")
    if event.get("selected_mixture") is not None:
        group["selected_mixture"] = _safe_mixture(event["selected_mixture"])
    if event.get("model_checksums") is not None:
        group["model_checksums"] = [
            checksum
            for checksum in _strings(event.get("model_checksums"))
            if _safe_checksum(checksum) != "unavailable"
        ]
    group["warnings"] = _unique(
        [*_strings(group.get("warnings")), *_strings(event.get("warnings"))]
    )
    timings = event.get("timings_ms")
    timing_values: list[object] | tuple[object, ...] = ()
    if isinstance(timings, list | tuple):
        timing_values = cast(list[object] | tuple[object, ...], timings)
        group["timings_ms"] = [
            _number(value, 0) for value in timing_values if _is_number(value)
        ]
    event_id = f"event-{int(_number(event.get('sequence'), 0))}"
    stage_history = _list_of_mappings(group.get("stage_history"))
    stage_history.append(
        {
            "event_id": event_id,
            "kind": _kind(event),
            "search_stage": _safe_identifier(
                event.get("search_stage", event.get("stage")), "unavailable"
            ),
            "selected_mixture": _safe_mixture(event.get("selected_mixture")),
            "cache_state": _safe_identifier(event.get("cache_state"), "unavailable"),
            "timings_ms": [
                _number(value, 0) for value in timing_values if _is_number(value)
            ],
        }
    )
    group["stage_history"] = stage_history
    if _kind(event) == "saliency-profiles-recorded":
        group.setdefault("profile_event_ids", []).append(event_id)
    if _kind(event) == "prominence-recorded":
        group.setdefault("operational_event_ids", []).append(event_id)
    source_event_id = event.get("source_event_id")
    if source_event_id is not None:
        group.setdefault("source_event_ids", []).append(str(source_event_id))
    artifact_ids = event.get("artifact_ids")
    if isinstance(artifact_ids, list | tuple):
        group.setdefault("artifact_ids", []).extend(
            _strings(cast(object, artifact_ids))
        )


def _read_saliency_profiles(path: Path, namespace: str) -> list[dict[str, Any]]:
    value = _read_saliency_json(path, SaliencyArtifactKind.PROFILES, namespace)
    if not isinstance(value, list):
        raise SaliencyReplayUnavailable("profiles JSON must be a list")
    profile_values = cast(list[object], value)
    profiles: list[dict[str, Any]] = []
    for raw in profile_values:
        if len(profiles) >= 10_000:
            raise SaliencyReplayUnavailable("saliency profile count exceeds limit")
        profile = _mapping(raw)
        if not profile or frozenset(profile) != _SALIENCY_PROFILE_KEYS:
            raise SaliencyReplayUnavailable("profile schema is invalid")
        if _text(profile.get("viewport_id"), "") != namespace:
            raise SaliencyReplayUnavailable("profile viewport is invalid")
        element_id = _safe_identifier(profile.get("element_id"), "unavailable")
        if element_id == "unavailable":
            raise SaliencyReplayUnavailable("profile element ID is invalid")
        item: dict[str, Any] = {
            "viewport_id": namespace,
            "element_id": element_id,
            "aggregation_version": _safe_identifier(
                profile.get("aggregation_version"), "unavailable"
            ),
            "aggregates": [],
            "prediction_provenance": [],
        }
        for estimate_name in ("immediate", "early", "eventual", "general"):
            estimate = _mapping(profile.get(estimate_name))
            if not estimate:
                if estimate_name in {"immediate", "early", "eventual"}:
                    raise SaliencyReplayUnavailable(
                        "profile duration coverage is incomplete"
                    )
                item[estimate_name] = None
                continue
            kind = _text(estimate.get("kind"), "unavailable")
            if kind not in {"predicted", "derived", "fallback", "unavailable"}:
                raise SaliencyReplayUnavailable("profile estimate kind is invalid")
            if estimate_name in {"immediate", "early", "eventual"} and (
                kind == "unavailable"
                or _optional_bounded(estimate.get("score")) is None
            ):
                raise SaliencyReplayUnavailable(
                    "profile duration estimate is unavailable"
                )
            item[estimate_name] = {
                "kind": kind,
                "score": _optional_bounded(estimate.get("score")),
                "source": _safe_identifier(estimate.get("source"), "unavailable"),
            }
        for raw_aggregate in _list_of_mappings(profile.get("aggregates")):
            if frozenset(raw_aggregate) != _SALIENCY_AGGREGATE_KEYS:
                raise SaliencyReplayUnavailable("profile aggregate schema is invalid")
            if len(item["aggregates"]) >= 10_000:
                raise SaliencyReplayUnavailable(
                    "saliency aggregate count exceeds limit"
                )
            if _text(raw_aggregate.get("element_id"), "") != element_id:
                raise SaliencyReplayUnavailable("profile aggregate element is invalid")
            duration = _text(raw_aggregate.get("duration"), "")
            if duration not in _SALIENCY_DURATIONS:
                raise SaliencyReplayUnavailable("profile aggregate duration is invalid")
            item["aggregates"].append(
                {
                    "viewport_id": namespace,
                    "element_id": element_id,
                    "duration": duration,
                    **{
                        field_name: _number(raw_aggregate.get(field_name), 0)
                        for field_name in _SALIENCY_AGGREGATE_KEYS
                        - {"viewport_id", "element_id", "duration"}
                    },
                }
            )
        if {aggregate["duration"] for aggregate in item["aggregates"]} != set(
            _SALIENCY_DURATIONS
        ):
            raise SaliencyReplayUnavailable(
                "profile aggregate duration coverage is incomplete"
            )
        provenance = _list_of_mappings(profile.get("prediction_provenance"))
        provenance_durations = {
            _text(record.get("duration"), "") for record in provenance
        }
        if provenance_durations != set(_SALIENCY_DURATIONS):
            raise SaliencyReplayUnavailable(
                "profile prediction provenance coverage is incomplete"
            )
        item["prediction_provenance"] = [
            {
                "duration": _text(record.get("duration"), "unavailable"),
                "metadata": _safe_prediction_metadata(_mapping(record.get("metadata"))),
            }
            for record in provenance
        ]
        profiles.append(item)
    return profiles


def _read_saliency_metadata(path: Path, namespace: str) -> dict[str, Any]:
    value = _read_saliency_json(path, SaliencyArtifactKind.METADATA, namespace)
    if not isinstance(value, dict):
        raise SaliencyReplayUnavailable("metadata JSON must be an object")
    metadata_value = cast(dict[str, Any], value)
    predictions: list[dict[str, Any]] = []
    for raw in _list_of_mappings(metadata_value.get("predictions")):
        duration = _text(raw.get("duration"), "")
        if duration not in _SALIENCY_DURATIONS:
            raise SaliencyReplayUnavailable("metadata duration is invalid")
        raw_metadata = _mapping(raw.get("metadata"))
        metadata: dict[str, Any] = {
            "provider_id": _safe_identifier(
                raw_metadata.get("provider_id"), "unavailable"
            ),
            "model_id": _safe_identifier(raw_metadata.get("model_id"), "unavailable"),
            "provider_version": _safe_identifier(
                raw_metadata.get("provider_version"), "unavailable"
            ),
            "model_version": _safe_identifier(
                raw_metadata.get("model_version"), "unavailable"
            ),
            "model_checksum": _safe_checksum(raw_metadata.get("model_checksum")),
            "input_dimensions": raw_metadata.get("input_dimensions"),
            "output_dimensions": raw_metadata.get("output_dimensions"),
            "geometry": _mapping(raw_metadata.get("geometry")),
            "preprocessing_version": _safe_identifier(
                raw_metadata.get("preprocessing_version"), "unavailable"
            ),
            "execution_provider": _safe_identifier(
                raw_metadata.get("execution_provider"), "unavailable"
            ),
            "inference_duration_ms": _number(
                raw_metadata.get("inference_duration_ms"), 0
            ),
            "warnings": _strings(raw_metadata.get("warnings")),
        }
        predictions.append({"duration": duration, "metadata": metadata})
    return {
        "viewport_id": _safe_identifier(metadata_value.get("viewport_id"), namespace),
        "cache_key": _mapping(metadata_value.get("cache_key")),
        "cache_key_digest": _safe_checksum(metadata_value.get("cache_key_digest")),
        "artifact_paths": [
            path
            for path in _strings(metadata_value.get("artifact_paths"))
            if _validated_saliency_artifact(path) is not None
        ],
        "predictions": predictions,
        "warnings": _strings(metadata_value.get("warnings")),
    }


def _source_screenshot_digest(
    run_path: Path, snapshot: dict[str, Any] | None
) -> str | None:
    if snapshot is None:
        return None
    artifact = _optional_text(
        snapshot.get("artifact") or snapshot.get("screenshot_artifact")
    )
    if not artifact:
        return None
    relative = PurePosixPath(artifact.replace("\\", "/"))
    if not _is_allowed_screenshot_artifact(relative):
        return None
    candidate = _secure_bundle_file(run_path, relative)
    if candidate is None:
        return None
    try:
        content = secure_read_bytes(candidate, "saliency source screenshot")
    except (OSError, RuntimeError):
        return None
    if len(content) > _MAX_SOURCE_SCREENSHOT_BYTES or _looks_redacted_source_png(
        content
    ):
        return None
    if content.startswith(b"\x89PNG"):
        try:
            with Image.open(BytesIO(content)) as image:
                image.verify()
        except (OSError, ValueError):
            return None
    elif not content.startswith((b"\xff\xd8", b"GIF8")):
        return None
    return hashlib.sha256(content).hexdigest()


def _read_saliency_json(
    path: Path, kind: SaliencyArtifactKind, namespace: str
) -> object:
    root = path.parents[2]
    candidate = _secure_bundle_file(
        root, PurePosixPath("saliency", namespace, path.name)
    )
    if candidate is None:
        raise SaliencyReplayUnavailable("saliency artifact is unavailable")
    try:
        raw = secure_read_bytes(candidate, "saliency JSON")
        if len(raw) > _MAX_SALIENCY_ARTIFACT_BYTES:
            raise SaliencyReplayUnavailable("saliency JSON exceeds size limit")
        parsed = parse_saliency_json_content(
            kind,
            raw,
            expected_viewport_id=namespace,
        )
        canonical = canonicalize_saliency_artifact_content(
            kind,
            raw,
            expected_viewport_id=namespace,
        )
        if canonical != raw:
            raise SaliencyReplayUnavailable("saliency artifact is not canonical")
        return parsed
    except SaliencyReplayUnavailable:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise SaliencyReplayUnavailable(
            "saliency artifact schema is invalid"
        ) from error


def _validated_saliency_namespace(value: str | None) -> str | None:
    if not value:
        return None
    try:
        normalized = validate_saliency_artifact_path(f"saliency/{value}/metadata.json")
    except ValueError:
        return None
    return normalized.parts[1]


def _validated_saliency_artifact(value: str) -> str | None:
    try:
        return validate_saliency_artifact_path(value).as_posix()
    except ValueError:
        return None


def _safe_prediction_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider_id": _safe_identifier(metadata.get("provider_id"), "unavailable"),
        "model_id": _safe_identifier(metadata.get("model_id"), "unavailable"),
        "model_version": _safe_identifier(metadata.get("model_version"), "unavailable"),
        "provider_version": _safe_identifier(
            metadata.get("provider_version"), "unavailable"
        ),
        "model_checksum": _safe_checksum(metadata.get("model_checksum")),
        "execution_provider": _safe_identifier(
            metadata.get("execution_provider"), "unavailable"
        ),
        "inference_duration_ms": _number(metadata.get("inference_duration_ms"), 0),
    }


def _validate_saliency_replay_linkage(
    run_path: Path,
    events: list[dict[str, Any]],
    group: dict[str, Any],
    profiles: list[dict[str, Any]],
    metadata: dict[str, Any],
    namespace: str,
    *,
    source_snapshot: dict[str, Any] | None,
    expected_provider_id: str = "unavailable",
) -> None:
    expected_paths = set(required_saliency_artifact_paths(namespace))
    metadata_paths = set(_strings(metadata.get("artifact_paths")))
    if metadata_paths != expected_paths:
        raise SaliencyReplayUnavailable("metadata artifact paths are incomplete")
    cache_key = _mapping(metadata.get("cache_key"))
    expected_screenshot_digest = _safe_checksum(cache_key.get("screenshot_sha256"))
    if expected_screenshot_digest == "unavailable":
        raise SaliencyReplayUnavailable("saliency screenshot digest is missing")
    actual_screenshot_digest = _source_screenshot_digest(run_path, source_snapshot)
    if (
        actual_screenshot_digest is not None
        and actual_screenshot_digest != expected_screenshot_digest
    ):
        raise SaliencyReplayUnavailable("saliency screenshot digest does not match")
    if actual_screenshot_digest is None and "overlay-redacted" not in _strings(
        group.get("warnings")
    ):
        raise SaliencyReplayUnavailable("saliency screenshot digest is unavailable")
    try:
        event_paths = {
            validate_saliency_artifact_path(path).as_posix()
            for path in _strings(group.get("artifact_ids"))
        }
    except ValueError as error:
        raise SaliencyReplayUnavailable("event artifact path is invalid") from error
    if event_paths != expected_paths:
        raise SaliencyReplayUnavailable("event artifact linkage is incomplete")
    canonical_expected_provider = _canonical_provider_id(expected_provider_id)
    if canonical_expected_provider is None:
        raise SaliencyReplayUnavailable("saliency provider identity is invalid")
    for field_name in ("provider_id", "active_provider_id"):
        if _canonical_provider_id(_text(group.get(field_name), "")) != (
            canonical_expected_provider
        ):
            raise SaliencyReplayUnavailable("saliency provider identity is invalid")
    timeline_checksums = set(_strings(group.get("model_checksums")))
    for prediction in _list_of_mappings(metadata.get("predictions")):
        prediction_metadata = _mapping(prediction.get("metadata"))
        if _canonical_provider_id(prediction_metadata.get("provider_id")) != (
            canonical_expected_provider
        ):
            raise SaliencyReplayUnavailable("saliency provider identity is invalid")
        if (
            timeline_checksums
            and _safe_checksum(prediction_metadata.get("model_checksum"))
            not in timeline_checksums
        ):
            raise SaliencyReplayUnavailable("saliency model checksum is invalid")
    metadata_by_duration = {
        _text(prediction.get("duration"), ""): _mapping(prediction.get("metadata"))
        for prediction in _list_of_mappings(metadata.get("predictions"))
    }
    for profile in profiles:
        for provenance in _list_of_mappings(profile.get("prediction_provenance")):
            duration = _text(provenance.get("duration"), "")
            profile_metadata = _mapping(provenance.get("metadata"))
            metadata_for_duration = metadata_by_duration.get(duration)
            if metadata_for_duration is None:
                raise SaliencyReplayUnavailable(
                    "saliency profile prediction duration is invalid"
                )
            for field_name in (
                "provider_id",
                "model_id",
                "model_version",
                "model_checksum",
                "execution_provider",
            ):
                if profile_metadata.get(field_name) != metadata_for_duration.get(
                    field_name
                ):
                    raise SaliencyReplayUnavailable(
                        "saliency profile prediction provenance does not match metadata"
                    )
    event_by_id = {
        f"event-{int(_number(event.get('sequence'), 0))}": event for event in events
    }
    profile_event_ids = {
        event_id for event_id in _strings(group.get("profile_event_ids"))
    }
    relevant_events = [
        event
        for event in events
        if _validated_saliency_namespace(
            _optional_text(event.get("artifact_namespace"))
            or _optional_text(event.get("viewport_id"))
        )
        == namespace
    ]
    for event in relevant_events:
        source_event_id = _optional_text(event.get("source_event_id"))
        if source_event_id is None:
            continue
        source = event_by_id.get(source_event_id)
        current_sequence = int(_number(event.get("sequence"), 0))
        source_sequence = int(_number(source.get("sequence"), 0)) if source else 0
        if source is None or source_sequence >= current_sequence:
            raise SaliencyReplayUnavailable("saliency source event ordering is invalid")
        kind = _kind(event)
        if kind in {
            "saliency-inference-recorded",
            "saliency-cache-hit",
            "saliency-fallback-recorded",
        }:
            if _kind(source) != "viewport-captured":
                raise SaliencyReplayUnavailable(
                    "saliency inference source event kind is invalid"
                )
            source_snapshot = _mapping(source.get("snapshot"))
            if _text(source_snapshot.get("id"), "") != _text(
                event.get("source_viewport_id"), ""
            ):
                raise SaliencyReplayUnavailable(
                    "saliency inference source viewport is invalid"
                )
        elif kind == "saliency-profiles-recorded":
            if _kind(source) == "viewport-captured":
                source_snapshot = _mapping(source.get("snapshot"))
                if _text(source_snapshot.get("id"), "") != _text(
                    event.get("source_viewport_id"), ""
                ):
                    raise SaliencyReplayUnavailable(
                        "saliency profile source viewport is invalid"
                    )
            elif _kind(source) in {
                "saliency-inference-recorded",
                "saliency-cache-hit",
            }:
                if _text(source.get("source_viewport_id"), "") != _text(
                    event.get("source_viewport_id"), ""
                ) or _text(source.get("artifact_namespace"), "") != _text(
                    event.get("artifact_namespace"), ""
                ):
                    raise SaliencyReplayUnavailable(
                        "saliency profile source namespace is invalid"
                    )
                if _canonical_provider_id(_text(source.get("provider_id"), "")) != (
                    _canonical_provider_id(_text(event.get("provider_id"), ""))
                ):
                    raise SaliencyReplayUnavailable(
                        "saliency profile source provider is invalid"
                    )
                if _text(source.get("cache_state"), "") != _text(
                    event.get("cache_state"), ""
                ):
                    raise SaliencyReplayUnavailable(
                        "saliency profile source cache state is invalid"
                    )
            else:
                raise SaliencyReplayUnavailable(
                    "saliency profile source event kind is invalid"
                )
        elif kind == "prominence-recorded":
            if source_event_id not in profile_event_ids:
                raise SaliencyReplayUnavailable(
                    "saliency operational source event kind is invalid"
                )
            if _kind(source) != "saliency-profiles-recorded":
                raise SaliencyReplayUnavailable(
                    "saliency operational source event kind is invalid"
                )
            if _text(source.get("source_viewport_id"), "") != _text(
                event.get("source_viewport_id"), ""
            ) or _text(source.get("artifact_namespace"), "") != _text(
                event.get("artifact_namespace"), ""
            ):
                raise SaliencyReplayUnavailable(
                    "saliency operational source namespace is invalid"
                )
            if _text(source.get("cache_state"), "") != _text(
                event.get("cache_state"), ""
            ):
                raise SaliencyReplayUnavailable(
                    "saliency operational cache state is invalid"
                )
            if _canonical_provider_id(_text(source.get("provider_id"), "")) != (
                _canonical_provider_id(_text(event.get("active_provider_id"), ""))
            ):
                raise SaliencyReplayUnavailable(
                    "saliency operational provider join is invalid"
                )
        else:
            raise SaliencyReplayUnavailable("saliency source event kind is invalid")
    if not profile_event_ids:
        raise SaliencyReplayUnavailable("saliency profile event linkage is missing")
    checksum_path = _secure_bundle_file(run_path, PurePosixPath("checksums.sha256"))
    if checksum_path is None:
        raise SaliencyReplayUnavailable("saliency checksums are unavailable")
    checksums: dict[str, str] = {}
    try:
        for line in (
            secure_read_bytes(checksum_path, "saliency checksums")
            .decode("utf-8")
            .splitlines()
        ):
            if "  " in line:
                digest, relative = line.split("  ", maxsplit=1)
                checksums[relative] = digest
        for relative in sorted(expected_paths):
            artifact = _secure_bundle_file(run_path, PurePosixPath(relative))
            if artifact is None or checksums.get(relative) is None:
                raise SaliencyReplayUnavailable("saliency artifact checksum is missing")
            try:
                content = secure_read_bytes(artifact, "saliency artifact")
            except (OSError, RuntimeError) as error:
                raise SaliencyReplayUnavailable(
                    "saliency artifact is unreadable"
                ) from error
            if len(content) > _MAX_SALIENCY_ARTIFACT_BYTES:
                raise SaliencyReplayUnavailable("saliency artifact exceeds size limit")
            if hashlib.sha256(content).hexdigest() != checksums[relative]:
                raise SaliencyReplayUnavailable("saliency artifact checksum is invalid")
            if PurePosixPath(relative).name.endswith(".npz"):
                prediction = _prediction_metadata_for_path(metadata, relative)
                try:
                    validate_saliency_native_map_content(
                        content,
                        expected_output_dimensions=(
                            int(
                                _number(
                                    prediction.get("output_dimensions", [0, 0])[0], 0
                                )
                            ),
                            int(
                                _number(
                                    prediction.get("output_dimensions", [0, 0])[1], 0
                                )
                            ),
                        ),
                        expected_geometry=_mapping(prediction.get("geometry")),
                    )
                except (IndexError, TypeError, ValueError) as error:
                    raise SaliencyReplayUnavailable(
                        "saliency native map is invalid"
                    ) from error
            elif PurePosixPath(relative).name.endswith("-heatmap.png"):
                try:
                    validate_saliency_heatmap_content(content)
                except ValueError as error:
                    raise SaliencyReplayUnavailable(
                        "saliency heatmap is invalid"
                    ) from error
    except (OSError, RuntimeError, UnicodeError):
        raise SaliencyReplayUnavailable("saliency checksums are unreadable")


def _aggregates_for_duration(
    profiles: list[dict[str, Any]], duration: str
) -> list[dict[str, Any]]:
    return [
        aggregate
        for profile in profiles
        for aggregate in _list_of_mappings(profile.get("aggregates"))
        if aggregate.get("duration") == duration
    ]


def _prediction_metadata_for_path(
    metadata: dict[str, Any], relative_path: str
) -> dict[str, Any]:
    duration = PurePosixPath(relative_path).stem
    prediction = next(
        (
            item
            for item in _list_of_mappings(metadata.get("predictions"))
            if item.get("duration") == duration
        ),
        None,
    )
    prediction_metadata = _mapping(_mapping(prediction).get("metadata"))
    if not prediction_metadata:
        raise ValueError("saliency prediction metadata is missing")
    return prediction_metadata


def _ranked_elements(
    aggregates: list[dict[str, Any]],
    snapshot: dict[str, Any] | None,
    duration: str,
) -> list[dict[str, Any]]:
    elements = {
        element["id"]: element for element in (snapshot or {}).get("elements", [])
    }
    ranked = sorted(
        aggregates,
        key=lambda aggregate: _number(aggregate.get("adjusted_score"), 0),
        reverse=True,
    )
    result: list[dict[str, Any]] = []
    for rank, aggregate in enumerate(ranked, start=1):
        element = elements.get(aggregate["element_id"], {})
        result.append(
            {
                "rank": rank,
                "element_id": aggregate["element_id"],
                "label": _text(element.get("label"), "Unlabelled element"),
                "role": _text(element.get("role"), "other"),
                "bounds": _mapping(element.get("bounds")),
                "duration": duration,
                "adjusted_score": aggregate.get("adjusted_score"),
                "visibility_fraction": aggregate.get("visibility_fraction"),
                "occlusion_fraction": aggregate.get("occlusion_fraction"),
            }
        )
    return result


def _saliency_runtime(
    saliency: list[dict[str, Any]], events: list[dict[str, Any]]
) -> dict[str, Any]:
    durations: list[dict[str, Any]] = []
    for group in saliency:
        for entry in _list_of_mappings(group.get("entries")):
            durations.append(
                {
                    "duration": entry.get("duration"),
                    "inference_duration_ms": _number(
                        entry.get("inference_duration_ms"), 0
                    ),
                    "cache_state": group.get("cache_state"),
                }
            )
    if not durations:
        for event in events:
            if _kind(event) in {"saliency-inference-recorded", "saliency-cache-hit"}:
                timings = event.get("timings_ms")
                if isinstance(timings, list | tuple):
                    timing_values = cast(list[object] | tuple[object, ...], timings)
                    durations.extend(
                        {
                            "duration": duration,
                            "inference_duration_ms": _number(timing, 0),
                            "cache_state": _text(
                                event.get("cache_state"), "unavailable"
                            ),
                        }
                        for duration, timing in zip(
                            _SALIENCY_DURATIONS, timing_values, strict=False
                        )
                    )
    return {
        "total_inference_ms": sum(
            _number(item.get("inference_duration_ms"), 0) for item in durations
        ),
        "durations": durations,
        "cache_states": _unique(
            _text(item.get("cache_state"), "unavailable") for item in durations
        ),
    }


def _saliency_stage_timeline(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _public_saliency_event(event)
        for event in events
        if _kind(event)
        in {
            "saliency-inference-recorded",
            "saliency-cache-hit",
            "saliency-profiles-recorded",
            "saliency-fallback-recorded",
        }
        or (_kind(event) == "prominence-recorded" and "search_stage" in event)
    ]


def _saliency_fallbacks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "viewport_id": _safe_identifier(event.get("viewport_id"), "unavailable"),
            "provider_id": _safe_identifier(event.get("provider_id"), "unavailable"),
            "fallback_provider_id": _safe_identifier(
                event.get("fallback_provider_id"), "unavailable"
            ),
            "search_stage": _safe_identifier(event.get("search_stage"), "unavailable"),
            "reason": _safe_text(event.get("reason"), "learned output unavailable"),
        }
        for event in events
        if _kind(event) == "saliency-fallback-recorded"
    ]


def _is_learned_provider(provider_id: str) -> bool:
    return provider_id in {"foveacast", "foveacast-prominence"}


def _public_saliency_event(event: dict[str, Any]) -> dict[str, Any]:
    kind = _kind(event)
    raw_timings = event.get("timings_ms")
    timing_values = (
        cast(list[object] | tuple[object, ...], raw_timings)
        if isinstance(raw_timings, list | tuple)
        else ()
    )
    result: dict[str, Any] = {
        "event_id": f"event-{int(_number(event.get('sequence'), 0))}",
        "sequence": int(_number(event.get("sequence"), 0)),
        "kind": kind,
        "viewport_id": _safe_identifier(event.get("viewport_id"), "unavailable"),
        "source_viewport_id": _safe_identifier(
            event.get("source_viewport_id"), "unavailable"
        ),
        "artifact_namespace": _safe_identifier(
            event.get("artifact_namespace"), "unavailable"
        ),
        "source_event_id": _safe_identifier(
            event.get("source_event_id"), "unavailable"
        ),
        "provider_id": _safe_identifier(event.get("provider_id"), "unavailable"),
        "active_provider_id": _safe_identifier(
            event.get("active_provider_id"), "unavailable"
        ),
        "search_stage": _safe_identifier(
            event.get("search_stage", event.get("stage")), "unavailable"
        ),
        "cache_state": _safe_identifier(event.get("cache_state"), "unavailable"),
        "execution_provider": _safe_identifier(
            event.get("execution_provider"), "unavailable"
        ),
        "model_checksums": [
            checksum
            for checksum in _strings(event.get("model_checksums"))
            if _safe_checksum(checksum) != "unavailable"
        ],
        "artifact_ids": [
            path
            for path in _strings(event.get("artifact_ids"))
            if _validated_saliency_artifact(path) is not None
        ],
        "selected_mixture": _safe_mixture(event.get("selected_mixture")),
        "timings_ms": [
            _number(value, 0) for value in timing_values if _is_number(value)
        ],
        "warnings": [
            _safe_text(value, "unavailable")
            for value in _strings(event.get("warnings"))
        ],
        "reason": _safe_text(event.get("reason"), "unavailable"),
    }
    return result


def _safe_mixture(value: object) -> list[list[object]]:
    values: list[list[object]] = []
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        values = [[key, item] for key, item in mapping.items()]
    elif isinstance(value, list | tuple):
        raw_values = cast(list[object] | tuple[object, ...], value)
        values = []
        for raw_value in raw_values:
            if isinstance(raw_value, list | tuple):
                pair = cast(list[object] | tuple[object, ...], raw_value)
                values.append(list(pair))
    result: list[list[object]] = []
    for item in values:
        if len(item) != 2:
            continue
        duration, weight = item
        if _text(duration) not in _SALIENCY_DURATIONS or not _is_number(weight):
            continue
        result.append([_text(duration), _number(weight, 0)])
    return result


def _safe_identifier(value: object, default: str = _REDACTED_IDENTIFIER) -> str:
    text = _text(value, "").strip()
    lowered = text.casefold()
    if (
        not text
        or ".." in text
        or any(marker in lowered for marker in _SENSITIVE_IDENTIFIER_MARKERS)
        or _SAFE_PUBLIC_IDENTIFIER.fullmatch(text) is None
    ):
        return default
    return text


def _safe_checksum(value: object) -> str:
    text = _text(value, "")
    if (
        len(text) == 64
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    ):
        return text
    return "unavailable"


def _safe_text(value: object, default: str) -> str:
    text = _text(value, default)
    if (
        not text
        or len(text) > 512
        or any(character in text for character in "\x00\r\n")
    ):
        return default
    lowered = text.casefold()
    if any(marker in lowered for marker in _SENSITIVE_IDENTIFIER_MARKERS):
        return _REDACTED_IDENTIFIER
    return text


def _observation(event: dict[str, Any]) -> dict[str, Any]:
    raw = _mapping(event.get("observation"))
    return {
        "sequence": _number(event.get("sequence"), 0),
        "viewport_id": _text(raw.get("viewport_id")),
        "newly_revealed_elements": _public_visible_elements(
            raw.get("newly_revealed_elements")
        ),
        "remembered_elements": _public_visible_elements(raw.get("remembered_elements")),
        "region_context": _public_region(raw.get("region_context")),
    }


def _public_visible_elements(value: object) -> list[dict[str, Any]]:
    return [
        {
            "id": _text(item.get("id")),
            "role": _text(item.get("role"), "other"),
            "label": _text(item.get("label"), "Unlabelled element"),
            "bounds": _bounds(item.get("bounds")),
            "visibility_fraction": _bounded(item.get("visibility_fraction")),
            "actionable": bool(item.get("actionable", False)),
            "disabled": bool(item.get("disabled", False)),
            "region_id": _optional_text(item.get("region_id")),
        }
        for item in _list_of_mappings(value)
        if item.get("id") and _bounds(item.get("bounds")) is not None
    ]


def _public_region(value: object) -> dict[str, str] | None:
    region = _mapping(value)
    if not region.get("id"):
        return None
    return {"id": _text(region.get("id")), "label": _text(region.get("label"))}


def _public_event(event: dict[str, Any]) -> dict[str, Any]:
    kind = _kind(event)
    sequence = int(_number(event.get("sequence"), 0))
    result: dict[str, Any] = {
        "event_id": f"event-{sequence}" if sequence > 0 else "",
        "sequence": sequence,
        "kind": kind,
    }
    if kind == "viewport-captured":
        snapshot = _mapping(event.get("snapshot"))
        result["viewport_id"] = _text(snapshot.get("id"))
    elif kind == "observation-recorded":
        result["observation"] = _observation(event)
    elif kind in {"action-proposed", "action-executed"}:
        result["action"] = _public_action(event.get("action"))
        for key in ("reason", "succeeded", "error", "viewport_id"):
            if key in event and key not in _PRIVATE_KEYS:
                result[key] = _safe_value(event[key])
    elif kind in {
        "saliency-inference-recorded",
        "saliency-cache-hit",
        "saliency-profiles-recorded",
        "saliency-fallback-recorded",
    }:
        result.update(_public_saliency_event(event))
    elif kind in {
        "coarse-scent",
        "coarse-scent-recorded",
        "full-scent",
        "full-scent-recorded",
        "prominence-scored",
        "prominence-recorded",
    }:
        result["scores"] = _public_scores(
            event.get("scores"), prominence="prominence" in kind
        )
    elif kind == "model-call-recorded":
        result["record"] = _public_model_call(event.get("record"))
    elif kind == "attention-selection-recorded":
        result.update(
            {
                "viewport_id": _text(event.get("viewport_id")),
                "selected_ids": _strings(event.get("selected_ids")),
                "selection_mode": _text(event.get("selection_mode")),
                "region_id": _optional_text(event.get("region_id")),
                "element_probabilities": _number_mapping(
                    event.get("element_probabilities")
                ),
                "region_probabilities": _number_mapping(
                    event.get("region_probabilities")
                ),
                "recovery_selected_ids": _strings(event.get("recovery_selected_ids")),
            }
        )
    elif kind == "verification-recorded":
        result["verification"] = _public_verification(event.get("result"))
    elif kind == "run-terminated":
        result["outcome"] = _public_outcome(event.get("outcome"))
    elif kind == "model-failure":
        result.update(
            {
                "role": _text(event.get("role"), "unknown"),
                "reason": _optional_text(event.get("reason")),
                "response_summary": _safe_value(event.get("response_summary")),
            }
        )
    elif kind in {
        "repeated-fixture-input",
        "repeated-action-detected",
        "repeated-action-cycle",
        "no-progress-recovery",
        "no-progress-detected",
        "fixture-input-completed",
        "model-call-budget-exhausted",
    }:
        if event.get("action") is not None:
            result["action"] = _public_action(event.get("action"))
        for key in (
            "element_id",
            "fixture_key",
            "count",
            "cycle_length",
            "limit",
            "model_calls",
            "reason",
        ):
            if key in event:
                result[key] = _safe_value(event[key])
    elif kind in {"decision-recorded", "agent-claim", "action-rejected"}:
        if event.get("action") is not None:
            result["action"] = _public_action(event.get("action"))
        for key in ("reason", "claimed_success", "message", "error"):
            if key in event:
                result[key] = _safe_value(event[key])
    else:
        for key in ("reason", "claimed_success", "message"):
            if key in event:
                result[key] = _safe_value(event[key])
    return result


def _public_action(value: object) -> dict[str, Any]:
    action = _mapping(value)
    allowed = ("kind", "element_id", "direction", "duration_seconds", "reason")
    return {key: _safe_value(action[key]) for key in allowed if key in action}


def _public_scores(value: object, *, prominence: bool) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for score in _list_of_mappings(value):
        item: dict[str, Any] = {}
        if score.get("element_id"):
            item["element_id"] = _text(score.get("element_id"))
        for key in (
            "score",
            "raw_score",
            "normalized_probability",
            "first_notice_probability",
            "notice_within_budget_probability",
        ):
            if key in score and _is_number(score[key]):
                item[key] = _number(score[key], 0)
        if prominence:
            for key in ("feature_contributions", "raw_values", "normalized_values"):
                item[key] = _number_mapping(score.get(key))
        scores.append(item)
    return scores


def _public_verification(value: object) -> dict[str, Any]:
    verification = _mapping(value)
    return {
        "verified": bool(verification.get("verified", False)),
        "evidence_ids": _strings(verification.get("evidence_ids")),
        "details": _optional_text(verification.get("details")),
    }


def _public_outcome(value: object) -> str:
    outcome = _mapping(value)
    return _first_string(outcome.get("kind"), value, "unknown")


def _prominence(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sequence": _number(event.get("sequence"), 0),
            "viewport_id": _text(event.get("viewport_id")),
            "scores": _public_scores(event.get("scores"), prominence=True),
        }
        for event in events
        if _kind(event) in {"prominence-recorded", "prominence-scored"}
    ]


def _scent_records(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sequence": _number(event.get("sequence"), 0),
            "kind": _kind(event),
            "viewport_id": _text(event.get("viewport_id")),
            "scores": _public_scores(event.get("scores"), prominence=False),
        }
        for event in events
        if _kind(event)
        in {
            "coarse-scent",
            "coarse-scent-recorded",
            "full-scent",
            "full-scent-recorded",
        }
    ]


def _selections(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sequence": _number(event.get("sequence"), 0),
            "viewport_id": _text(event.get("viewport_id")),
            "selected_ids": _strings(event.get("selected_ids")),
            "selection_mode": _text(event.get("selection_mode")),
            "region_id": _optional_text(event.get("region_id")),
            "element_probabilities": _number_mapping(
                event.get("element_probabilities")
            ),
            "region_probabilities": _number_mapping(event.get("region_probabilities")),
        }
        for event in events
        if _kind(event) == "attention-selection-recorded"
    ]


def _decisions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sequence": _number(event.get("sequence"), 0),
            "reason": _optional_text(event.get("reason")),
            "action": _public_action(event.get("action")),
            "claimed_success": bool(event.get("claimed_success", False)),
        }
        for event in events
        if _kind(event) in {"decision-recorded", "action-proposed", "agent-claim"}
    ]


def _model_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in events:
        if _kind(event) != "model-call-recorded":
            continue
        record = _public_model_call(event.get("record"))
        if record:
            records.append(record)
    return records


def _public_model_call(value: object) -> dict[str, Any]:
    record = _mapping(value)
    if not record:
        return {}
    usage = _mapping(record.get("token_usage"))
    return {
        "role": _text(record.get("role"), "unknown"),
        "model": _safe_identifier(
            _first_string(record.get("model"), record.get("model_id")),
            "unavailable",
        ),
        "endpoint_origin": _optional_text(record.get("endpoint_origin")),
        "prompt_digest": _optional_text(record.get("prompt_digest")),
        "schema_version": _optional_text(record.get("schema_version")),
        "attempts": int(_number(record.get("attempts"), 1)),
        "latency_ms": int(_number(record.get("latency_ms"), 0)),
        "token_usage": {
            "prompt_tokens": int(_number(usage.get("prompt_tokens"), 0)),
            "completion_tokens": int(_number(usage.get("completion_tokens"), 0)),
            "total_tokens": int(_number(usage.get("total_tokens"), 0)),
        },
        "request": _safe_value(record.get("request")),
        "response": _safe_value(record.get("response")),
        "retries": [
            {
                "attempt": int(_number(item.get("attempt"), 0)),
                "reason": _text(item.get("reason")),
                "status_code": (
                    int(_number(item.get("status_code"), 0))
                    if item.get("status_code") is not None
                    else None
                ),
                "delay_seconds": _number(item.get("delay_seconds"), 0),
            }
            for item in _list_of_mappings(record.get("retries"))
        ],
    }


def _actions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sequence": _number(event.get("sequence"), 0),
            "action": _public_action(event.get("action")),
            "succeeded": event.get("succeeded"),
            "error": _optional_text(event.get("error")),
        }
        for event in events
        if _kind(event) == "action-executed"
    ]


def _verification(
    events: list[dict[str, Any]], result: dict[str, Any]
) -> dict[str, Any]:
    records = [
        _public_verification(event.get("result"))
        for event in events
        if _kind(event) == "verification-recorded"
    ]
    if records:
        return records[-1]
    return _public_verification(result.get("verification"))


def _outcome(events: list[dict[str, Any]], result: dict[str, Any]) -> str:
    terminal = [
        _public_outcome(event.get("outcome"))
        for event in events
        if _kind(event) == "run-terminated"
    ]
    if terminal:
        return terminal[-1]
    return _public_outcome(result.get("outcome"))


def _memory(attention: dict[str, Any]) -> list[dict[str, Any]]:
    memory: list[dict[str, Any]] = []
    for entry in _list_of_mappings(attention.get("memory")):
        memory.append(
            {
                "key": _text(entry.get("key", entry.get("element_id", "memory"))),
                "value": _text(entry.get("value", entry.get("label", ""))),
                "strength": _bounded(entry.get("strength"), default=1),
                "age": _number(entry.get("age"), 0),
                "importance": _bounded(entry.get("importance"), default=0.5),
                "is_failure": bool(entry.get("is_failure", False)),
            }
        )
    return memory


def _manifests(
    manifest: dict[str, Any], result: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    terminal = _list_of_mappings(state.get("provider_manifests"))
    if not terminal:
        terminal = _list_of_mappings(result.get("provider_manifests"))
    return {
        "endpoint_origin": _optional_text(manifest.get("endpoint_origin")),
        "model_ids": _safe_string_mapping(manifest.get("model_ids")),
        "prompt_versions": _safe_string_mapping(manifest.get("prompt_versions")),
        "provider_versions": _safe_string_mapping(manifest.get("provider_versions")),
        "provider_manifests": [
            {
                "provider_id": _safe_identifier(item.get("provider_id"), "unavailable"),
                "role": _safe_identifier(item.get("role"), "unavailable"),
                "model_id": (
                    _safe_identifier(item.get("model_id"), "unavailable")
                    if item.get("model_id") is not None
                    else None
                ),
                "endpoint_origin": _optional_text(item.get("endpoint_origin")),
                "version": _safe_identifier(item.get("version"), "unavailable"),
                "prompt_version": (
                    _safe_identifier(item.get("prompt_version"), "unavailable")
                    if item.get("prompt_version") is not None
                    else None
                ),
                "schema_version": (
                    _safe_identifier(item.get("schema_version"), "unavailable")
                    if item.get("schema_version") is not None
                    else None
                ),
            }
            for item in terminal
        ],
    }


def _extract_metrics(result: dict[str, Any]) -> dict[str, Any]:
    candidates: tuple[object, ...] = (
        result.get("metrics"),
        result.get("run_metrics"),
        _mapping(result.get("evaluation")).get("metrics"),
        _mapping(result.get("evaluation")).get("run_metrics"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return cast(dict[str, Any], candidate)
    return {}


def _provider_identity_failures(
    manifest: dict[str, Any],
    result: dict[str, Any],
    state: dict[str, Any],
    metrics: dict[str, Any],
    *,
    saliency: Sequence[dict[str, Any]] = (),
) -> tuple[str, list[str]]:
    """Validate canonical provider identity across every persisted evidence surface."""

    raw_provider = manifest.get("prominence_provider_id", "heuristic")
    provider = _canonical_provider_id(raw_provider) or "unavailable"
    failures: list[str] = []
    if provider == "unavailable":
        failures.append("manifest prominence provider ID is invalid")
    if (
        metrics
        and _canonical_provider_id(metrics.get("prominence_provider_id")) != provider
    ):
        failures.append("metrics prominence provider ID does not match manifest")
    spec = _mapping(state.get("spec"))
    if (
        "prominence_provider_id" in spec
        and _canonical_provider_id(spec.get("prominence_provider_id")) != provider
    ):
        failures.append("run spec prominence provider ID does not match manifest")
    nested = [
        *_list_of_mappings(manifest.get("provider_manifests")),
        *_list_of_mappings(state.get("provider_manifests")),
        *_list_of_mappings(result.get("provider_manifests")),
    ]
    for item in nested:
        if item.get("role") != "prominence":
            continue
        nested_provider = _canonical_provider_id(item.get("provider_id"))
        if nested_provider is None:
            failures.append("nested prominence provider ID is invalid")
        elif nested_provider != provider:
            failures.append("nested prominence provider ID does not match manifest")
    for group in saliency:
        if not group.get("replay_available", False):
            if "provider identity" in _text(group.get("replay_error"), ""):
                failures.append("saliency provider identity does not match manifest")
            continue
        for field_name in ("provider_id", "active_provider_id"):
            if _canonical_provider_id(group.get(field_name)) != provider:
                failures.append("saliency provider ID does not match manifest")
        for entry in _list_of_mappings(group.get("entries")):
            if _canonical_provider_id(entry.get("provider_id")) != provider:
                failures.append("saliency metadata provider ID does not match manifest")
            checksums = set(_strings(group.get("model_checksums")))
            checksum = _safe_checksum(entry.get("model_checksum"))
            if checksums and checksum not in checksums:
                failures.append("saliency model checksum does not match timeline")
    return provider, failures


def _canonical_provider_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return _CANONICAL_PROVIDER_IDS.get(value)


def _metric_rows(
    metrics: dict[str, Any], verification: dict[str, Any], outcome: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    explicit_names: set[str] = set()
    explicit: object = metrics.get("metrics")
    if isinstance(explicit, list):
        for item in _list_of_mappings(cast(object, explicit)):
            if _evidence_class(
                item.get("evidence_class")
            ) != "unsupported-human-claim" and _is_number(item.get("value")):
                row = _metric_row(item.get("name"), item.get("value"), item)
                rows.append(row)
                explicit_names.add(row["name"])
    for name, value in metrics.items():
        if name in _METRIC_IDENTITY_KEYS or name in {"metrics", "evidence"}:
            continue
        rendered_name = name.replace("_", "-")
        if rendered_name in explicit_names:
            continue
        if name == "discovery_cost" and isinstance(value, dict):
            discovery_cost = cast(dict[str, Any], value)
            total: object = discovery_cost.get("total")
            if _is_number(total):
                rows.append(_metric_row("discovery-cost", total, discovery_cost))
            continue
        if _is_number(value):
            rows.append(_metric_row(rendered_name, value, metrics))
    if not any(row["name"] == "verified-completion" for row in rows):
        rows.append(
            _metric_row(
                "verified-completion",
                float(verification.get("verified", False)),
                {"evidence_class": "deterministic-fact"},
            )
        )
    if not any(row["name"] == "outcome" for row in rows):
        rows.append(
            {
                "name": "outcome",
                "value": outcome,
                "evidence_class": "deterministic-fact",
                "evidence_ids": [],
            }
        )
    return rows


def _metric_row(name: object, value: object, source: object) -> dict[str, Any]:
    metric_name = _text(name, "metric")
    return {
        "name": metric_name,
        "value": _number(value, 0),
        "evidence_class": _metric_evidence_class(metric_name, source),
        "evidence_ids": _strings(_mapping(source).get("evidence_ids")),
    }


def _metric_evidence_class(name: str, source: object) -> str:
    explicit = _mapping(source).get("evidence_class")
    if explicit is not None:
        return _evidence_class(explicit)
    deterministic = {
        "backtracks",
        "false-success",
        "inspected-elements",
        "inspected-regions",
        "navigation-depth",
        "outcome",
        "recovery-actions",
        "scrolls",
        "verified-completion",
        "wrong-actions",
    }
    return "deterministic-fact" if name in deterministic else "model-estimate"


def _evidence(
    result: dict[str, Any], metrics: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    result_records = result.get("evidence")
    records: object = metrics.get("evidence")
    if isinstance(result_records, list | tuple):
        records = cast(object, result_records)
    supported: list[dict[str, Any]] = []
    limitations: list[str] = []
    for item in _list_of_mappings(records):
        evidence_class = _evidence_class(item.get("evidence_class"))
        description = _text(item.get("description"), "Evidence record")
        if evidence_class == "unsupported-human-claim":
            limitations.append(f"Unsupported human claim excluded: {description}")
            continue
        supported.append(
            {
                "evidence_id": _text(item.get("evidence_id")),
                "evidence_class": evidence_class,
                "description": description,
                "source_event_ids": _strings(item.get("source_event_ids")),
            }
        )
    return supported, limitations


def _findings(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    findings: list[dict[str, Any]] = []
    limitations: list[str] = []
    for item in _list_of_mappings(result.get("findings")):
        evidence_class = _evidence_class(item.get("evidence_class"))
        if evidence_class == "unsupported-human-claim":
            limitations.append(
                f"Unsupported human claim finding excluded: {_text(item.get('category'), 'finding')}"
            )
            continue
        findings.append(
            {
                "finding_id": _text(item.get("finding_id")),
                "category": _text(item.get("category")),
                "severity": _text(item.get("severity")),
                "reproducibility": _text(item.get("reproducibility")),
                "evidence_class": evidence_class,
                "evidence_ids": _strings(item.get("evidence_ids")),
                "limitations": _strings(item.get("limitations")),
                "explanation": _optional_text(item.get("generated_explanation")),
                "title": _text(item.get("title"), _text(item.get("category"))),
                "cause": _text(
                    item.get("cause"),
                    _text(item.get("generated_explanation"), "Cause unavailable"),
                ),
                "run_ids": _strings(item.get("run_ids")),
                "viewport_ids": _strings(item.get("viewport_ids")),
                "element_ids": _strings(item.get("element_ids")),
                "supporting_metrics": _number_mapping(item.get("supporting_metrics")),
                "action_sequence": _strings(item.get("action_sequence")),
                "replay_links": _strings(item.get("replay_links")),
            }
        )
    return findings, limitations


def _comparison_rows(runs: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for run in runs:
        if not run["trusted"] or not run["comparison_valid"] or not run["metrics"]:
            continue
        groups[
            (
                run["scenario_id"],
                run["persona_id"],
                run["policy"],
                run["version_id"],
                _text(run.get("prominence_provider_id"), "heuristic"),
                int(run["model_trial"]),
            )
        ].append(run)
    rows: list[dict[str, Any]] = []
    for identity in sorted(groups):
        (
            scenario_id,
            persona_id,
            policy,
            version_id,
            prominence_provider_id,
            model_trial,
        ) = identity
        grouped = groups[identity]
        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_label": grouped[0]["scenario_label"],
                "persona_id": persona_id,
                "persona_label": grouped[0]["persona_label"],
                "policy": policy,
                "version_id": version_id,
                "version_label": grouped[0]["version_label"],
                "prominence_provider_id": prominence_provider_id,
                "model_trial": model_trial,
                "reproducibility_label": _aggregate_reproducibility_label(grouped),
                "run_count": len(grouped),
                "verified_rate": sum(run["verified"] for run in grouped) / len(grouped),
                "discovery_cost": _median_metric(grouped, "discovery-cost"),
                "wrong_actions": _median_metric(grouped, "wrong-actions"),
                "backtracks": _median_metric(grouped, "backtracks"),
                "estimated_task_seconds": _median_projection(
                    grouped, "user_effort", "estimated_task_seconds"
                ),
                "model_calls": _median_projection(
                    grouped, "analysis_cost", "model_calls"
                ),
                "analysis_latency_ms": _median_projection(
                    grouped, "analysis_cost", "latency_ms"
                ),
                "analysis_tokens": _median_projection(
                    grouped, "analysis_cost", "total_tokens"
                ),
            }
        )
    return rows


def _aggregate_reproducibility_label(runs: list[dict[str, Any]]) -> str:
    labels = sorted({_text(run.get("reproducibility_label"), "seeded") for run in runs})
    return "; ".join(labels) or "seeded"


def _run_overview_rows(
    runs: tuple[dict[str, Any], ...], gate_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        gate = _gate_for_run(run, gate_rows)
        user_effort = _mapping(run.get("user_effort"))
        analysis_cost = _mapping(run.get("analysis_cost"))
        rows.append(
            {
                "run_id": run["run_id"],
                "scenario_id": run["scenario_id"],
                "scenario_label": run["scenario_label"],
                "version_id": run["version_id"],
                "version_label": run["version_label"],
                "persona_id": run["persona_id"],
                "persona_label": run["persona_label"],
                "policy": run["policy"],
                "prominence_provider_id": _text(
                    run.get("prominence_provider_id"), "heuristic"
                ),
                "model_trial": int(_number(run.get("model_trial"), 0)),
                "reproducibility_label": _text(
                    run.get("reproducibility_label"), "seeded"
                ),
                "user_actions": len(run["actions"]),
                "observations": len(run["observations"]),
                "discovery_cost": _format_overview_cost(
                    _median_metric([run], "discovery-cost")
                ),
                "estimated_task_seconds": _format_seconds(
                    _number(user_effort.get("estimated_task_seconds"), 0)
                ),
                "model_calls": int(_number(analysis_cost.get("model_calls"), 0)),
                "analysis_latency": _format_milliseconds(
                    _number(analysis_cost.get("latency_ms"), 0)
                ),
                "analysis_tokens": int(_number(analysis_cost.get("total_tokens"), 0)),
                "outcome": run["outcome"],
                "stage": run["stage"],
                "terminal_state": run["terminal_state"],
                "verified": run["verified"],
                "failure_reason": run["failure_reason"] or "None recorded",
                "gate_status": gate["label"],
                "gate_reason": gate["reason"],
                "status_class": run["status_class"],
            }
        )
    return rows


def _gate_for_run(
    run: dict[str, Any], gate_rows: list[dict[str, Any]]
) -> dict[str, str]:
    matching = [
        row
        for row in gate_rows
        if row["scenario_id"] == run["scenario_id"]
        and row["persona_id"] == run["persona_id"]
        and row["policy"] == run["policy"]
        and row["prominence_provider_id"]
        == _text(run.get("prominence_provider_id"), "heuristic")
        and int(_number(row.get("model_trial"), 0))
        == int(_number(run.get("model_trial"), 0))
        and run["version_id"] in {row["baseline_version"], row["improved_version"]}
    ]
    if not matching:
        return {
            "label": "Gate unavailable",
            "reason": "No directional comparison recorded for this run.",
        }
    if any(not row["available"] for row in matching):
        reasons = _unique(
            reason
            for row in matching
            if not row["available"]
            for reason in row["reasons"]
        )
        return {
            "label": "Gate unavailable",
            "reason": "; ".join(reasons) or "Comparison evidence is unavailable.",
        }
    if any(not row["passed"] for row in matching):
        reasons = _unique(reason for row in matching for reason in row["reasons"])
        return {
            "label": "Gate fail",
            "reason": "; ".join(reasons) or "Directional comparison failed.",
        }
    return {"label": "Gate pass", "reason": "All directional checks passed."}


def _focused_acceptance(summary: dict[str, Any]) -> dict[str, Any]:
    """Expose allowlisted focused-gate evidence in experiment reports."""

    focused = _mapping(summary.get("focused_acceptance"))
    if not focused:
        return {}
    gate = _mapping(focused.get("gate"))
    cells: list[dict[str, Any]] = []
    for raw_cell in _list_of_mappings(focused.get("cells")):
        key = _mapping(raw_cell.get("key"))
        operational = _mapping(raw_cell.get("operational"))
        cells.append(
            {
                "scenario_id": _text(key.get("scenario_id"), "unknown"),
                "version_id": _text(key.get("version_id"), "unknown"),
                "provider_id": _text(key.get("provider_id"), "unknown"),
                "status": _text(operational.get("status"), "unavailable"),
                "latency_ms": _number(operational.get("latency_ms"), 0),
                "peak_rss_bytes": _number(operational.get("peak_rss_bytes"), 0),
                "sampler_provider": _text(
                    operational.get("sampler_provider"), "unknown"
                ),
                "synthetic": bool(operational.get("synthetic", False)),
            }
        )
    return {
        "gate": {
            "passed": bool(gate.get("passed", False)),
            "reasons": _strings(gate.get("reasons")),
            "cell_count": int(_number(gate.get("cell_count"), 0)),
            "paired_cell_count": int(_number(gate.get("paired_cell_count"), 0)),
            "learned_cell_count": int(_number(gate.get("learned_cell_count"), 0)),
        },
        "cells": cells,
    }


def _gate_rows(
    summary: dict[str, Any], runs: tuple[dict[str, Any], ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in _list_of_mappings(summary.get("variant_comparisons")):
        baseline = _mapping(comparison.get("baseline"))
        improved = _mapping(comparison.get("improved"))
        gate = _mapping(comparison.get("gate"))
        available = _comparison_runs_are_trusted(baseline, improved, runs)
        scenario_id = _text(baseline.get("scenario_id"))
        persona_id = _text(baseline.get("persona_id"))
        policy = _text(baseline.get("policy"))
        prominence_provider_id = _text(
            baseline.get("prominence_provider_id"), "heuristic"
        )
        model_trial = int(_number(baseline.get("model_trial"), 0))
        matching = next(
            (
                run
                for run in runs
                if run["scenario_id"] == scenario_id
                and run["persona_id"] == persona_id
                and run["policy"] == policy
                and _text(run.get("prominence_provider_id"), "heuristic")
                == prominence_provider_id
                and int(_number(run.get("model_trial"), 0)) == model_trial
                and run["trusted"]
                and run["comparison_valid"]
            ),
            None,
        )
        if matching is None:
            matching = {}
        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_label": matching.get("scenario_label", scenario_id),
                "persona_id": persona_id,
                "persona_label": matching.get("persona_label", persona_id),
                "policy": policy,
                "prominence_provider_id": prominence_provider_id,
                "model_trial": model_trial,
                "reproducibility_label": _text(
                    matching.get("reproducibility_label"), "seeded"
                ),
                "baseline_version": _text(
                    baseline.get("application_version_id"), "defective"
                ),
                "improved_version": _text(
                    improved.get("application_version_id"), "improved"
                ),
                "paired_seed_count": int(_number(gate.get("paired_seed_count"), 0)),
                "available": available,
                "passed": available and bool(gate.get("passed", False)),
                "reasons": (
                    _strings(gate.get("reasons"))
                    if available
                    else [
                        "Directional gate unavailable because one or more comparison runs are missing or untrusted."
                    ]
                ),
            }
        )
    groups: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for run in runs:
        if (
            run["trusted"]
            and run["comparison_valid"]
            and run["ux_sample_valid"]
            and run["metrics"]
        ):
            groups[
                (
                    run["scenario_id"],
                    run["persona_id"],
                    run["policy"],
                    _text(run.get("prominence_provider_id"), "heuristic"),
                    int(_number(run.get("model_trial"), 0)),
                )
            ].append(run)
    for identity in sorted(groups):
        derived = _derived_gate_row(groups[identity])
        if derived is not None:
            rows = [
                row
                for row in rows
                if (
                    row["scenario_id"],
                    row["persona_id"],
                    row["policy"],
                    _text(row.get("prominence_provider_id"), "heuristic"),
                    int(_number(row.get("model_trial"), 0)),
                )
                != identity
            ]
            rows.append(derived)
    return rows


def _derived_gate_row(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline = {
        int(run["seed"]): run for run in runs if _version_kind(run) == "defective"
    }
    improved = {
        int(run["seed"]): run for run in runs if _version_kind(run) == "improved"
    }
    seeds = tuple(sorted(set(baseline) & set(improved)))
    if not seeds:
        return None
    cost_deltas = _paired_metric_deltas(baseline, improved, seeds, "discovery-cost")
    wrong_deltas = _paired_metric_deltas(baseline, improved, seeds, "wrong-actions")
    backtrack_deltas = _paired_metric_deltas(baseline, improved, seeds, "backtracks")
    if cost_deltas is None or wrong_deltas is None or backtrack_deltas is None:
        return None
    baseline_completion = sum(baseline[seed]["verified"] for seed in seeds) / len(seeds)
    improved_completion = sum(improved[seed]["verified"] for seed in seeds) / len(seeds)
    gated_seeds = tuple(
        seed
        for seed in seeds
        if not (not baseline[seed]["verified"] and improved[seed]["verified"])
    )
    gated_cost_deltas = _paired_metric_deltas(
        baseline, improved, gated_seeds, "discovery-cost"
    )
    gated_wrong_deltas = _paired_metric_deltas(
        baseline, improved, gated_seeds, "wrong-actions"
    )
    gated_backtrack_deltas = _paired_metric_deltas(
        baseline, improved, gated_seeds, "backtracks"
    )
    checks = (
        (
            "paired median discovery cost did not decrease",
            not gated_seeds
            or (
                gated_cost_deltas is not None and _median_values(gated_cost_deltas) < 0
            ),
        ),
        (
            "paired median wrong-action burden increased",
            not gated_seeds
            or (
                gated_wrong_deltas is not None
                and _median_values(gated_wrong_deltas) <= 0
            ),
        ),
        (
            "paired median backtrack burden increased",
            not gated_seeds
            or (
                gated_backtrack_deltas is not None
                and _median_values(gated_backtrack_deltas) <= 0
            ),
        ),
        (
            "verified completion rate regressed",
            improved_completion >= baseline_completion,
        ),
    )
    sample = runs[0]
    reasons = [reason for reason, passed in checks if not passed]
    return {
        "scenario_id": sample["scenario_id"],
        "scenario_label": sample["scenario_label"],
        "persona_id": sample["persona_id"],
        "persona_label": sample["persona_label"],
        "policy": sample["policy"],
        "prominence_provider_id": _text(
            sample.get("prominence_provider_id"), "heuristic"
        ),
        "model_trial": int(_number(sample.get("model_trial"), 0)),
        "reproducibility_label": _aggregate_reproducibility_label(runs),
        "baseline_version": baseline[seeds[0]]["version_id"],
        "improved_version": improved[seeds[0]]["version_id"],
        "paired_seed_count": len(seeds),
        "available": True,
        "passed": not reasons,
        "reasons": reasons,
    }


def _version_kind(run: dict[str, Any]) -> str | None:
    identity = f"{run['version_id']} {run['version_label']}".casefold()
    if "improved" in identity:
        return "improved"
    if "defective" in identity:
        return "defective"
    return None


def _paired_metric_deltas(
    baseline: dict[int, dict[str, Any]],
    improved: dict[int, dict[str, Any]],
    seeds: tuple[int, ...],
    name: str,
) -> list[float] | None:
    deltas: list[float] = []
    for seed in seeds:
        baseline_value = _median_metric([baseline[seed]], name)
        improved_value = _median_metric([improved[seed]], name)
        if baseline_value is None or improved_value is None:
            return None
        deltas.append(improved_value - baseline_value)
    return deltas


def _median_values(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _comparison_runs_are_trusted(
    baseline: dict[str, Any],
    improved: dict[str, Any],
    runs: tuple[dict[str, Any], ...],
) -> bool:
    for cell in (baseline, improved):
        matching = [
            run
            for run in runs
            if run["scenario_id"] == _text(cell.get("scenario_id"))
            and run["persona_id"] == _text(cell.get("persona_id"))
            and run["policy"] == _text(cell.get("policy"))
            and run["version_id"] == _text(cell.get("application_version_id"))
            and _text(run.get("prominence_provider_id"), "heuristic")
            == _text(cell.get("prominence_provider_id"), "heuristic")
            and int(_number(run.get("model_trial"), 0))
            == int(_number(cell.get("model_trial"), 0))
        ]
        if not matching or any(
            not run["trusted"] or not run["comparison_valid"] for run in matching
        ):
            return False
    return True


def _median_metric(runs: list[dict[str, Any]], name: str) -> float | None:
    values = [
        float(row["value"])
        for run in runs
        for row in run["metrics"]
        if row["name"] == name and _is_number(row["value"])
    ]
    if not values:
        return None
    values.sort()
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _format_overview_cost(value: float | None) -> str | None:
    return f"{value:.1f}" if value is not None else None


def _median_projection(
    runs: list[dict[str, Any]], projection: str, field: str
) -> float | None:
    values = [
        _number(value, 0)
        for run in runs
        if _is_number(value := _mapping(run.get(projection)).get(field))
    ]
    if not values:
        return None
    values.sort()
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _format_seconds(value: float) -> str:
    return f"{value:.1f} s"


def _format_milliseconds(value: float) -> str:
    return f"{int(value)} ms"


def _user_effort(
    actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    action_count = len(actions)
    observation_count = len(observations)
    discovery = _mapping(metrics.get("discovery_cost")).get("total")
    return {
        "formula_version": "simulated-task-time-v1",
        "action_count": action_count,
        "observation_count": observation_count,
        "estimated_task_seconds": observation_count * 1.35 + action_count * 1.1,
        "discovery_cost": _number(discovery, 0) if _is_number(discovery) else None,
    }


def _analysis_cost(model_calls: list[dict[str, Any]]) -> dict[str, Any]:
    usages = [_mapping(record.get("token_usage")) for record in model_calls]
    return {
        "model_calls": len(model_calls),
        "attempts": sum(
            int(_number(record.get("attempts"), 0)) for record in model_calls
        ),
        "latency_ms": sum(
            int(_number(record.get("latency_ms"), 0)) for record in model_calls
        ),
        "prompt_tokens": sum(
            int(_number(usage.get("prompt_tokens"), 0)) for usage in usages
        ),
        "completion_tokens": sum(
            int(_number(usage.get("completion_tokens"), 0)) for usage in usages
        ),
        "total_tokens": sum(
            int(_number(usage.get("total_tokens"), 0)) for usage in usages
        ),
        "monetary_estimate": None,
        "monetary_reason": (
            "Monetary estimate unavailable: model pricing is not configured."
        ),
    }


def _evidence_summary(runs: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for run in runs:
        for evidence in run["evidence"]:
            counts[evidence["evidence_class"]] += 1
        for finding in run["findings"]:
            counts[finding["evidence_class"]] += 1
    return [
        {"evidence_class": evidence_class, "count": counts[evidence_class]}
        for evidence_class in (
            "deterministic-fact",
            "model-estimate",
            "unsupported-human-claim",
        )
    ]


def _failure_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = (
        "run_id",
        "error_type",
        "stage",
        "terminal_state",
        "reason",
        "scenario_id",
        "application_version_id",
        "persona_id",
        "policy",
        "prominence_provider_id",
        "seed",
        "model_trial",
    )
    return [
        {key: _safe_value(item.get(key)) for key in allowed if key in item}
        for item in _list_of_mappings(summary.get("failures"))
        if item.get("run_id")
    ]


def _merge_failure(run: dict[str, Any], failure: dict[str, Any]) -> None:
    run["failed"] = True
    stage = _text(failure.get("stage"), run["stage"])
    reason = _text(failure.get("reason"), run["failure_reason"])
    run["stage"] = stage
    run["terminal_state"] = _text(failure.get("terminal_state"), run["terminal_state"])
    if stage == "evaluation":
        run["evaluation_failure_reason"] = reason
    elif not run.get("terminal_reason"):
        run["terminal_reason"] = reason
    run["failure_reason"] = "; ".join(
        _unique(
            (
                run.get("terminal_reason"),
                run.get("evaluation_failure_reason"),
                *run.get("integrity_failures", []),
            )
        )
    )
    run["status_class"] = _status_class(
        run["outcome"], run["stage"], run["terminal_state"], run["trusted"]
    )
    for target, source in (
        ("scenario_id", "scenario_id"),
        ("version_id", "application_version_id"),
        ("persona_id", "persona_id"),
        ("policy", "policy"),
        ("prominence_provider_id", "prominence_provider_id"),
        ("seed", "seed"),
        ("model_trial", "model_trial"),
    ):
        if failure.get(source) is not None:
            run[target] = failure[source]


def _failed_run(failure: dict[str, Any]) -> dict[str, Any]:
    run_id = _text(failure.get("run_id"), "unknown-run")
    scenario_id = _text(failure.get("scenario_id"), "unknown")
    version_id = _text(failure.get("application_version_id"), "unknown")
    persona_id = _text(failure.get("persona_id"), "unknown")
    prominence_provider_id = _text(failure.get("prominence_provider_id"), "heuristic")
    return {
        "run_id": run_id,
        "bundle_path": "",
        "integrity_status": "failed",
        "seed": _number(failure.get("seed"), 0),
        "model_trial": _number(failure.get("model_trial"), 0),
        "reproducibility": "not-reproducible",
        "reproducibility_label": "not-reproducible",
        "scenario_id": scenario_id,
        "scenario_label": scenario_id.replace("-", " ").title(),
        "goal": "Goal unavailable",
        "version_id": version_id,
        "version_label": version_id.replace("-", " ").title(),
        "persona_id": persona_id,
        "persona_label": persona_id.replace("-", " ").title(),
        "policy": _text(failure.get("policy"), "unknown"),
        "prominence_provider_id": prominence_provider_id,
        "outcome": _text(failure.get("error_type"), "failed"),
        "verified": False,
        "claimed": False,
        "terminal_state": _text(failure.get("terminal_state"), "failed"),
        "stage": _text(failure.get("stage"), "execution"),
        "terminal_reason": (
            None
            if _text(failure.get("stage"), "execution") == "evaluation"
            else _text(failure.get("reason"), "run failed")
        ),
        "evaluation_failure_reason": (
            _text(failure.get("reason"), "result evaluation failed")
            if _text(failure.get("stage"), "execution") == "evaluation"
            else None
        ),
        "ux_sample_valid": False,
        "ux_sample_invalid_reason": "execution-failure",
        "comparison_valid": False,
        "failure_reason": _text(failure.get("reason"), "run failed"),
        "status_class": "status-untrusted",
        "failed": True,
        "trusted": False,
        "integrity_failures": [],
        "timeline": [],
        "snapshots": [],
        "observations": [],
        "selections": [],
        "prominence": [],
        "saliency": [],
        "saliency_stage_timeline": [],
        "saliency_runtime": {
            "total_inference_ms": 0,
            "durations": [],
            "cache_states": [],
        },
        "saliency_fallbacks": [],
        "scent_records": [],
        "decisions": [],
        "actions": [],
        "verification": {"verified": False, "evidence_ids": [], "details": None},
        "memory": [],
        "manifests": {"provider_manifests": []},
        "model_calls": [],
        "user_effort": {},
        "analysis_cost": {},
        "evidence": [],
        "findings": [],
        "metrics": [],
        "limitations": [
            "Run failed before complete analysis; available safe evidence is shown."
        ],
    }


def _run_stage(
    result: dict[str, Any],
    crash: dict[str, Any],
    outcome: str,
    integrity_failures: list[str],
) -> str:
    if result.get("evaluation_failure_reason"):
        return "evaluation"
    if crash:
        reason = _text(crash.get("reason")).lower()
        return "bundle-finalization" if "finalization" in reason else "execution"
    if integrity_failures:
        return "integrity"
    return "complete" if outcome == "verified-success" else "terminal"


def _status_class(outcome: str, stage: str, terminal_state: str, trusted: bool) -> str:
    if not trusted or terminal_state in {"partial", "crashed", "untrusted"}:
        return "status-untrusted"
    if stage == "evaluation":
        return "status-evaluation"
    if outcome == "verified-success":
        return "status-success"
    if outcome == "timed-out":
        return "status-timeout"
    if outcome in {"provider-failure", "model-failure", "internal-error"}:
        return "status-error"
    return "status-terminal"


def _safe_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("/", "\\u002f")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _safe_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _run_page_names(runs: object) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for run in _list_of_mappings(runs):
        run_id = _text(run.get("run_id"), "run")
        grouped[_safe_filename(run_id) or "run"].append(run_id)
    names: dict[str, str] = {}
    for safe_name, run_ids in grouped.items():
        for run_id in run_ids:
            suffix = (
                f"-{hashlib.sha256(run_id.encode('utf-8')).hexdigest()[:8]}"
                if len(run_ids) > 1
                else ""
            )
            names[run_id] = f"{safe_name}{suffix}.html"
    return names


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, Any], value)


def _list_of_mappings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    items = cast(list[object] | tuple[object, ...], value)
    return [cast(dict[str, Any], item) for item in items if isinstance(item, dict)]


def _kind(event: dict[str, Any]) -> str:
    return _text(event.get("kind"), "unknown")


def _strings(value: object) -> list[str]:
    if not isinstance(value, list | tuple | set | frozenset):
        return []
    items = cast(
        list[object] | tuple[object, ...] | set[object] | frozenset[object], value
    )
    return [_text(item) for item in items if item is not None]


def _safe_string_mapping(value: object) -> dict[str, str]:
    return {
        _safe_identifier(key, "unavailable"): _safe_identifier(item, "unavailable")
        for key, item in _mapping(value).items()
        if key not in _PRIVATE_KEYS
    }


def _number_mapping(value: object) -> dict[str, float]:
    return {
        _text(key): _number(item, 0)
        for key, item in _mapping(value).items()
        if _is_number(item) and key not in _PRIVATE_KEYS
    }


def _bounds(value: object) -> dict[str, float] | None:
    bounds = _mapping(value)
    if not all(_is_number(bounds.get(key)) for key in ("x", "y", "width", "height")):
        return None
    if bounds["width"] <= 0 or bounds["height"] <= 0:
        return None
    return {key: _number(bounds[key], 0) for key in ("x", "y", "width", "height")}


def _bounded(value: object, *, default: float = 0) -> float:
    return min(1.0, max(0.0, _number(value, default)))


def _optional_bounded(value: object) -> float | None:
    return _bounded(value) if _is_number(value) else None


def _number(value: object, default: float) -> float:
    return float(cast(int | float, value)) if _is_number(value) else default


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _first_string(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
        if value is not None and not isinstance(value, (dict, list, tuple)):
            text = str(value)
            if text:
                return text
    return "unknown"


def _first_optional_string(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
        if value is not None and not isinstance(value, (dict, list, tuple)):
            text = str(value)
            if text:
                return text
    return None


def _safe_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list | tuple):
        items = cast(list[object] | tuple[object, ...], value)
        return [_safe_value(item) for item in items]
    if isinstance(value, dict):
        mapping = cast(dict[str, object], cast(object, value))
        result: dict[str, object] = {}
        for text_key, item in mapping.items():
            if text_key not in _PRIVATE_KEYS:
                result[text_key] = _safe_value(item)
        return result
    return _text(value)


def _evidence_class(value: object) -> str:
    candidate = _text(value, "model-estimate")
    if candidate in {
        "deterministic-fact",
        "model-estimate",
        "unsupported-human-claim",
    }:
        return candidate
    return "model-estimate"


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in result:
            result.append(text)
    return result
