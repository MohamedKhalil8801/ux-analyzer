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
from pathlib import Path
from typing import Any, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

DEFAULT_SINGLE_FILE_THRESHOLD = 2_000_000
_HASH_CHUNK_BYTES = 1024 * 1024
_CHECKSUM_PATTERN = re.compile(r"[0-9a-f]{64}")
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
    }
)


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
    if not root.exists() or not root.is_dir():
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
        "limitations": _unique(
            limitation for run in ordered_runs for limitation in run["limitations"]
        ),
    }


def _run_directories(root: Path) -> tuple[Path, ...]:
    if any(
        (root / name).exists() for name in (*_REQUIRED_BUNDLE_FILES, "crash.marker")
    ):
        return (root,)
    candidates: list[Path] = []
    runs_root = root / "runs"
    if runs_root.is_dir():
        candidates.extend(path for path in runs_root.iterdir() if path.is_dir())
    staging_root = root / ".staging"
    if staging_root.is_dir():
        candidates.extend(
            path
            for path in staging_root.iterdir()
            if path.is_dir()
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
    finalized_candidate = (
        path.parent.name == "runs" or (path / "checksums.sha256").is_file()
    )
    if finalized_candidate:
        integrity_failures.extend(_bundle_integrity_failures(path))
    if result and not any(_kind(event) == "run-terminated" for event in events):
        integrity_failures.append("missing terminal run event")
    integrity_failures = _unique(integrity_failures)
    trusted = bool(result and not crash and not integrity_failures)
    metrics = _extract_metrics(result) if trusted else {}
    snapshots = _snapshots(path, events)
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
    stage = _run_stage(result, crash, outcome, integrity_failures)
    actions = _actions(events)
    model_calls = _model_calls(events)
    user_effort = _user_effort(actions, observations, metrics)
    analysis_cost = _analysis_cost(model_calls)
    return {
        "run_id": run_id,
        "bundle_path": f"{path.parent.name}/{path.name}",
        "integrity_status": "trusted" if trusted else "failed",
        "seed": _number(manifest.get("seed"), 0),
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
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value)


def _read_object_safely(
    path: Path, *, required: bool = True
) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        failure = f"missing required bundle file: {path.name}" if required else ""
        return {}, [failure] if failure else []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, [f"invalid JSON bundle file: {path.name}"]
    if not isinstance(value, dict):
        return {}, [f"bundle file must contain object: {path.name}"]
    return cast(dict[str, Any], value), []


def _read_jsonl_safely(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], [f"missing required bundle file: {path.name}"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return [], [f"unreadable bundle file: {path.name}"]
    events: list[dict[str, Any]] = []
    failures: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            failures.append(f"invalid timeline JSON at line {line_number}")
            continue
        if not isinstance(value, dict):
            failures.append(f"timeline entry is not object at line {line_number}")
            continue
        events.append(cast(dict[str, Any], value))
    return events, failures


def _bundle_integrity_failures(path: Path) -> list[str]:
    failures: list[str] = []
    checksum_path = path / "checksums.sha256"
    if not checksum_path.is_file():
        return ["missing checksum file: checksums.sha256"]
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return ["unreadable checksum file: checksums.sha256"]

    checksums: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or _CHECKSUM_PATTERN.fullmatch(parts[0]) is None:
            failures.append(f"invalid checksum entry at line {line_number}")
            continue
        digest, relative_path = parts
        relative = Path(relative_path)
        if (
            not relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_path == "checksums.sha256"
        ):
            failures.append(f"invalid checksum path at line {line_number}")
            continue
        normalized = relative.as_posix()
        if normalized in checksums:
            failures.append(f"duplicate checksum entry: {normalized}")
            continue
        checksums[normalized] = digest

    actual_files = {
        candidate.relative_to(path).as_posix()
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.name not in {"checksums.sha256", ".active"}
    }
    for required in sorted(_REQUIRED_BUNDLE_FILES):
        if required not in actual_files:
            failures.append(f"missing required bundle file: {required}")
        elif required not in checksums:
            failures.append(f"required file missing checksum: {required}")
    for relative_path in sorted(actual_files - checksums.keys()):
        failures.append(f"checksum entry missing: {relative_path}")
    for relative_path in sorted(checksums.keys() - actual_files):
        failures.append(f"checksummed file missing: {relative_path}")
    for relative_path in sorted(actual_files & checksums.keys()):
        digest = _file_sha256(path / relative_path)
        if digest != checksums[relative_path]:
            failures.append(f"checksum mismatch: {relative_path}")
    return _unique(failures)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshots(path: Path, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for event in events:
        if _kind(event) != "viewport-captured":
            continue
        snapshot = _mapping(event.get("snapshot"))
        if not snapshot:
            continue
        public = _public_snapshot(path, snapshot, event)
        if public["id"] and all(item["id"] != public["id"] for item in snapshots):
            snapshots.append(public)
    return snapshots


def _public_snapshot(
    run_path: Path,
    snapshot: dict[str, Any],
    event: dict[str, Any],
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
    viewport = _viewport(snapshot, event, elements)
    return {
        "id": _text(snapshot.get("id")),
        "viewport": viewport,
        "screenshot": _artifact_data_uri(run_path, artifact),
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


def _artifact_data_uri(run_path: Path, artifact: str | None) -> str | None:
    if not artifact:
        return None
    relative = Path(artifact)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = (run_path / relative).resolve()
    try:
        candidate.relative_to(run_path.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    content = candidate.read_bytes()
    mime = mimetypes.guess_type(candidate.name)[0] or _image_mime(content)
    return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"


def _image_mime(content: bytes) -> str:
    if content.startswith(b"\x89PNG"):
        return "image/png"
    if content.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if content.startswith(b"GIF8"):
        return "image/gif"
    return "application/octet-stream"


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
        "model": _first_string(record.get("model"), record.get("model_id")),
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
        "model_ids": _string_mapping(manifest.get("model_ids")),
        "prompt_versions": _string_mapping(manifest.get("prompt_versions")),
        "provider_versions": _string_mapping(manifest.get("provider_versions")),
        "provider_manifests": [
            {
                "provider_id": _text(item.get("provider_id")),
                "role": _text(item.get("role")),
                "model_id": _optional_text(item.get("model_id")),
                "endpoint_origin": _optional_text(item.get("endpoint_origin")),
                "version": _text(item.get("version")),
                "prompt_version": _optional_text(item.get("prompt_version")),
                "schema_version": _optional_text(item.get("schema_version")),
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
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if not run["trusted"] or not run["metrics"]:
            continue
        groups[
            (
                run["scenario_id"],
                run["persona_id"],
                run["policy"],
                run["version_id"],
            )
        ].append(run)
    rows: list[dict[str, Any]] = []
    for identity in sorted(groups):
        scenario_id, persona_id, policy, version_id = identity
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


def _run_overview_rows(
    runs: tuple[dict[str, Any], ...], gate_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        gate = _gate_for_run(run, gate_rows)
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
                "user_actions": len(run["actions"]),
                "observations": len(run["observations"]),
                "discovery_cost": _format_overview_cost(
                    _median_metric([run], "discovery-cost")
                ),
                "estimated_task_seconds": _format_seconds(
                    _number(run["user_effort"].get("estimated_task_seconds"), 0)
                ),
                "model_calls": int(_number(run["analysis_cost"].get("model_calls"), 0)),
                "analysis_latency": _format_milliseconds(
                    _number(run["analysis_cost"].get("latency_ms"), 0)
                ),
                "analysis_tokens": int(
                    _number(run["analysis_cost"].get("total_tokens"), 0)
                ),
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
        matching = next(
            (
                run
                for run in runs
                if run["scenario_id"] == scenario_id
                and run["persona_id"] == persona_id
                and run["policy"] == policy
                and run["trusted"]
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
    existing = {(row["scenario_id"], row["persona_id"], row["policy"]) for row in rows}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run["trusted"] and run["ux_sample_valid"] and run["metrics"]:
            groups[(run["scenario_id"], run["persona_id"], run["policy"])].append(run)
    for identity in sorted(groups):
        if identity in existing:
            continue
        derived = _derived_gate_row(groups[identity])
        if derived is not None:
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
    checks = (
        (
            "paired median discovery cost did not decrease",
            _median_values(cost_deltas) < 0,
        ),
        (
            "paired median wrong-action burden increased",
            _median_values(wrong_deltas) <= 0,
        ),
        (
            "paired median backtrack burden increased",
            _median_values(backtrack_deltas) <= 0,
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
        ]
        if not matching or any(not run["trusted"] for run in matching):
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
        "seed",
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
        ("seed", "seed"),
    ):
        if failure.get(source) is not None:
            run[target] = failure[source]


def _failed_run(failure: dict[str, Any]) -> dict[str, Any]:
    run_id = _text(failure.get("run_id"), "unknown-run")
    scenario_id = _text(failure.get("scenario_id"), "unknown")
    version_id = _text(failure.get("application_version_id"), "unknown")
    persona_id = _text(failure.get("persona_id"), "unknown")
    return {
        "run_id": run_id,
        "seed": _number(failure.get("seed"), 0),
        "scenario_id": scenario_id,
        "scenario_label": scenario_id.replace("-", " ").title(),
        "goal": "Goal unavailable",
        "version_id": version_id,
        "version_label": version_id.replace("-", " ").title(),
        "persona_id": persona_id,
        "persona_label": persona_id.replace("-", " ").title(),
        "policy": _text(failure.get("policy"), "unknown"),
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
        "scent_records": [],
        "decisions": [],
        "actions": [],
        "verification": {"verified": False, "evidence_ids": [], "details": None},
        "memory": [],
        "manifests": {"provider_manifests": []},
        "model_calls": [],
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


def _string_mapping(value: object) -> dict[str, str]:
    return {
        _text(key): _text(item)
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
