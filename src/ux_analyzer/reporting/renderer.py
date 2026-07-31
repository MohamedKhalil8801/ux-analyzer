"""Build self-contained HTML replay pages from immutable run bundles."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

DEFAULT_SINGLE_FILE_THRESHOLD = 2_000_000
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
    full_context = _report_context(experiment)
    single_html = _render_html(full_context, "Attention-guided experiment replay")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(single_html.encode("utf-8")) <= threshold:
        destination.write_text(single_html, encoding="utf-8")
        return destination

    run_directory = destination.parent / f"{destination.stem}-runs"
    run_directory.mkdir(parents=True, exist_ok=True)
    run_links = {
        run["run_id"]: f"{run_directory.name}/{_safe_filename(run['run_id'])}.html"
        for run in experiment["runs"]
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
        (run_directory / f"{_safe_filename(run['run_id'])}.html").write_text(
            _render_html(
                run_context,
                f"Run replay: {run['run_id']}",
            ),
            encoding="utf-8",
        )
    return destination


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
    return {
        "runs": ordered_runs,
        "comparison_rows": _comparison_rows(ordered_runs),
        "gate_rows": _gate_rows(summary, ordered_runs),
        "failure_rows": [run for run in ordered_runs if run["failed"]],
        "evidence_summary": _evidence_summary(ordered_runs),
        "limitations": _unique(
            limitation for run in ordered_runs for limitation in run["limitations"]
        ),
    }


def _run_directories(root: Path) -> tuple[Path, ...]:
    if (root / "manifest.json").is_file() and (root / "timeline.jsonl").is_file():
        return (root,)
    candidates: list[Path] = []
    runs_root = root / "runs"
    if runs_root.is_dir():
        candidates.extend(
            path
            for path in runs_root.iterdir()
            if path.is_dir()
            and (path / "manifest.json").is_file()
            and (path / "timeline.jsonl").is_file()
        )
    staging_root = root / ".staging"
    if staging_root.is_dir():
        candidates.extend(
            path
            for path in staging_root.iterdir()
            if path.is_dir()
            and (path / "manifest.json").is_file()
            and (path / "timeline.jsonl").is_file()
            and (path / "crash.marker").is_file()
        )
    return tuple(sorted(candidates, key=lambda path: path.name))


def _load_run(path: Path) -> dict[str, Any]:
    manifest = _read_object(path / "manifest.json")
    result = _read_object(path / "result.json", required=False)
    crash = _read_object(path / "crash.marker", required=False)
    events = _read_jsonl(path / "timeline.jsonl")
    if not events:
        state = _mapping(result.get("state"))
        events = _list_of_mappings(state.get("events"))
    state = _mapping(result.get("state"))
    spec = _mapping(result.get("spec")) or _mapping(state.get("spec"))
    trusted = _trusted_bundle(path, result, crash, events)
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
    failure_reason = next(
        (
            _text(value)
            for value in (
                result.get("evaluation_failure_reason"),
                result.get("terminal_reason"),
                crash.get("reason"),
                _mapping(result.get("outcome")).get("reason"),
            )
            if value
        ),
        "",
    )
    terminal_state = (
        "crashed"
        if crash
        else "finalized"
        if result or (path / "checksums.sha256").is_file()
        else "partial"
    )
    stage = _run_stage(result, crash, outcome)
    return {
        "run_id": run_id,
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
        "failure_reason": failure_reason,
        "failed": stage != "complete",
        "trusted": trusted,
        "timeline": public_events,
        "snapshots": snapshots,
        "observations": observations,
        "selections": _selections(events),
        "prominence": _prominence(events),
        "scent_records": _scent_records(events),
        "decisions": _decisions(events),
        "actions": _actions(events),
        "verification": verification,
        "memory": _memory(attention),
        "manifests": _manifests(manifest, result, state),
        "model_calls": _model_calls(events),
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
                    "failure_reason",
                    "failed",
                    "trusted",
                    "metrics",
                    "limitations",
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
    return {
        "runs": runs,
        "comparison_rows": experiment["comparison_rows"],
        "gate_rows": experiment["gate_rows"],
        "failure_rows": experiment["failure_rows"],
        "evidence_summary": experiment["evidence_summary"],
        "limitations": experiment["limitations"],
        "initial_viewport_width": initial_width,
        "report_json": _safe_json(
            {
                "runs": runs,
                "comparison_rows": experiment["comparison_rows"],
                "gate_rows": experiment["gate_rows"],
                "failure_rows": experiment["failure_rows"],
                "evidence_summary": experiment["evidence_summary"],
                "limitations": experiment["limitations"],
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    result: dict[str, Any] = {
        "sequence": _number(event.get("sequence"), 0),
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
        result["record"] = _safe_value(event.get("record"))
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
            }
        )
    elif kind == "verification-recorded":
        result["verification"] = _public_verification(event.get("result"))
    elif kind == "run-terminated":
        result["outcome"] = _public_outcome(event.get("outcome"))
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
        record = _safe_value(event.get("record"))
        if isinstance(record, dict):
            records.append(cast(dict[str, Any], record))
    return records


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
    explicit: object = metrics.get("metrics")
    if isinstance(explicit, list):
        for item in _list_of_mappings(cast(object, explicit)):
            if _evidence_class(
                item.get("evidence_class")
            ) != "unsupported-human-claim" and _is_number(item.get("value")):
                rows.append(_metric_row(item.get("name"), item.get("value"), item))
    for name, value in metrics.items():
        if name in _METRIC_IDENTITY_KEYS or name in {"metrics", "evidence"}:
            continue
        if name == "discovery_cost" and isinstance(value, dict):
            discovery_cost = cast(dict[str, Any], value)
            total: object = discovery_cost.get("total")
            if _is_number(total):
                rows.append(_metric_row("discovery-cost", total, discovery_cost))
            continue
        if _is_number(value):
            rows.append(_metric_row(name.replace("_", "-"), value, metrics))
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
    return {
        "name": _text(name, "metric"),
        "value": _number(value, 0),
        "evidence_class": _evidence_class(_mapping(source).get("evidence_class")),
        "evidence_ids": _strings(_mapping(source).get("evidence_ids")),
    }


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
            }
        )
    return rows


def _gate_rows(
    summary: dict[str, Any], runs: tuple[dict[str, Any], ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in _list_of_mappings(summary.get("variant_comparisons")):
        baseline = _mapping(comparison.get("baseline"))
        improved = _mapping(comparison.get("improved"))
        gate = _mapping(comparison.get("gate"))
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
                "passed": bool(gate.get("passed", False)),
                "reasons": _strings(gate.get("reasons")),
            }
        )
    return rows


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
    run["stage"] = _text(failure.get("stage"), run["stage"])
    run["terminal_state"] = _text(failure.get("terminal_state"), run["terminal_state"])
    run["failure_reason"] = _text(failure.get("reason"), run["failure_reason"])
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
        "failure_reason": _text(failure.get("reason"), "run failed"),
        "failed": True,
        "trusted": False,
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


def _run_stage(result: dict[str, Any], crash: dict[str, Any], outcome: str) -> str:
    if result.get("evaluation_failure_reason"):
        return "evaluation"
    if crash:
        reason = _text(crash.get("reason")).lower()
        return "bundle-finalization" if "finalization" in reason else "execution"
    return "complete" if outcome == "verified-success" else "terminal"


def _trusted_bundle(
    path: Path,
    result: dict[str, Any],
    crash: dict[str, Any],
    events: list[dict[str, Any]],
) -> bool:
    return bool(
        result
        and not crash
        and (path / "checksums.sha256").is_file()
        and any(_kind(event) == "run-terminated" for event in events)
    )


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
    candidate = _text(value, "deterministic-fact")
    if candidate in {
        "deterministic-fact",
        "model-estimate",
        "unsupported-human-claim",
    }:
        return candidate
    return "deterministic-fact"


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in result:
            result.append(text)
    return result
