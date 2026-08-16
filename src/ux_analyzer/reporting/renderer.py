"""Build self-contained HTML replay pages from immutable run bundles."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import mimetypes
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast
from uuid import uuid4

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from PIL import Image

from ux_analyzer.application.checkpoint import finalized_bundle_failures
from ux_analyzer.application.evaluation import (
    persisted_comparison_sample_is_valid,
)
from ux_analyzer.application.evidence_corpus import (
    EvidenceCorpus,
    EvidenceEntry,
    validate_evidence_refs,
)
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.synthesis import EvidenceRef, SynthesisAttempt, SynthesisStatus
from ux_analyzer.ports.artifacts import (
    validate_saliency_artifact_path,
    validate_timeline_event_order,
)
from ux_analyzer.storage.run_bundle import (
    SecureDirectoryHandle,
    secure_assert_ancestors,
    secure_create_exclusive_file,
    secure_is_link_or_reparse,
    secure_open_directory,
    secure_read_bytes,
    secure_replace_exclusive_file,
    validate_saliency_heatmap_content,
    validate_saliency_native_map_content,
)
from ux_analyzer.storage.saliency_replay import (
    load_saliency_replay as _load_trusted_saliency_replay,
)
from ux_analyzer.storage.saliency_replay import (
    merge_saliency_event as _merge_trusted_saliency_event,
)
from ux_analyzer.storage.saliency_replay import (
    saliency_artifact_data_uri as _trusted_saliency_artifact_data_uri,
)
from ux_analyzer.storage.synthesis_artifacts import (
    MAX_SYNTHESIS_JSON_BYTES,
    SynthesisArtifactError,
    SynthesisArtifactStore,
    validate_publishable_synthesis_attempt,
)

DEFAULT_SINGLE_FILE_THRESHOLD = 4_000_000
_MAX_REPORT_JSON_BYTES = 8 * 1024 * 1024
_MAX_REPORT_TIMELINE_BYTES = 16 * 1024 * 1024
_MAX_SOURCE_SCREENSHOT_BYTES = 16 * 1024 * 1024
_MAX_SYNTHESIS_HEATMAP_BYTES = 8 * 1024 * 1024
_MAX_SYNTHESIS_NATIVE_MAP_BYTES = 64 * 1024 * 1024
_MAX_BUNDLE_CHECKSUM_BYTES = 16 * 1024 * 1024
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
_SYNTHESIS_SELECTED_STATUSES = frozenset(
    {SynthesisStatus.ACCEPTED, SynthesisStatus.NO_ISSUES}
)
_SYNTHESIS_RUN_IDENTITY_FIELDS = (
    "run_id",
    "seed",
    "model_trial",
    "config_digest",
    "scenario_id",
    "application_version_id",
    "persona_id",
    "policy",
    "prominence_provider_id",
)
_FALLBACK_SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}
_SYNTHESIS_ARTIFACT_FILES = ("index.json", "synthesis.json", "corpus-manifest.json")


def _json_object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains duplicate fields")
        result[key] = value
    return result


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
_CANONICAL_PROVIDER_IDS = {
    "heuristic": "heuristic",
    "heuristic-prominence": "heuristic",
    "foveacast": "foveacast",
    "foveacast-prominence": "foveacast",
}


def load_saliency_replay(
    run_path: Path,
    events: Sequence[Mapping[str, object]],
    snapshots: Sequence[Mapping[str, object]],
    *,
    expected_provider_id: str = "unavailable",
) -> tuple[dict[str, Any], ...]:
    """Load trusted saliency replay data for bounded evidence consumers."""

    return _load_trusted_saliency_replay(
        Path(run_path),
        [dict(event) for event in events],
        [dict(snapshot) for snapshot in snapshots],
        expected_provider_id=expected_provider_id,
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
    with secure_open_directory(
        destination.parent,
        "report output directory",
        create=True,
    ) as output_parent:
        if _estimated_full_report_bytes(experiment) <= threshold:
            full_context = _report_context(experiment)
            single_html = _render_html(
                full_context,
                "Attention-guided experiment replay",
            )
            if len(single_html.encode("utf-8")) <= threshold:
                _publish_report_text(output_parent, destination.name, single_html)
                return destination

        run_directory = destination.parent / f"{destination.stem}-runs"
        with secure_open_directory(
            run_directory,
            "report run-page directory",
            create=True,
        ) as run_parent:
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
            _publish_report_text(
                output_parent,
                destination.name,
                _render_html(index_context, "Attention-guided experiment replay"),
            )
            for run in experiment["runs"]:
                run_context = _report_context(
                    {**experiment, "runs": [run]},
                    run_links={run["run_id"]: ""},
                    run_scope=frozenset({run["run_id"]}),
                )
                run_html = _render_html(
                    run_context,
                    f"Run replay: {run['run_id']}",
                )
                if len(run_html.encode("utf-8")) > threshold:
                    run_html = _oversized_run_html(run["run_id"], threshold)
                _publish_report_text(
                    run_parent,
                    run_page_names[run["run_id"]],
                    run_html,
                )
    return destination


def _publish_report_text(
    parent: SecureDirectoryHandle,
    destination_name: str,
    content: str,
) -> None:
    temporary_name = f".{destination_name}.{uuid4().hex}.tmp"
    encoded = content.encode("utf-8")
    with secure_create_exclusive_file(
        parent,
        temporary_name,
        "report temporary file",
    ) as temporary:
        offset = 0
        while offset < len(encoded):
            written = os.write(temporary.descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("failed to write report output")
            offset += written
        os.fsync(temporary.descriptor)
        secure_replace_exclusive_file(
            parent,
            temporary,
            destination_name,
            "report publication",
            replace_existing=True,
        )


def _estimated_full_report_bytes(experiment: dict[str, Any]) -> int:
    template_root = Path(__file__).parent
    shell_bytes = sum(
        path.stat().st_size
        for path in (
            template_root / "templates" / "experiment.html.j2",
            template_root / "static" / "report.css",
            template_root / "static" / "report.js",
            template_root / "static" / "report-index.js",
        )
    )
    synthesis_bytes = experiment.get("_synthesis_artifact_bytes", 0)
    if not isinstance(synthesis_bytes, int) or synthesis_bytes < 0:
        synthesis_bytes = 0
    return shell_bytes + len(_safe_json(experiment).encode("utf-8")) + synthesis_bytes


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
    synthesis, synthesis_artifact_bytes = _load_synthesis(root, ordered_runs)
    return {
        "runs": ordered_runs,
        "run_rows": _run_overview_rows(ordered_runs, gate_rows),
        "comparison_rows": _comparison_rows(ordered_runs),
        "provider_comparisons": _provider_comparisons(ordered_runs),
        "gate_rows": gate_rows,
        "failure_rows": [run for run in ordered_runs if run["failed"]],
        "evidence_summary": _evidence_summary(ordered_runs),
        "focused_acceptance": _focused_acceptance(summary),
        "limitations": _unique(
            limitation for run in ordered_runs for limitation in run["limitations"]
        ),
        "synthesis": synthesis,
        "_synthesis_artifact_bytes": synthesis_artifact_bytes,
    }


def _load_synthesis(
    root: Path,
    runs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], int]:
    fallback_findings = _deterministic_fallback_findings(runs)
    index_path = root / "synthesis" / "index.json"
    if secure_is_link_or_reparse(index_path):
        return (
            _fallback_synthesis(
                "invalid",
                runs,
                fallback_findings,
                "Persisted report synthesis failed deterministic validation.",
            ),
            _synthesis_artifact_bytes(root, None),
        )
    if not index_path.is_file():
        return (
            _fallback_synthesis(
                "missing",
                runs,
                fallback_findings,
                "No persisted report synthesis is available.",
            ),
            0,
        )

    try:
        store = SynthesisArtifactStore(root)
        attempt = store.report_attempt
        selected_id = (
            attempt.attempt_id
            if attempt is not None and attempt.status in _SYNTHESIS_SELECTED_STATUSES
            else None
        )
        artifact_bytes = _synthesis_artifact_bytes(root, selected_id)
        if attempt is None:
            return (
                _fallback_synthesis(
                    "unavailable",
                    runs,
                    fallback_findings,
                    "Only rejected or unavailable synthesis attempts were published.",
                ),
                artifact_bytes,
            )
        if attempt.status in {SynthesisStatus.REJECTED, SynthesisStatus.UNAVAILABLE}:
            limitation = (
                attempt.limitations[-1]
                if attempt.limitations
                else "Only rejected or unavailable synthesis attempts were published."
            )
            return (
                _fallback_synthesis(
                    _synthesis_enum_text(attempt.status),
                    runs,
                    fallback_findings,
                    limitation,
                ),
                artifact_bytes,
            )
        if attempt.status not in _SYNTHESIS_SELECTED_STATUSES:
            raise SynthesisArtifactError(
                "selected synthesis attempt has an ineligible status"
            )
        validate_publishable_synthesis_attempt(attempt)
        corpus = _load_synthesis_corpus(root, attempt)
        for objection in attempt.objections:
            validate_evidence_refs(corpus, objection.resolution_evidence_refs)
        _validate_synthesis_run_scope(corpus, runs)
        findings = _synthesis_findings(attempt, corpus, runs, root)
        status = _synthesis_enum_text(attempt.status)
        assessment = (
            "No supported UX issues were established in the tested scenarios."
            if attempt.status is SynthesisStatus.NO_ISSUES
            else (
                f"{len(findings)} evidence-grounded finding"
                f"{'s' if len(findings) != 1 else ''} passed independent review. "
                f"Start with: {findings[0]['title']}"
            )
        )
        return (
            {
                "synthesis_status": status,
                "status": status,
                "using_fallback": False,
                "attempt_id": attempt.attempt_id,
                "corpus_digest": attempt.corpus_digest,
                "assessment": assessment,
                "findings": findings,
                "fallback_findings": fallback_findings,
                "limitations": list(attempt.limitations),
                "tested_scope": _synthesis_scope(runs),
            },
            artifact_bytes,
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
        SynthesisArtifactError,
    ):
        return (
            _fallback_synthesis(
                "invalid",
                runs,
                fallback_findings,
                "Persisted report synthesis failed deterministic validation.",
            ),
            _synthesis_artifact_bytes(root, None),
        )


def _fallback_synthesis(
    status: str,
    runs: Sequence[Mapping[str, Any]],
    fallback_findings: list[dict[str, Any]],
    limitation: str,
) -> dict[str, Any]:
    publishable_findings = [
        finding
        for finding in fallback_findings
        if any(
            bool(reference.get("available"))
            for reference in _list_of_mappings(finding.get("evidence_refs"))
        )
    ]
    omitted_count = len(fallback_findings) - len(publishable_findings)
    boundary_rejection = (
        status == "rejected" and "evidence boundary" in limitation.casefold()
    )
    if status in {"missing", "unavailable"}:
        assessment = (
            "Model review is unavailable. "
            f"{len(publishable_findings)} recorded signal"
            f"{'s are' if len(publishable_findings) != 1 else ' is'} linked directly "
            "to evidence; verify each replay before changing the UI."
        )
        model_review_status = "unavailable"
    elif status == "invalid":
        assessment = (
            "Model review could not be validated. "
            f"{len(publishable_findings)} recorded signal"
            f"{'s are' if len(publishable_findings) != 1 else ' is'} linked directly "
            "to evidence; verify each replay before changing the UI."
        )
        model_review_status = "invalid"
    elif boundary_rejection:
        assessment = (
            "Model review was rejected at the bounded evidence boundary. "
            f"{len(publishable_findings)} recorded signal"
            f"{'s remain' if len(publishable_findings) != 1 else ' remains'} linked "
            "directly to evidence for manual review."
        )
        model_review_status = "rejected"
    else:
        assessment = (
            "Model review was rejected. "
            f"{len(publishable_findings)} recorded signal"
            f"{'s remain' if len(publishable_findings) != 1 else ' remains'} linked "
            "directly to evidence for manual review."
        )
        model_review_status = "rejected"
    limitations = [limitation]
    if omitted_count:
        limitations.append(
            f"{omitted_count} recorded signal"
            f"{'s were' if omitted_count != 1 else ' was'} omitted because no "
            "canonical evidence target could be resolved."
        )
    return {
        "synthesis_status": status,
        "status": status,
        "using_fallback": True,
        "attempt_id": None,
        "corpus_digest": None,
        "assessment": assessment,
        "findings": publishable_findings,
        "fallback_findings": publishable_findings,
        "limitations": limitations,
        "tested_scope": _synthesis_scope(runs),
        "model_review_status": model_review_status,
        "review_basis": "recorded-deterministic-evidence",
        "missing_review_fields": [
            "UX principles",
            "counterevidence",
            "reviewer status",
        ],
    }


def _deterministic_fallback_findings(
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for run in runs:
        run_id = _text(run.get("run_id"))
        for finding in _list_of_mappings(run.get("findings")):
            evidence_refs, evidence_targets = _fallback_evidence_targets(
                finding,
                run,
                run_id,
            )
            fallback_context = _fallback_finding_context(finding, run, run_id)
            fallback_copy = _fallback_finding_copy(finding, fallback_context)
            findings.append(
                {
                    **finding,
                    "title": fallback_copy["title"],
                    "run_ids": _unique([*_strings(finding.get("run_ids")), run_id]),
                    "source": "deterministic-fallback",
                    "evidence_refs": evidence_refs,
                    "evidence_targets": evidence_targets,
                    "fallback_context": fallback_context,
                    "fallback_title": fallback_copy["title"],
                    "fallback_issue": fallback_copy["issue"],
                    "fallback_impact": fallback_copy["impact"],
                    "fallback_root_cause": fallback_copy["root_cause"],
                    "fallback_fix": fallback_copy["fix"],
                }
            )
    return sorted(
        findings,
        key=lambda item: (
            _FALLBACK_SEVERITY_ORDER.get(
                _text(item.get("severity"), "").lower(),
                len(_FALLBACK_SEVERITY_ORDER),
            ),
            _text(item.get("finding_id"), ""),
        ),
    )


def _fallback_finding_context(
    finding: Mapping[str, Any],
    run: Mapping[str, Any],
    run_id: str,
) -> dict[str, str]:
    actions = _comparison_action_path(dict(run))
    action_values = [
        action_text
        for action in actions
        if (
            action_text := _fallback_action_text(
                {
                    **action,
                    "element_label": _fallback_element_label(
                        run,
                        _text(action.get("element_id")),
                    ),
                }
            )
        )
    ]
    if not action_values:
        action_values = _strings(finding.get("action_sequence"))
    target_ids = _strings(finding.get("element_ids"))
    if not target_ids:
        target_ids = [
            _text(action.get("element_id"))
            for action in actions
            if action.get("element_id")
        ]
    target_id = target_ids[0] if target_ids else ""
    target = _fallback_element_label(run, target_id)
    return {
        "run_id": run_id,
        "scenario": _text(run.get("scenario_label"), "Scenario unavailable"),
        "persona": _text(run.get("persona_label"), "Persona unavailable"),
        "goal": _text(run.get("goal"), "Goal unavailable"),
        "version": _text(run.get("version_label"), "Version unavailable"),
        "target": target,
        "action": "; ".join(action_values) or "No action context recorded",
        "outcome": _text(run.get("outcome"), "Outcome unavailable"),
        "verification": "verified" if bool(run.get("verified")) else "not verified",
    }


def _fallback_action_text(action: Mapping[str, Any]) -> str:
    kind = _text(action.get("kind"), "action")
    element_label = _optional_text(action.get("element_label"))
    label = f"{kind} on {element_label}" if element_label else kind
    succeeded = action.get("succeeded")
    if succeeded is True:
        return f"{label}: succeeded"
    if succeeded is False:
        return f"{label}: failed"
    return label


def _fallback_element_label(run: Mapping[str, Any], element_id: str) -> str:
    if element_id:
        for snapshot in _list_of_mappings(run.get("snapshots")):
            for element in _list_of_mappings(snapshot.get("elements")):
                if _text(element.get("id")) == element_id:
                    label = _text(
                        element.get("label"), "recorded interaction target"
                    )
                    if "<" not in label and ">" not in label:
                        return label
                    return (
                        element_id
                        if len(element_id) <= 64
                        else "recorded interaction target"
                    )
        for action in _comparison_action_path(dict(run)):
            if _text(action.get("element_id")) == element_id:
                return _text(
                    action.get("element_label"),
                    "recorded interaction target",
                )
    return "recorded interaction target"


def _fallback_finding_copy(
    finding: Mapping[str, Any],
    context: Mapping[str, str],
) -> dict[str, str]:
    category = _text(finding.get("category")).casefold()
    target = context["target"]
    scenario = context["scenario"]
    persona = context["persona"]
    goal = context["goal"]
    cause = _text(finding.get("cause"), "Recorded cause unavailable")
    if category == "weak-scent":
        title = f"{target} does not clearly signal the task goal"
        issue = (
            f"The {scenario} scenario recorded weak goal cues on "
            f'"{target}" while {persona} worked toward "{goal}".'
        )
        impact = (
            f"The recorded interaction gives {persona} less information about where "
            f'to start the task "{goal}".'
        )
        fix = (
            f'Inspect "{target}" in the linked replay and make its label or nearby '
            f'cue name the goal "{goal}".'
        )
    elif category == "missing-feedback":
        title = f"{target} does not confirm the completed result"
        issue = (
            f'After the recorded action on "{target}", no visible '
            f'confirmation for "{goal}" was captured.'
        )
        impact = (
            f"{persona} cannot tell from the recorded result whether the task "
            f'"{goal}" completed.'
        )
        fix = (
            f'Inspect the post-action replay for "{target}" and add a visible, '
            f'goal-specific confirmation for "{goal}".'
        )
    elif category == "poor-recovery":
        title = f"{target} recovery does not restore the task"
        issue = (
            f'The recorded task hit an error or failed step around "{target}" '
            f'without restoring progress toward "{goal}".'
        )
        impact = (
            f"{persona} is left without a recorded path back to the task after the "
            f"failure, so the outcome is harder to trust."
        )
        fix = (
            f'Inspect the failed action and replay sequence for "{target}"; provide '
            f'a recovery path that restores progress toward "{goal}".'
        )
    else:
        category_label = category or "unclassified"
        title = f"Recorded interaction needs review for {target}"
        issue = (
            f"The {scenario} scenario recorded category "
            f'"{category_label}" for "{target}" while pursuing "{goal}".'
        )
        impact = (
            f"The recorded result may affect {persona} during the tested task, but "
            "the fallback evidence does not establish a broader user claim."
        )
        fix = f'Inspect the linked evidence for "{target}" before making a product change.'
    return {
        "title": title,
        "issue": issue
        + " Model review is unavailable, so this statement uses recorded fallback evidence only.",
        "impact": impact,
        "root_cause": cause,
        "fix": fix,
    }


def _fallback_evidence_targets(
    finding: Mapping[str, Any],
    run: Mapping[str, Any],
    run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    refs: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for evidence_id in _strings(finding.get("evidence_ids")):
        target = _fallback_evidence_target(evidence_id, run, run_id)
        if target is None:
            target = {
                "kind": "unavailable",
                "run_id": run_id,
                "reason": "Recorded evidence target unavailable.",
            }
            refs.append(
                {
                    "evidence_id": evidence_id,
                    "available": False,
                    "target": target,
                }
            )
        else:
            refs.append(
                {
                    "evidence_id": evidence_id,
                    "available": True,
                    "target": target,
                }
            )
        targets.append(target)
    return refs, targets


def _fallback_evidence_target(
    evidence_id: str,
    run: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any] | None:
    prefix = f"{run_id}:"
    if evidence_id.startswith(prefix):
        metric_id = evidence_id.removeprefix(prefix)
        if any(
            row.get("name") == metric_id
            for row in _list_of_mappings(run.get("metrics"))
        ):
            return {
                "kind": "metric",
                "run_id": run_id,
                "metric_id": metric_id,
            }

    parts = evidence_id.split(":")
    if len(parts) == 3 and parts[0] in {"event", "replay"}:
        evidence_run_id = parts[1]
        sequence = _integer(parts[2])
        if evidence_run_id != run_id or sequence is None:
            return None
        event = _event_by_sequence(run, sequence)
        if event is None:
            return None
        target: dict[str, Any] = {
            "kind": parts[0],
            "run_id": run_id,
            "sequence": sequence,
            "event_id": _text(event.get("event_id"), f"event-{sequence}"),
        }
        viewport_id = _recorded_event_viewport_id(run, event)
        if viewport_id is not None:
            target["viewport_id"] = viewport_id
        element_id = _recorded_event_element_id(event)
        if element_id is not None:
            target["element_id"] = element_id
        return target
    return None


def _synthesis_scope(runs: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    return {
        "run_ids": _unique(_text(run.get("run_id")) for run in runs),
        "scenario_ids": _unique(_text(run.get("scenario_id")) for run in runs),
        "version_ids": _unique(_text(run.get("version_id")) for run in runs),
        "persona_ids": _unique(_text(run.get("persona_id")) for run in runs),
    }


def _validate_synthesis_run_scope(
    corpus: EvidenceCorpus,
    runs: Sequence[Mapping[str, Any]],
) -> None:
    metadata = corpus.metadata
    expected_ids = tuple(_text(run.get("run_id")) for run in runs)
    expected_id_set = frozenset(expected_ids)
    if not expected_ids or len(expected_id_set) != len(expected_ids):
        raise SynthesisArtifactError("report runs have invalid synthesis scope")

    persisted_ids = _synthesis_scope_run_ids(metadata.get("experiment_run_ids"))
    if frozenset(persisted_ids) != expected_id_set:
        raise SynthesisArtifactError("synthesis run IDs do not match report scope")

    raw_identities = metadata.get("experiment_run_identities")
    if not isinstance(raw_identities, (list, tuple)):
        raise SynthesisArtifactError("synthesis run identities are missing")
    persisted_identities = tuple(
        _synthesis_run_identity(identity)
        for identity in cast(Sequence[object], raw_identities)
    )
    expected_identities = tuple(_report_run_identity(run) for run in runs)
    if len(set(persisted_identities)) != len(persisted_identities) or set(
        persisted_identities
    ) != set(expected_identities):
        raise SynthesisArtifactError(
            "synthesis run identities do not match report scope"
        )

    persisted_checksums = _synthesis_bundle_checksums(
        metadata.get("finalized_bundle_checksums")
    )
    current_checksums: dict[str, str] = {}
    for run in runs:
        run_id = _text(run.get("run_id"))
        bundle_path = _text(run.get("bundle_path"))
        if not bundle_path:
            continue
        checksum_bytes = secure_read_bytes(
            corpus.output_root / bundle_path / "checksums.sha256",
            f"run {run_id} checksum manifest",
            max_bytes=_MAX_BUNDLE_CHECKSUM_BYTES,
        )
        current_checksums[run_id] = hashlib.sha256(checksum_bytes).hexdigest()
    if persisted_checksums != current_checksums:
        raise SynthesisArtifactError(
            "synthesis finalized bundle checksums do not match report scope"
        )


def _synthesis_scope_run_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise SynthesisArtifactError("synthesis run IDs are missing")
    raw_run_ids = cast(Sequence[object], value)
    run_ids = tuple(item for item in raw_run_ids if isinstance(item, str) and item)
    if len(run_ids) != len(raw_run_ids) or len(set(run_ids)) != len(run_ids):
        raise SynthesisArtifactError("synthesis run IDs are invalid")
    return run_ids


def _synthesis_bundle_checksums(value: object) -> dict[str, str]:
    if not isinstance(value, (list, tuple)):
        raise SynthesisArtifactError("synthesis finalized bundle checksums are missing")
    checksums: dict[str, str] = {}
    for raw_entry in cast(Sequence[object], value):
        if not isinstance(raw_entry, Mapping):
            raise SynthesisArtifactError(
                "synthesis finalized bundle checksum is invalid"
            )
        entry = cast(Mapping[str, object], raw_entry)
        run_id = entry.get("run_id")
        digest = entry.get("checksums_sha256")
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id in checksums
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise SynthesisArtifactError(
                "synthesis finalized bundle checksum is invalid"
            )
        checksums[run_id] = digest
    return checksums


def _synthesis_run_identity(value: object) -> tuple[object, ...]:
    if not isinstance(value, Mapping):
        raise SynthesisArtifactError("synthesis run identity is invalid")
    identity = cast(Mapping[str, object], value)
    values: list[object] = []
    for name in _SYNTHESIS_RUN_IDENTITY_FIELDS:
        item = identity.get(name)
        if name in {"seed", "model_trial"}:
            if isinstance(item, bool) or not isinstance(item, int):
                raise SynthesisArtifactError("synthesis run identity is invalid")
        elif not isinstance(item, str) or not item:
            raise SynthesisArtifactError("synthesis run identity is invalid")
        values.append(item)
    provider = _canonical_provider_id(values[-1])
    if provider is None:
        raise SynthesisArtifactError("synthesis run identity is invalid")
    values[-1] = provider
    return tuple(values)


def _report_run_identity(run: Mapping[str, Any]) -> tuple[object, ...]:
    identity = {
        "run_id": run.get("run_id"),
        "seed": run.get("seed"),
        "model_trial": run.get("model_trial"),
        "config_digest": run.get("config_digest"),
        "scenario_id": run.get("scenario_id"),
        "application_version_id": run.get("version_id"),
        "persona_id": run.get("persona_id"),
        "policy": run.get("policy"),
        "prominence_provider_id": run.get("prominence_provider_id"),
    }
    return _synthesis_run_identity(identity)


def _synthesis_artifact_bytes(root: Path, attempt_id: str | None) -> int:
    paths = [root / "synthesis" / "index.json"]
    if attempt_id is not None:
        attempt_root = root / "synthesis" / "attempts" / attempt_id
        paths.extend(attempt_root / name for name in _SYNTHESIS_ARTIFACT_FILES[1:])
    total = 0
    for path in paths:
        if not path.is_file() or secure_is_link_or_reparse(path):
            continue
        try:
            total += len(
                secure_read_bytes(
                    path,
                    "synthesis artifact",
                    max_bytes=MAX_SYNTHESIS_JSON_BYTES,
                )
            )
        except (OSError, RuntimeError, ValueError):
            continue
    return total


def _load_synthesis_corpus(root: Path, attempt: SynthesisAttempt) -> EvidenceCorpus:
    attempt_root = root / "synthesis" / "attempts" / attempt.attempt_id
    manifest_path = attempt_root / "corpus-manifest.json"
    raw = secure_read_bytes(
        manifest_path,
        "synthesis corpus manifest",
        max_bytes=MAX_SYNTHESIS_JSON_BYTES,
    )
    if hashlib.sha256(raw).hexdigest() != attempt.corpus_digest:
        raise SynthesisArtifactError("synthesis corpus digest mismatch")
    value = json.loads(
        raw.decode("ascii"), object_pairs_hook=_json_object_without_duplicates
    )
    manifest = _mapping(value)
    if manifest.get("schema_version") != "evidence-corpus-v1":
        raise SynthesisArtifactError("unsupported synthesis corpus schema")
    entries: list[EvidenceEntry] = []
    for raw_entry in _list_of_mappings(manifest.get("entries")):
        raw_ref = {
            name: raw_entry.get(name)
            for name in (
                "evidence_id",
                "kind",
                "run_id",
                "viewport_id",
                "element_id",
                "event_id",
                "metric_id",
                "artifact_path",
                "replay_sequence",
                "sha256",
            )
        }
        replay_sequence = raw_ref["replay_sequence"]
        if replay_sequence is not None and (
            isinstance(replay_sequence, bool) or not isinstance(replay_sequence, int)
        ):
            raise SynthesisArtifactError("synthesis replay sequence is invalid")
        attachment_value = raw_entry.get("attachment_path")
        if attachment_value is not None and not isinstance(attachment_value, str):
            raise SynthesisArtifactError("synthesis attachment path is invalid")
        artifact_value = raw_ref["artifact_path"]
        if artifact_value is not None:
            _synthesis_relative_path(artifact_value, "synthesis artifact path")
        if attachment_value is not None:
            attachment_path = _synthesis_relative_path(
                attachment_value, "synthesis attachment path"
            )
            if artifact_value is None or attachment_path != _synthesis_relative_path(
                artifact_value, "synthesis artifact path"
            ):
                raise SynthesisArtifactError(
                    "synthesis attachment path does not match artifact path"
                )
            attachment = Path(attachment_path)
        else:
            attachment = None
        payload = raw_entry.get("payload")
        if not isinstance(payload, Mapping):
            raise SynthesisArtifactError("synthesis corpus payload is invalid")
        try:
            entries.append(
                EvidenceEntry(
                    ref=EvidenceRef(
                        evidence_id=_text(raw_ref.get("evidence_id")),
                        kind=_text(raw_ref.get("kind")),
                        run_id=_text(raw_ref.get("run_id")),
                        viewport_id=_optional_text(raw_ref.get("viewport_id")),
                        element_id=_optional_text(raw_ref.get("element_id")),
                        event_id=_optional_text(raw_ref.get("event_id")),
                        metric_id=_optional_text(raw_ref.get("metric_id")),
                        artifact_path=_optional_text(raw_ref.get("artifact_path")),
                        replay_sequence=replay_sequence,
                        sha256=_optional_text(raw_ref.get("sha256")),
                    ),
                    evidence_class=EvidenceClass(raw_entry.get("evidence_class", "")),
                    summary=_text(raw_entry.get("summary")),
                    payload=cast(Mapping[str, object], payload),
                    attachment_path=attachment,
                )
            )
        except (TypeError, ValueError) as error:
            raise SynthesisArtifactError("invalid synthesis corpus entry") from error
    try:
        corpus = EvidenceCorpus(
            output_root=root,
            entries=tuple(entries),
            principle_pack_version=_text(
                manifest.get("principle_pack_version"), "unavailable"
            ),
            principle_pack_digest=_text(
                manifest.get("principle_pack_digest"), "unavailable"
            ),
            metadata=cast(Mapping[str, object], _mapping(manifest.get("metadata"))),
        )
    except (TypeError, ValueError) as error:
        raise SynthesisArtifactError("invalid synthesis corpus") from error
    if (
        corpus.to_json().encode("ascii") != raw
        or corpus.digest != attempt.corpus_digest
    ):
        raise SynthesisArtifactError("synthesis corpus is not canonical")
    return corpus


def _synthesis_relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise SynthesisArtifactError(f"{label} must be a relative path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise SynthesisArtifactError(f"{label} must be a safe relative path")
    return path


def _synthesis_findings(
    attempt: SynthesisAttempt,
    corpus: EvidenceCorpus,
    runs: Sequence[Mapping[str, Any]],
    root: Path,
) -> list[dict[str, Any]]:
    run_map = {str(run.get("run_id")): run for run in runs}
    finding_ids: set[str] = set()
    result: list[dict[str, Any]] = []
    for finding in attempt.findings:
        if finding.finding_id in finding_ids:
            raise SynthesisArtifactError("synthesis findings contain duplicate IDs")
        finding_ids.add(finding.finding_id)
        validate_evidence_refs(corpus, finding.evidence_refs)
        targets: list[dict[str, Any]] = []
        public_refs: list[dict[str, Any]] = []
        for reference in finding.evidence_refs:
            entry = corpus.require(reference.evidence_id)
            target = _synthesis_navigation_target(entry, run_map, root)
            targets.append(target)
            public_refs.append(_public_synthesis_ref(entry.ref))
        counterevidence: list[object] = []
        counter_refs = tuple(
            item for item in finding.counterevidence if isinstance(item, EvidenceRef)
        )
        if counter_refs:
            validate_evidence_refs(corpus, counter_refs)
        for item in finding.counterevidence:
            if isinstance(item, EvidenceRef):
                entry = corpus.require(item.evidence_id)
                counterevidence.append(
                    {
                        "reference": _public_synthesis_ref(entry.ref),
                        "target": _synthesis_navigation_target(entry, run_map, root),
                    }
                )
            else:
                counterevidence.append(item)
        result.append(
            {
                "finding_id": finding.finding_id,
                "title": finding.title,
                "issue": finding.issue,
                "impact": finding.impact,
                "root_cause": finding.root_cause,
                "fixes": list(finding.fixes),
                "severity": _synthesis_enum_text(finding.severity),
                "confidence": finding.confidence,
                "evidence_refs": public_refs,
                "evidence_targets": targets,
                "affected_surfaces": list(finding.affected_surfaces),
                "principles": list(finding.principles),
                "counterevidence": counterevidence,
                "limitations": list(finding.limitations),
                "reviewer_state": finding.reviewer_state,
                "evidence_class": _synthesis_enum_text(finding.evidence_class),
                "reproducibility": _synthesis_enum_text(finding.reproducibility),
                "severity_justification": finding.severity_justification,
                "reviewer_notes": list(finding.reviewer_notes),
            }
        )
    return result


def _public_synthesis_ref(reference: EvidenceRef) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evidence_id": reference.evidence_id,
        "kind": reference.kind,
        "run_id": reference.run_id,
    }
    for name in (
        "viewport_id",
        "element_id",
        "event_id",
        "metric_id",
        "replay_sequence",
        "sha256",
    ):
        value = getattr(reference, name)
        if value is not None:
            result[name] = value
    return result


def _synthesis_navigation_target(
    entry: EvidenceEntry,
    run_map: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> dict[str, Any]:
    reference = entry.ref
    run = run_map.get(reference.run_id)
    if run is None or not bool(run.get("trusted", False)):
        raise SynthesisArtifactError(
            "synthesis reference does not target a trusted run"
        )
    target: dict[str, Any] = {
        "kind": reference.kind,
        "run_id": reference.run_id,
    }
    if reference.kind in {"event", "replay"}:
        sequence = reference.replay_sequence
        if sequence is None:
            raise SynthesisArtifactError("synthesis event reference has no sequence")
        event = _event_by_sequence(run, sequence)
        if event is None:
            raise SynthesisArtifactError("synthesis event reference is not recorded")
        if reference.event_id and event.get("event_id") != reference.event_id:
            raise SynthesisArtifactError("synthesis event reference ID mismatch")
        recorded_viewport_id = _recorded_event_viewport_id(run, event)
        if (
            reference.viewport_id is not None
            and recorded_viewport_id != reference.viewport_id
        ):
            raise SynthesisArtifactError(
                "synthesis event reference viewport ID mismatch"
            )
        recorded_element_id = _recorded_event_element_id(event)
        if (
            reference.element_id is not None
            and recorded_element_id != reference.element_id
        ):
            raise SynthesisArtifactError(
                "synthesis event reference element ID mismatch"
            )
        target.update(
            {
                "sequence": sequence,
                "event_id": _text(event.get("event_id")),
                "viewport_id": reference.viewport_id or recorded_viewport_id,
            }
        )
        if reference.element_id is not None or recorded_element_id is not None:
            target["element_id"] = reference.element_id or recorded_element_id
        return target
    if reference.kind == "viewport":
        if _snapshot_by_id(run, reference.viewport_id) is None:
            raise SynthesisArtifactError("synthesis viewport reference is not recorded")
        target["viewport_id"] = reference.viewport_id
        return target
    if reference.kind == "element":
        snapshot = _snapshot_by_id(run, reference.viewport_id)
        if snapshot is None or not any(
            element.get("id") == reference.element_id
            for element in _list_of_mappings(snapshot.get("elements"))
        ):
            raise SynthesisArtifactError("synthesis element reference is not recorded")
        target.update(
            {"viewport_id": reference.viewport_id, "element_id": reference.element_id}
        )
        return target
    if reference.kind in {"heatmap", "native-map", "saliency-metadata"}:
        payload = entry.payload
        namespace = _text(payload.get("namespace"), reference.viewport_id or "")
        duration = _optional_text(payload.get("duration"))
        group = _synthesis_saliency_group(run, namespace)
        if group is None or group.get("replay_available") is False:
            raise SynthesisArtifactError("synthesis saliency reference is unavailable")
        if duration is not None and not any(
            entry.get("duration") == duration
            for entry in _list_of_mappings(group.get("entries"))
        ):
            raise SynthesisArtifactError("synthesis saliency duration is not recorded")
        if reference.kind in {"heatmap", "native-map"}:
            if duration is None:
                raise SynthesisArtifactError(
                    "synthesis saliency reference has no duration"
                )
            _validate_synthesis_saliency_artifact(
                entry,
                run,
                root,
                namespace=namespace,
                duration=duration,
            )
        target.update(
            {
                "viewport_id": reference.viewport_id,
                "duration": duration,
                "namespace": namespace,
            }
        )
        return target
    if reference.kind == "screenshot":
        return _screenshot_navigation_target(entry, run, root, target)
    if reference.kind == "metric":
        metric_id = reference.metric_id
        if metric_id is None or not any(
            row.get("name") == metric_id
            for row in _list_of_mappings(run.get("metrics"))
        ):
            raise SynthesisArtifactError("synthesis metric reference is not recorded")
        target["metric_id"] = metric_id
        return target
    if reference.kind == "model-estimate":
        if reference.event_id is None or not reference.event_id.startswith("event-"):
            raise SynthesisArtifactError("synthesis estimate has no recorded event")
        sequence = _integer(reference.event_id.removeprefix("event-"))
        if sequence is None:
            raise SynthesisArtifactError("synthesis estimate event is not recorded")
        event = _event_by_sequence(run, sequence)
        if event is None or _text(event.get("event_id")) != reference.event_id:
            raise SynthesisArtifactError("synthesis estimate event ID mismatch")
        recorded_viewport_id = _recorded_event_viewport_id(run, event)
        if (
            reference.viewport_id is not None
            and recorded_viewport_id != reference.viewport_id
        ):
            raise SynthesisArtifactError("synthesis estimate viewport ID mismatch")
        if not _event_records_element(event, reference.element_id):
            raise SynthesisArtifactError("synthesis estimate element ID mismatch")
        target_viewport_id = reference.viewport_id or recorded_viewport_id
        _validate_synthesis_element_target(
            run,
            viewport_id=target_viewport_id,
            element_id=reference.element_id,
        )
        target.update(
            {
                "sequence": sequence,
                "event_id": reference.event_id,
                "viewport_id": target_viewport_id,
                "element_id": reference.element_id,
            }
        )
        return target
    if reference.kind == "ranked-element":
        payload = entry.payload
        namespace = _text(payload.get("namespace"), reference.viewport_id or "")
        duration = _optional_text(payload.get("duration"))
        if duration is None:
            raise SynthesisArtifactError("synthesis ranked element has no duration")
        group = _synthesis_saliency_group(run, namespace)
        if group is None or group.get("replay_available") is False:
            raise SynthesisArtifactError("synthesis ranked element is unavailable")
        if not any(
            item.get("duration") == duration
            for item in _list_of_mappings(group.get("entries"))
        ):
            raise SynthesisArtifactError(
                "synthesis ranked element duration is not recorded"
            )
        target_viewport_id = _optional_text(group.get("viewport_id"))
        _validate_synthesis_element_target(
            run,
            viewport_id=target_viewport_id,
            element_id=reference.element_id,
        )
        target.update(
            {
                "viewport_id": target_viewport_id,
                "element_id": reference.element_id,
                "duration": duration,
                "namespace": namespace,
            }
        )
        return target
    if any(
        getattr(reference, name) is not None
        for name in ("viewport_id", "element_id", "metric_id")
    ):
        raise SynthesisArtifactError(
            "synthesis reference contains unsupported target fields"
        )
    target.update(
        {
            "kind": "evidence-detail",
            "evidence_kind": reference.kind,
            "summary": entry.summary,
            "detail": _safe_value(dict(entry.payload)),
        }
    )
    return target


def _event_by_sequence(run: Mapping[str, Any], sequence: int) -> dict[str, Any] | None:
    return next(
        (
            event
            for event in _list_of_mappings(run.get("timeline"))
            if int(_number(event.get("sequence"), 0)) == sequence
        ),
        None,
    )


def _recorded_event_viewport_id(
    run: Mapping[str, Any], event: Mapping[str, Any]
) -> str | None:
    direct = _optional_text(event.get("viewport_id"))
    if direct is not None:
        return direct
    if _text(event.get("kind")) == "viewport-captured":
        snapshot_id = _optional_text(event.get("viewport_id"))
        if snapshot_id is not None:
            return snapshot_id
    observation = _mapping(event.get("observation"))
    observed = _optional_text(observation.get("viewport_id"))
    if observed is not None:
        return observed
    sequence = _integer(event.get("sequence"))
    if sequence is None:
        return None
    viewport_id: str | None = None
    for record in _list_of_mappings(run.get("timeline")):
        record_sequence = _integer(record.get("sequence"))
        if record_sequence is None or record_sequence > sequence:
            break
        if _text(record.get("kind")) == "viewport-captured":
            viewport_id = _optional_text(record.get("viewport_id"))
    return viewport_id


def _recorded_event_element_id(event: Mapping[str, Any]) -> str | None:
    direct = _optional_text(event.get("element_id"))
    if direct is not None:
        return direct
    action = _mapping(event.get("action"))
    return _optional_text(action.get("element_id"))


def _snapshot_by_id(
    run: Mapping[str, Any], viewport_id: str | None
) -> dict[str, Any] | None:
    if viewport_id is None:
        return None
    return next(
        (
            snapshot
            for snapshot in _list_of_mappings(run.get("snapshots"))
            if snapshot.get("id") == viewport_id
        ),
        None,
    )


def _validate_synthesis_element_target(
    run: Mapping[str, Any],
    *,
    viewport_id: str | None,
    element_id: str | None,
) -> None:
    if viewport_id is None or element_id is None:
        raise SynthesisArtifactError("synthesis target is incomplete")
    snapshot = _snapshot_by_id(run, viewport_id)
    if snapshot is None or not any(
        element.get("id") == element_id
        for element in _list_of_mappings(snapshot.get("elements"))
    ):
        raise SynthesisArtifactError("synthesis target element is not recorded")


def _event_records_element(event: Mapping[str, Any], element_id: str | None) -> bool:
    if element_id is None:
        return False
    if _recorded_event_element_id(event) == element_id:
        return True
    return any(
        score.get("element_id") == element_id
        for score in _list_of_mappings(event.get("scores"))
    )


def _synthesis_saliency_group(
    run: Mapping[str, Any], namespace: str
) -> dict[str, Any] | None:
    return next(
        (
            group
            for group in _list_of_mappings(run.get("saliency"))
            if group.get("artifact_namespace") == namespace
            or group.get("viewport_id") == namespace
        ),
        None,
    )


def _validate_synthesis_saliency_artifact(
    entry: EvidenceEntry,
    run: Mapping[str, Any],
    root: Path,
    *,
    namespace: str,
    duration: str,
) -> None:
    reference = entry.ref
    if reference.artifact_path is None or reference.sha256 is None:
        raise SynthesisArtifactError("synthesis saliency reference is incomplete")
    root_relative = _synthesis_relative_path(
        reference.artifact_path, "synthesis saliency path"
    )
    parts = root_relative.parts
    if len(parts) < 3 or parts[0] != "runs" or parts[1] != reference.run_id:
        raise SynthesisArtifactError("synthesis saliency path is outside its run")
    run_relative = PurePosixPath(*parts[2:])
    filename = (
        f"{duration}-heatmap.png" if reference.kind == "heatmap" else f"{duration}.npz"
    )
    expected_relative = PurePosixPath("saliency", namespace, filename)
    if run_relative != expected_relative:
        raise SynthesisArtifactError("synthesis saliency path is not recorded")
    group = _synthesis_saliency_group(run, namespace)
    if group is None or group.get("replay_available") is False:
        raise SynthesisArtifactError("synthesis saliency reference is unavailable")
    if expected_relative.as_posix() not in _strings(group.get("artifact_ids")):
        raise SynthesisArtifactError("synthesis saliency artifact is not recorded")
    if reference.kind == "heatmap" and not any(
        _text(item.get("heatmap_path"), "") == expected_relative.as_posix()
        for item in _list_of_mappings(group.get("entries"))
        if item.get("duration") == duration
    ):
        raise SynthesisArtifactError("synthesis heatmap is not recorded")
    run_path = root / _text(run.get("bundle_path"))
    candidate = _secure_bundle_file(run_path, run_relative)
    if candidate is None:
        raise SynthesisArtifactError("synthesis saliency path is unavailable")
    maximum = (
        _MAX_SYNTHESIS_HEATMAP_BYTES
        if reference.kind == "heatmap"
        else _MAX_SYNTHESIS_NATIVE_MAP_BYTES
    )
    content = secure_read_bytes(
        candidate,
        "synthesis saliency artifact",
        max_bytes=maximum,
    )
    if hashlib.sha256(content).hexdigest() != reference.sha256:
        raise SynthesisArtifactError("synthesis saliency artifact digest mismatch")
    try:
        if reference.kind == "heatmap":
            validate_saliency_heatmap_content(content)
        else:
            validate_saliency_native_map_content(content)
    except ValueError as error:
        raise SynthesisArtifactError(
            "synthesis saliency artifact is invalid"
        ) from error


def _screenshot_navigation_target(
    entry: EvidenceEntry,
    run: Mapping[str, Any],
    root: Path,
    target: dict[str, Any],
) -> dict[str, Any]:
    reference = entry.ref
    artifact = reference.artifact_path
    if artifact is None or reference.sha256 is None:
        raise SynthesisArtifactError("synthesis screenshot reference is incomplete")
    root_relative = _synthesis_relative_path(artifact, "synthesis screenshot path")
    parts = root_relative.parts
    if len(parts) < 3 or parts[0] != "runs" or parts[1] != reference.run_id:
        raise SynthesisArtifactError("synthesis screenshot path is outside its run")
    run_relative = PurePosixPath(*parts[2:])
    snapshot = next(
        (
            snapshot
            for snapshot in _list_of_mappings(run.get("snapshots"))
            if _text(snapshot.get("artifact"), "").replace("\\", "/")
            == run_relative.as_posix()
            and (
                reference.viewport_id is None
                or snapshot.get("id") == reference.viewport_id
            )
        ),
        None,
    )
    if snapshot is None:
        raise SynthesisArtifactError("synthesis screenshot is not recorded")
    run_path = root / _text(run.get("bundle_path"))
    candidate = _secure_bundle_file(run_path, run_relative)
    if candidate is None:
        raise SynthesisArtifactError("synthesis screenshot path is unavailable")
    content = secure_read_bytes(
        candidate,
        "synthesis screenshot",
        max_bytes=_MAX_SOURCE_SCREENSHOT_BYTES,
    )
    if hashlib.sha256(content).hexdigest() != reference.sha256:
        raise SynthesisArtifactError("synthesis screenshot digest mismatch")
    target["viewport_id"] = snapshot.get("id")
    return target


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _project_synthesis(
    synthesis: dict[str, Any],
    *,
    run_links: Mapping[str, str] | None,
    run_scope: frozenset[str] | None,
) -> dict[str, Any]:
    if not synthesis:
        return {
            "synthesis_status": "missing",
            "status": "missing",
            "using_fallback": True,
            "findings": [],
            "fallback_findings": [],
            "limitations": ["No persisted report synthesis is available."],
            "tested_scope": {},
            "assessment": (
                "Deterministic findings are shown because report synthesis is unavailable."
            ),
        }
    projected = dict(synthesis)
    for name in ("findings", "fallback_findings"):
        projected[name] = [
            _project_synthesis_finding(item, run_links=run_links)
            for item in _list_of_mappings(synthesis.get(name))
            if run_scope is None or _finding_targets_run(item, run_scope)
        ]
    return projected


def _project_synthesis_finding(
    finding: Mapping[str, Any],
    *,
    run_links: Mapping[str, str] | None,
) -> dict[str, Any]:
    projected = dict(finding)
    targets: list[dict[str, Any]] = []
    for target in _list_of_mappings(finding.get("evidence_targets")):
        item = dict(target)
        if run_links is not None:
            item["run_page"] = run_links.get(_text(item.get("run_id")), "")
        targets.append(item)
    projected["evidence_targets"] = targets
    references: list[dict[str, Any]] = []
    for reference in _list_of_mappings(finding.get("evidence_refs")):
        item = dict(reference)
        target = _mapping(reference.get("target"))
        if target:
            target = dict(target)
            if run_links is not None:
                target["run_page"] = run_links.get(_text(target.get("run_id")), "")
            item["target"] = target
        references.append(item)
    projected["evidence_refs"] = references
    return projected


def _finding_targets_run(finding: Mapping[str, Any], run_scope: frozenset[str]) -> bool:
    if any(run_id in run_scope for run_id in _strings(finding.get("run_ids"))):
        return True
    return any(
        _text(target.get("run_id")) in run_scope
        for target in _list_of_mappings(finding.get("evidence_targets"))
    )


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
        "config_digest": _text(manifest.get("config_digest")),
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
    run_scope: frozenset[str] | None = None,
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
    provider_comparisons = _report_provider_comparisons(
        experiment.get("provider_comparisons", []),
        {run["run_id"] for run in runs},
        run_links,
    )
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
    synthesis = _project_synthesis(
        _mapping(experiment.get("synthesis")),
        run_links=run_links,
        run_scope=run_scope,
    )
    if synthesis.get("using_fallback"):
        runs = _project_fallback_run_findings(runs, synthesis.get("findings"))
    concise_index_fallback = (
        not include_run_payload
        and run_scope is None
        and bool(synthesis.get("using_fallback"))
    )
    report_payload = {
        "runs": runs,
        "run_rows": experiment["run_rows"],
        "comparison_rows": experiment["comparison_rows"],
        "provider_comparisons": provider_comparisons,
        "gate_rows": experiment["gate_rows"],
        "failure_rows": failure_rows,
        "evidence_summary": experiment["evidence_summary"],
        "focused_acceptance": experiment["focused_acceptance"],
        "limitations": limitations,
    }
    if not concise_index_fallback:
        report_payload["synthesis"] = synthesis
        report_payload["synthesis_status"] = synthesis["synthesis_status"]
    return {
        "runs": runs,
        "run_rows": experiment["run_rows"],
        "comparison_rows": experiment["comparison_rows"],
        "provider_comparisons": provider_comparisons,
        "gate_rows": experiment["gate_rows"],
        "failure_rows": failure_rows,
        "evidence_summary": experiment["evidence_summary"],
        "focused_acceptance": experiment["focused_acceptance"],
        "limitations": limitations,
        "initial_viewport_width": initial_width,
        "synthesis": synthesis,
        "synthesis_status": synthesis["synthesis_status"],
        "report_json": _safe_json(report_payload),
    }


def _project_fallback_run_findings(
    runs: Sequence[Mapping[str, Any]],
    fallback_findings: object,
) -> list[dict[str, Any]]:
    fallback_by_key = {
        (
            _text(finding.get("run_ids", [""])[0]),
            _text(finding.get("finding_id")),
        ): finding
        for finding in _list_of_mappings(fallback_findings)
        if _strings(finding.get("run_ids"))
    }
    projected_runs: list[dict[str, Any]] = []
    for source in runs:
        run = dict(source)
        projected_findings: list[dict[str, Any]] = []
        for finding in _list_of_mappings(source.get("findings")):
            key = (_text(source.get("run_id")), _text(finding.get("finding_id")))
            fallback = fallback_by_key.get(key)
            if fallback is None:
                projected_findings.append(dict(finding))
                continue
            projected_findings.append(
                {
                    **finding,
                    "title": fallback.get("fallback_title", finding.get("title")),
                    "cause": fallback.get("fallback_issue", finding.get("cause")),
                    "fallback_context": fallback.get("fallback_context", {}),
                    "fallback_title": fallback.get("fallback_title", ""),
                    "fallback_issue": fallback.get("fallback_issue", ""),
                    "fallback_impact": fallback.get("fallback_impact", ""),
                    "fallback_root_cause": fallback.get("fallback_root_cause", ""),
                    "fallback_fix": fallback.get("fallback_fix", ""),
                }
            )
        run["findings"] = projected_findings
        projected_runs.append(run)
    return projected_runs


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
    index_javascript = (template_root / "static" / "report-index.js").read_text(
        encoding="utf-8"
    )
    return template.render(
        title=title,
        css=css,
        javascript=javascript,
        index_javascript=index_javascript,
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
    if path.suffix.casefold() in {".gif", ".jpeg", ".jpg", ".png"}:
        return True
    return len(path.name) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in path.name
    )


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


_saliency_artifact_data_uri = _trusted_saliency_artifact_data_uri


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
    return list(
        _load_trusted_saliency_replay(
            Path(run_path),
            events,
            snapshots,
            expected_provider_id=expected_provider_id,
        )
    )


_merge_saliency_event = _merge_trusted_saliency_event


def _validated_saliency_artifact(value: str) -> str | None:
    try:
        return validate_saliency_artifact_path(value).as_posix()
    except ValueError:
        return None


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


def _provider_comparisons(runs: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for run in runs:
        provider_id = _text(run.get("prominence_provider_id"), "heuristic")
        if provider_id not in {"heuristic", "foveacast"}:
            continue
        groups[
            (
                run["scenario_id"],
                run["version_id"],
                run["persona_id"],
                run["policy"],
                int(_number(run.get("seed"), 0)),
                int(_number(run.get("model_trial"), 0)),
            )
        ].append(run)

    comparisons: list[dict[str, Any]] = []
    for identity, grouped in sorted(groups.items()):
        providers = [
            _provider_comparison(run)
            for run in sorted(
                grouped,
                key=lambda item: _text(item.get("prominence_provider_id")),
            )
        ]
        if len({provider["provider_id"] for provider in providers}) < 2:
            continue
        scenario_id, version_id, persona_id, policy, seed, model_trial = identity
        comparisons.append(
            {
                "key": "|".join(
                    (
                        scenario_id,
                        version_id,
                        persona_id,
                        policy,
                        str(seed),
                        str(model_trial),
                    )
                ),
                "scenario_id": scenario_id,
                "scenario_label": grouped[0]["scenario_label"],
                "version_id": version_id,
                "version_label": grouped[0]["version_label"],
                "persona_id": persona_id,
                "persona_label": grouped[0]["persona_label"],
                "policy": policy,
                "seed": seed,
                "model_trial": model_trial,
                "providers": providers,
            }
        )
    return comparisons


def _provider_comparison(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "provider_id": _text(run.get("prominence_provider_id"), "heuristic"),
        "outcome": run["outcome"],
        "stage": run["stage"],
        "verified": bool(run["verified"]),
        "trusted": bool(run["trusted"]),
        "comparison_valid": bool(run["comparison_valid"]),
        "failure_reason": run["failure_reason"],
        "heatmaps": _comparison_heatmaps(run),
        "prominence": _comparison_prominence(run),
        "action_path": _comparison_action_path(run),
    }


def _comparison_heatmaps(run: dict[str, Any]) -> list[dict[str, Any]]:
    heatmaps: list[dict[str, Any]] = []
    for group in _list_of_mappings(run.get("saliency")):
        for entry in _list_of_mappings(group.get("entries")):
            heatmaps.append(
                {
                    "duration": _text(entry.get("duration"), "unavailable"),
                    "heatmap": entry.get("heatmap"),
                    "heatmap_path": _safe_text(
                        entry.get("heatmap_path"), "unavailable"
                    ),
                    "viewport_id": _safe_identifier(
                        group.get("viewport_id"), "unavailable"
                    ),
                    "provider_id": _safe_identifier(
                        entry.get("provider_id"), "unavailable"
                    ),
                    "execution_provider": _safe_identifier(
                        entry.get("execution_provider"), "unavailable"
                    ),
                    "cache_state": _safe_identifier(
                        group.get("cache_state"), "unavailable"
                    ),
                    "inference_duration_ms": _number(
                        entry.get("inference_duration_ms"), 0
                    ),
                    "ranked_elements": _list_of_mappings(entry.get("ranked_elements"))[
                        :12
                    ],
                }
            )
    return heatmaps


def _comparison_prominence(run: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots = {
        snapshot["id"]: snapshot
        for snapshot in _list_of_mappings(run.get("snapshots"))
        if snapshot.get("id")
    }
    records: list[dict[str, Any]] = []
    for record in _list_of_mappings(run.get("prominence")):
        snapshot = snapshots.get(_text(record.get("viewport_id")), {})
        elements = {
            item["id"]: item
            for item in _list_of_mappings(snapshot.get("elements"))
            if item.get("id")
        }
        rankings: list[dict[str, Any]] = []
        scores = sorted(
            _list_of_mappings(record.get("scores")),
            key=lambda item: _number(item.get("raw_score", item.get("score")), 0),
            reverse=True,
        )
        for rank, score in enumerate(scores[:12], start=1):
            element_id = _safe_identifier(score.get("element_id"), "unavailable")
            element = elements.get(element_id, {})
            rankings.append(
                {
                    "rank": rank,
                    "element_id": element_id,
                    "label": _safe_text(element.get("label"), "Unlabelled element"),
                    "role": _safe_text(element.get("role"), "other"),
                    "score": _number(score.get("raw_score", score.get("score")), 0),
                    "normalized_probability": _number(
                        score.get("normalized_probability"), 0
                    ),
                }
            )
        records.append(
            {
                "sequence": int(_number(record.get("sequence"), 0)),
                "viewport_id": _safe_identifier(
                    record.get("viewport_id"), "unavailable"
                ),
                "rankings": rankings,
            }
        )
    return records


def _comparison_action_path(run: dict[str, Any]) -> list[dict[str, Any]]:
    path: list[dict[str, Any]] = []
    for action_record in _list_of_mappings(run.get("actions")):
        action = _mapping(action_record.get("action"))
        element_id = _optional_text(action.get("element_id"))
        element_label = "No element target"
        if element_id:
            snapshot = _snapshot_before_sequence(
                run, int(_number(action_record.get("sequence"), 0))
            )
            element = next(
                (
                    item
                    for item in _list_of_mappings(snapshot.get("elements"))
                    if item.get("id") == element_id
                ),
                cast(dict[str, Any], {}),
            )
            element_label = _safe_text(element.get("label"), "Unlabelled element")
        path.append(
            {
                "sequence": int(_number(action_record.get("sequence"), 0)),
                "kind": _safe_text(action.get("kind"), "action"),
                "element_id": element_id,
                "element_label": element_label,
                "succeeded": action_record.get("succeeded"),
                "error": _optional_text(action_record.get("error")),
            }
        )
    return path


def _snapshot_before_sequence(run: dict[str, Any], sequence: int) -> dict[str, Any]:
    viewport_id = ""
    for event in _list_of_mappings(run.get("timeline")):
        if int(_number(event.get("sequence"), 0)) > sequence:
            break
        if _text(event.get("kind")) == "viewport-captured":
            viewport_id = _text(event.get("viewport_id"))
    return next(
        (
            snapshot
            for snapshot in _list_of_mappings(run.get("snapshots"))
            if snapshot.get("id") == viewport_id
        ),
        cast(dict[str, Any], {}),
    )


def _report_provider_comparisons(
    comparisons: object,
    run_ids: set[str],
    run_links: dict[str, str] | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for comparison in _list_of_mappings(comparisons):
        providers: list[dict[str, Any]] = []
        for provider in _list_of_mappings(comparison.get("providers")):
            run_id = _text(provider.get("run_id"))
            if run_id not in run_ids:
                continue
            copied = dict(provider)
            if run_links is not None:
                copied["run_page"] = run_links.get(run_id, "")
            providers.append(copied)
        if len({item.get("provider_id") for item in providers}) < 2:
            continue
        result.append({**comparison, "providers": providers})
    return result


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


def _synthesis_enum_text(value: object) -> str:
    return _text(getattr(value, "value", value))


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
