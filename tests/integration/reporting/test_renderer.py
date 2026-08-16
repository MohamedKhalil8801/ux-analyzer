from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
import pytest
from PIL import Image
from playwright.async_api import Route, async_playwright

import ux_analyzer.reporting.renderer as renderer
from ux_analyzer.application.evidence_corpus import EvidenceCorpus, EvidenceEntry
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.run import RunStarted
from ux_analyzer.domain.synthesis import (
    CANONICAL_SYNTHESIS_ROLES,
    EvidenceRef,
    SynthesisAttempt,
    SynthesisFinding,
    SynthesisRoleReceipt,
    SynthesisStatus,
)
from ux_analyzer.ports.artifacts import (
    BundleManifest,
    BundleStateError,
    RedactionPolicy,
    SaliencyArtifactKind,
    canonicalize_saliency_artifact_content,
)
from ux_analyzer.reporting.renderer import render_experiment_report
from ux_analyzer.storage.run_bundle import FilesystemRunBundleWriter
from ux_analyzer.storage.synthesis_artifacts import (
    MAX_SYNTHESIS_JSON_BYTES,
    SynthesisArtifactStore,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _synthesis_expectation_digest() -> str:
    return hashlib.sha256(b"[]").hexdigest()


def _synthesis_ref(
    kind: str,
    run_id: str = "run-1",
    *,
    sequence: int = 7,
    viewport_id: str | None = "viewport-1",
    element_id: str | None = None,
    metric_id: str | None = None,
) -> EvidenceRef:
    evidence_id = f"{kind}:{run_id}:"
    if kind in {"event", "replay"}:
        evidence_id += str(sequence)
        return EvidenceRef(
            evidence_id,
            kind,
            run_id,
            viewport_id=viewport_id,
            event_id=f"event-{sequence}",
            replay_sequence=sequence,
        )
    if kind == "viewport":
        return EvidenceRef(
            f"viewport:{run_id}:{viewport_id}",
            kind,
            run_id,
            viewport_id=viewport_id,
        )
    if kind == "element":
        return EvidenceRef(
            f"element:{run_id}:{viewport_id}:{element_id}",
            kind,
            run_id,
            viewport_id=viewport_id,
            element_id=element_id,
        )
    if kind == "metric":
        return EvidenceRef(
            f"metric:{run_id}:{metric_id}",
            kind,
            run_id,
            metric_id=metric_id,
        )
    if kind == "heatmap":
        return EvidenceRef(
            f"heatmap:{run_id}:{viewport_id}:3s",
            kind,
            run_id,
            viewport_id=viewport_id,
            artifact_path=f"runs/{run_id}/saliency/inference-1/3s-heatmap.png",
            sha256="866f97bc38d8251e7c689fe970efb5c943bfbe0c72b0cfdab1b68c79ce005aa5",
        )
    raise AssertionError(f"unsupported synthesis fixture reference: {kind}")


def _synthesis_payload(ref: EvidenceRef) -> dict[str, object]:
    payload: dict[str, object] = {"evidence_id": ref.evidence_id}
    if ref.kind in {"heatmap", "native-map"}:
        payload.update({"namespace": "inference-1", "duration": "3s"})
    return payload


def _write_synthesis(
    root: Path,
    *,
    status: SynthesisStatus = SynthesisStatus.ACCEPTED,
    corpus_refs: tuple[EvidenceRef, ...] | None = None,
    finding_refs: tuple[EvidenceRef, ...] = (),
    finding_title: str = "Accepted synthesis finding",
    sequence: int = 1,
    run_id: str = "run-1",
    findings: tuple[SynthesisFinding, ...] | None = None,
    limitations: tuple[str, ...] | None = None,
    include_scope_identity: bool = True,
    scope_run_ids: tuple[str, ...] | None = None,
    corpus_marker: str | None = None,
    created_at: str = "2026-08-10T12:00:00+00:00",
) -> None:
    finding_values = findings
    if finding_values is None:
        finding_values = ()
        if status is SynthesisStatus.ACCEPTED and not finding_refs:
            finding_refs = (_synthesis_ref("event", run_id),)
        if finding_refs:
            finding_values = (
                SynthesisFinding(
                    finding_id="synthesis-finding",
                    title=finding_title,
                    issue="The tested task takes extra navigation.",
                    impact="The tested task takes longer to complete.",
                    root_cause="The task entry point is hard to identify.",
                    fixes=("Label the entry point around the user's task.",),
                    severity="high",
                    confidence=0.9,
                    evidence_refs=finding_refs,
                    reviewer_state="accepted",
                    severity_justification="The recorded action sequence shows extra navigation.",
                ),
            )
    refs = (
        corpus_refs
        or finding_refs
        or tuple(ref for finding in finding_values for ref in finding.evidence_refs)
    )
    entries = tuple(
        EvidenceEntry(
            ref=ref,
            evidence_class=EvidenceClass.DETERMINISTIC_FACT,
            summary=f"Recorded {ref.evidence_id}.",
            payload=_synthesis_payload(ref),
        )
        for ref in refs
    )
    scoped_run_ids = scope_run_ids or (run_id,)
    metadata: dict[str, object] = {"experiment_run_ids": scoped_run_ids}
    metadata["finalized_bundle_checksums"] = tuple(
        {
            "run_id": scoped_run_id,
            "checksums_sha256": hashlib.sha256(
                (root / "runs" / scoped_run_id / "checksums.sha256").read_bytes()
            ).hexdigest(),
        }
        for scoped_run_id in scoped_run_ids
    )
    if corpus_marker is not None:
        metadata["marker"] = corpus_marker
    if include_scope_identity:
        metadata["experiment_run_identities"] = tuple(
            {
                "run_id": scoped_run_id,
                "seed": loaded_run["seed"],
                "model_trial": loaded_run["model_trial"],
                "config_digest": manifest.get("config_digest"),
                "scenario_id": loaded_run["scenario_id"],
                "application_version_id": loaded_run["version_id"],
                "persona_id": loaded_run["persona_id"],
                "policy": loaded_run["policy"],
                "prominence_provider_id": loaded_run["prominence_provider_id"],
            }
            for scoped_run_id in scoped_run_ids
            for loaded_run in (renderer._load_run(root / "runs" / scoped_run_id),)
            for manifest in (
                json.loads(
                    (root / "runs" / scoped_run_id / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )
        )
    corpus = EvidenceCorpus(
        output_root=root,
        entries=entries,
        metadata=metadata,
    )
    expectation_payloads = [
        dict(entry.payload) for entry in entries if entry.ref.kind == "expectation"
    ]
    expectation_digest = hashlib.sha256(
        json.dumps(
            expectation_payloads,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    attempt = SynthesisAttempt(
        attempt_id=f"20260810T120000Z-{corpus.digest[:12]}-{sequence}",
        status=status,
        corpus_digest=corpus.digest,
        expectation_digest=expectation_digest,
        principle_pack_digest=corpus.principle_pack_digest,
        prompt_version="report-synthesis-orchestrator-v1",
        schema_version="synthesis-v1",
        role_receipts=tuple(
            SynthesisRoleReceipt(
                role=role,
                provider_id="fixture-provider",
                model_id="fixture-model",
                prompt_digest=hashlib.sha256(f"{role}:prompt".encode()).hexdigest(),
                schema_digest=hashlib.sha256(f"{role}:schema".encode()).hexdigest(),
                output_digest=hashlib.sha256(f"{role}:output".encode()).hexdigest(),
            )
            for role in CANONICAL_SYNTHESIS_ROLES
        ),
        candidate_findings=finding_values,
        rejected_findings=(
            tuple(
                replace(finding, reviewer_state="not-established")
                for finding in finding_values
            )
            if status is SynthesisStatus.REJECTED
            else ()
        ),
        findings=(finding_values if status is SynthesisStatus.ACCEPTED else ()),
        limitations=(
            limitations
            if limitations is not None
            else ("Fixture synthesis evidence only.",)
        ),
        created_at=created_at,
    )
    SynthesisArtifactStore(root).write_attempt(attempt, corpus)


def _rewrite_selected_synthesis(
    root: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    attempt_root = next((root / "synthesis" / "attempts").iterdir())
    synthesis_path = attempt_root / "synthesis.json"
    value = json.loads(synthesis_path.read_text(encoding="ascii"))
    mutate(value)
    content = (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    synthesis_path.write_bytes(content)
    index_path = root / "synthesis" / "index.json"
    index = json.loads(index_path.read_text(encoding="ascii"))
    index["attempts"][0]["status"] = value["status"]
    index["attempts"][0]["synthesis_digest"] = hashlib.sha256(content).hexdigest()
    index_path.write_bytes(
        (
            json.dumps(index, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")
    )


def _write_checksums(run: Path) -> None:
    files = sorted(
        path
        for path in run.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    (run / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(run).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def _write_saliency_replay_evidence(
    root: Path,
    run_id: str,
    *,
    namespace: str = "inference-1",
    source_viewport_id: str = "viewport-1",
    source_event_id: str = "event-1",
) -> None:
    run = root / "runs" / run_id
    saliency_root = run / "saliency" / namespace
    saliency_root.mkdir(parents=True)
    heatmap = BytesIO()
    Image.new("L", (2, 2), color=180).save(heatmap, format="PNG")
    for duration in ("1s", "3s", "7s"):
        (saliency_root / f"{duration}-heatmap.png").write_bytes(heatmap.getvalue())
    checksums = ["1" * 64, "2" * 64, "3" * 64]
    geometry = {
        "geometry_version": "saliency-geometry-v1",
        "source_dimensions": [800, 600],
        "native_dimensions": [2, 2],
        "content_dimensions": [2, 2],
        "pad_left": 0,
        "pad_top": 0,
        "pad_right": 0,
        "pad_bottom": 0,
        "scale": 1.0,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "device_pixel_ratio": 1.0,
        "zoom": 1.0,
    }
    native_map = BytesIO()
    np.savez_compressed(
        native_map,
        values=np.full((2, 2), 0.5, dtype=np.float32),
        geometry=np.asarray(
            (
                *geometry["source_dimensions"],
                *geometry["native_dimensions"],
                *geometry["content_dimensions"],
                geometry["pad_left"],
                geometry["pad_top"],
                geometry["pad_right"],
                geometry["pad_bottom"],
                geometry["scale"],
                geometry["scale_x"],
                geometry["scale_y"],
                geometry["device_pixel_ratio"],
                geometry["zoom"],
            ),
            dtype=np.float64,
        ),
    )
    for duration in ("1s", "3s", "7s"):
        (saliency_root / f"{duration}.npz").write_bytes(native_map.getvalue())

    def prediction_metadata(index: int) -> dict[str, object]:
        return {
            "provider_id": "foveacast",
            "model_id": "foveacast-v0.2.0",
            "provider_version": "foveacast-adapter-v1",
            "model_version": "v0.2.0",
            "model_checksum": checksums[index],
            "input_dimensions": [2, 2],
            "output_dimensions": [2, 2],
            "geometry": geometry,
            "preprocessing_version": "foveacast-preprocess-v1",
            "inference_duration_ms": float(index + 1),
            "execution_provider": "CPUExecutionProvider",
            "warnings": [],
            "cache_state": "miss",
        }

    profiles = [
        {
            "viewport_id": namespace,
            "element_id": "target",
            "immediate": {
                "kind": "predicted",
                "score": 0.9,
                "source": "foveacast-v0.2.0",
            },
            "early": {
                "kind": "predicted",
                "score": 0.7,
                "source": "foveacast-v0.2.0",
            },
            "eventual": {
                "kind": "predicted",
                "score": 0.5,
                "source": "foveacast-v0.2.0",
            },
            "general": None,
            "aggregates": [
                {
                    "viewport_id": namespace,
                    "element_id": "target",
                    "duration": duration,
                    "density": 0.4,
                    "robust_peak": 0.8,
                    "raw_mass": 4.0,
                    "mass_share": 0.6,
                    "clipped_area": 7200.0,
                    "visibility_fraction": 1.0,
                    "occlusion_fraction": 0.0,
                    "raw_score": 0.5,
                    "adjusted_score": 0.5,
                }
                for duration in ("1s", "3s", "7s")
            ],
            "aggregation_version": "element-saliency-aggregation-v1",
            "prediction_provenance": [
                {
                    "duration": duration,
                    "metadata": prediction_metadata(index),
                }
                for index, duration in enumerate(("1s", "3s", "7s"))
            ],
        }
    ]
    artifact_paths = [
        f"saliency/{namespace}/{filename}"
        for filename in (
            "1s.npz",
            "3s.npz",
            "7s.npz",
            "1s-heatmap.png",
            "3s-heatmap.png",
            "7s-heatmap.png",
            "profiles.json",
            "metadata.json",
        )
    ]
    cache_key = {
        "viewport_id": namespace,
        "screenshot_sha256": "a" * 64,
        "screenshot_dimensions": [800, 600],
        "device_pixel_ratio": 1.0,
        "zoom": 1.0,
        "model_checksums": checksums,
        "preprocessing_version": "foveacast-preprocess-v1",
        "precision": "fp16",
        "execution_provider": "CPUExecutionProvider",
        "aggregation_version": "element-saliency-aggregation-v1",
        "geometry_version": "saliency-geometry-v1",
    }
    metadata = {
        "cache_version": "saliency-cache-v1",
        "cache_key": cache_key,
        "cache_key_digest": hashlib.sha256(
            json.dumps(cache_key, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "viewport_id": namespace,
        "aggregation_version": "element-saliency-aggregation-v1",
        "predictions": [
            {"duration": duration, "metadata": prediction_metadata(index)}
            for index, duration in enumerate(("1s", "3s", "7s"))
        ],
        "saliency_metadata": {
            "provider_manifests": [
                {
                    "provider_id": "foveacast",
                    "role": "prominence",
                    "model_id": "foveacast-v0.2.0",
                    "endpoint_origin": "internal",
                    "version": "v0.2.0",
                    "prompt_version": None,
                    "schema_version": None,
                }
            ],
            "aggregation_version": "element-saliency-aggregation-v1",
            "warnings": [],
        },
        "warnings": ["overlay-redacted"],
        "artifact_paths": artifact_paths,
    }
    profiles_bytes = canonicalize_saliency_artifact_content(
        SaliencyArtifactKind.PROFILES,
        json.dumps(profiles, sort_keys=True, separators=(",", ":")).encode(),
        expected_viewport_id=namespace,
    )
    metadata_bytes = canonicalize_saliency_artifact_content(
        SaliencyArtifactKind.METADATA,
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(),
        expected_viewport_id=namespace,
    )
    (saliency_root / "profiles.json").write_bytes(profiles_bytes)
    (saliency_root / "metadata.json").write_bytes(metadata_bytes)
    events = [
        json.loads(line)
        for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    terminal = events.pop()
    artifact_ids = [
        f"saliency/{namespace}/{filename}"
        for filename in (
            "1s.npz",
            "3s.npz",
            "7s.npz",
            "1s-heatmap.png",
            "3s-heatmap.png",
            "7s-heatmap.png",
            "profiles.json",
            "metadata.json",
        )
    ]
    profile_event_id = f"event-{len(events) + 2}"
    events.extend(
        [
            {
                "kind": "saliency-inference-recorded",
                "viewport_id": namespace,
                "source_viewport_id": source_viewport_id,
                "artifact_namespace": namespace,
                "provider_id": "foveacast",
                "search_stage": "initial",
                "model_checksums": checksums,
                "execution_provider": "CPUExecutionProvider",
                "preprocessing_version": "foveacast-preprocess-v1",
                "precision": "fp16",
                "cache_state": "miss",
                "timings_ms": [1.0, 2.0, 3.0],
                "artifact_ids": artifact_ids,
                "warnings": ["overlay-redacted"],
            },
            {
                "kind": "saliency-profiles-recorded",
                "viewport_id": namespace,
                "source_viewport_id": source_viewport_id,
                "artifact_namespace": namespace,
                "source_event_id": source_event_id,
                "provider_id": "foveacast",
                "search_stage": "initial",
                "model_checksums": checksums,
                "execution_provider": "CPUExecutionProvider",
                "preprocessing_version": "foveacast-preprocess-v1",
                "precision": "fp16",
                "cache_state": "miss",
                "timings_ms": [1.0, 2.0, 3.0],
                "artifact_ids": artifact_ids,
                "warnings": ["overlay-redacted"],
            },
            {
                "kind": "prominence-recorded",
                "viewport_id": source_viewport_id,
                "source_viewport_id": source_viewport_id,
                "artifact_namespace": namespace,
                "source_event_id": profile_event_id,
                "provider_id": "foveacast-prominence",
                "active_provider_id": "foveacast",
                "search_stage": "initial",
                "selected_mixture": [["1s", 1.0]],
                "selected_element_ids": ["target"],
                "cache_state": "miss",
            },
        ]
    )
    events.append(terminal)
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_checksums(run)


def _write_run(
    root: Path,
    run_id: str,
    *,
    version: str,
    discovery_cost: float,
    model_trial: int = 2,
    screenshot: bytes = b"not-an-image",
    outcome: str = "verified-success",
    verified: bool | None = None,
    terminal_reason: str | None = None,
    evaluation_failure_reason: str | None = None,
    ux_sample_valid: bool | None = None,
    ux_sample_invalid_reason: str | None = None,
    prominence_provider_id: str = "heuristic",
    event_overrides: dict[int, dict[str, object]] | None = None,
) -> None:
    is_verified = outcome == "verified-success" if verified is None else verified
    is_valid_sample = (
        outcome in {"verified-success", "agent-abandoned", "budget-exhausted"}
        and evaluation_failure_reason is None
        if ux_sample_valid is None
        else ux_sample_valid
    )
    run = root / "runs" / run_id
    (run / "artifacts").mkdir(parents=True)
    (run / "artifacts" / "screenshot.png").write_bytes(screenshot)
    _write_json(
        run / "manifest.json",
        {
            "run_id": run_id,
            "seed": 7,
            "model_trial": model_trial,
            "prominence_provider_id": prominence_provider_id,
            "config_digest": "config-sha",
            "endpoint_origin": "https://llm.example.test/v1",
            "model_ids": {"cognitive": "model-v1"},
            "prompt_versions": {"cognitive": "cognitive-v1"},
            "package_version": "0.1.0",
            "provider_versions": {"observation": "fixture-v1"},
            "provider_manifests": [
                {
                    "provider_id": "provider",
                    "role": "cognitive",
                    "model_id": "model-v1",
                    "endpoint_origin": "https://llm.example.test",
                    "version": "1",
                }
            ],
        },
    )
    events = [
        {
            "sequence": 1,
            "kind": "viewport-captured",
            "viewport_width": 800,
            "viewport_height": 600,
            "snapshot": {
                "id": "viewport-1",
                "screenshot_artifact": "artifacts/screenshot.png",
                "elements": [
                    {
                        "id": "target",
                        "role": "button",
                        "label": '<img src=x onerror="alert(1)">',
                        "bounds": {"x": 40, "y": 50, "width": 180, "height": 40},
                        "visibility_fraction": 1,
                        "occlusion_fraction": 0.25,
                        "local_contrast": 0.75,
                        "actionable": True,
                        "disabled": False,
                        "region_id": "team",
                        "selector": "button[data-testid=secret]",
                        "test_id": "secret",
                        "execution_reference": {
                            "provider_id": "provider",
                            "token": "secret-token",
                        },
                    },
                    {
                        "id": "competitor",
                        "role": "button",
                        "label": "Share",
                        "bounds": {"x": 300, "y": 50, "width": 100, "height": 40},
                        "visibility_fraction": 1,
                        "actionable": True,
                        "disabled": False,
                        "region_id": "team",
                    },
                ],
                "regions": [{"id": "team", "label": "Team"}],
            },
        },
        {
            "sequence": 2,
            "kind": "observation-recorded",
            "observation": {
                "viewport_id": "viewport-1",
                "newly_revealed_elements": [
                    {
                        "id": "target",
                        "role": "button",
                        "label": '<img src=x onerror="alert(1)">',
                        "bounds": {"x": 40, "y": 50, "width": 180, "height": 40},
                        "visibility_fraction": 1,
                        "actionable": True,
                        "disabled": False,
                        "region_id": "team",
                    }
                ],
                "remembered_elements": [],
                "region_context": {"id": "team", "label": "Team"},
            },
        },
        {
            "sequence": 3,
            "kind": "prominence-recorded",
            "viewport_id": "viewport-1",
            "scores": [
                {
                    "element_id": "target",
                    "raw_score": 0.2,
                    "normalized_probability": 0.3,
                    "feature_contributions": {"area": 0.1, "contrast": 0.2},
                    "raw_values": {"area": 7200, "contrast": 4.5},
                    "normalized_values": {"area": 0.4, "contrast": 0.8},
                }
            ],
        },
        {
            "sequence": 4,
            "kind": "coarse-scent-recorded",
            "scores": [{"element_id": "target", "score": 0.4}],
        },
        {
            "sequence": 5,
            "kind": "full-scent-recorded",
            "scores": [{"element_id": "target", "score": 0.6}],
        },
        {
            "sequence": 6,
            "kind": "action-proposed",
            "action": {"kind": "interact-with-element", "element_id": "target"},
            "reason": "Target matches goal.",
        },
        {
            "sequence": 7,
            "kind": "action-executed",
            "action": {"kind": "interact-with-element", "element_id": "target"},
            "succeeded": True,
            "viewport_id": "viewport-1",
            "execution_reference": {"token": "secret-token"},
        },
        {
            "sequence": 8,
            "kind": "verification-recorded",
            "result": {
                "verified": is_verified,
                "evidence_ids": ["verify-1"],
                "details": (
                    "Independent verifier passed."
                    if is_verified
                    else "Independent verifier did not confirm completion."
                ),
            },
        },
        {
            "sequence": 9,
            "kind": "model-call-recorded",
            "record": {
                "role": "cognitive",
                "model": "model-v1",
                "endpoint_origin": "https://llm.example.test",
                "prompt_digest": "prompt-sha",
                "schema_version": "cognitive-v1",
                "attempts": 2,
                "latency_ms": 125,
                "token_usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "total_tokens": 19,
                },
                "request": {"messages": [{"role": "user", "content": "safe request"}]},
                "response": {"summary": "safe response"},
                "retries": [{"attempt": 1, "reason": "rate-limit"}],
            },
        },
        {
            "sequence": 10,
            "kind": "run-terminated",
            "outcome": {"kind": outcome},
        },
    ]
    if event_overrides:
        for index, event in enumerate(events):
            sequence = event.get("sequence")
            if isinstance(sequence, int) and sequence in event_overrides:
                events[index] = {"sequence": sequence, **event_overrides[sequence]}
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_json(
        run / "result.json",
        {
            "run_id": run_id,
            "spec": {
                "scenario": {
                    "id": "invite",
                    "name": "Invite",
                    "goal": "Invite a teammate to the workspace",
                },
                "application_version": {
                    "id": version,
                    "label": version.title(),
                },
                "persona": {
                    "id": "persona",
                    "name": "Workspace administrator",
                },
            },
            "agent_claimed_success": True,
            "outcome": {"kind": outcome},
            "terminal_reason": terminal_reason,
            "evaluation_failure_reason": evaluation_failure_reason,
            "ux_sample_valid": is_valid_sample,
            "ux_sample_invalid_reason": ux_sample_invalid_reason,
            "evidence": {
                "prominence": [],
                "scent": [],
                "selections": [],
                "decisions": [],
                "model_calls": [],
                "screenshot_artifacts": [],
            },
            "metrics": {
                "run_id": run_id,
                "scenario_id": "invite",
                "application_version_id": version,
                "persona_id": "persona",
                "policy": "progressive-prominence-scent",
                "model_trial": model_trial,
                "prominence_provider_id": prominence_provider_id,
                "comparison_valid": is_valid_sample,
                "prominence_fallback": False,
                "prominence_fallback_reason": None,
                "reproducibility": "model-dependent",
                "verified_completion": is_verified,
                "wrong_actions": 0 if version == "improved" else 2,
                "backtracks": 0,
                "discovery_cost": {"total": discovery_cost},
                "evidence": [
                    {
                        "evidence_id": f"{run_id}:discovery-cost",
                        "evidence_class": "model-estimate",
                        "description": "Seeded discovery cost.",
                    },
                    {
                        "evidence_id": f"{run_id}:human",
                        "evidence_class": "unsupported-human-claim",
                        "description": "People will love it.",
                    },
                ],
            },
            "findings": [
                {
                    "finding_id": f"{run_id}:weak-scent",
                    "category": "weak-scent",
                    "title": "Target wording gives weak goal cues",
                    "cause": "Target scent 0.2 is below configured threshold 0.3.",
                    "severity": "medium",
                    "reproducibility": "model-dependent",
                    "evidence_class": "model-estimate",
                    "evidence_ids": [f"{run_id}:discovery-cost"],
                    "limitations": ["simulated benchmark evidence"],
                    "run_ids": [run_id],
                    "viewport_ids": ["viewport-1"],
                    "element_ids": ["target"],
                    "supporting_metrics": {"target-scent": 0.2},
                    "action_sequence": ["interact-with-element target: succeeded"],
                    "replay_links": [f"#run={run_id}&element=target"],
                }
            ],
            "limitations": ["Simulated benchmark; not human satisfaction evidence."],
        },
    )
    _write_checksums(run)


def test_renderer_loads_accepted_synthesis_and_maps_safe_evidence_targets(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    refs = (
        _synthesis_ref("event"),
        _synthesis_ref("replay"),
        _synthesis_ref("viewport"),
        _synthesis_ref("element", element_id="target"),
        _synthesis_ref("metric", viewport_id=None, metric_id="discovery-cost"),
    )
    _write_synthesis(tmp_path, finding_refs=refs)

    context = renderer._report_context(renderer._load_experiment(tmp_path))
    synthesis = context["synthesis"]

    assert synthesis["synthesis_status"] == "accepted"
    assert synthesis["using_fallback"] is False
    assert synthesis["findings"][0]["title"] == "Accepted synthesis finding"
    targets = synthesis["findings"][0]["evidence_targets"]
    assert {target["kind"] for target in targets} == {
        "event",
        "replay",
        "viewport",
        "element",
        "metric",
    }
    assert (
        next(target for target in targets if target["kind"] == "event")["sequence"] == 7
    )
    assert (
        next(target for target in targets if target["kind"] == "element")["element_id"]
        == "target"
    )
    assert "artifact_path" not in json.dumps(synthesis)
    assert "attachment_path" not in json.dumps(synthesis)


@pytest.mark.parametrize(
    ("kind", "filename"),
    [("heatmap", "3s-heatmap.png"), ("native-map", "3s.npz")],
)
def test_renderer_rejects_mismatched_synthesis_saliency_artifact_digest(
    tmp_path: Path,
    kind: str,
    filename: str,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_saliency_replay_evidence(tmp_path, "run-1")
    relative_artifact = f"runs/run-1/saliency/inference-1/{filename}"
    reference = EvidenceRef(
        f"{kind}:run-1:inference-1:3s",
        kind,
        "run-1",
        viewport_id="inference-1",
        artifact_path=relative_artifact,
        sha256="0" * 64,
    )
    _write_synthesis(
        tmp_path,
        corpus_refs=(reference,),
        finding_refs=(reference,),
        finding_title="Mismatched saliency digest must not render",
    )

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True
    assert all(
        finding["title"] != "Mismatched saliency digest must not render"
        for finding in synthesis["findings"]
    )


def test_renderer_rejects_model_estimate_target_with_unrecorded_element(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    reference = EvidenceRef(
        "model-estimate:run-1:prominence:3:forged",
        "model-estimate",
        "run-1",
        viewport_id="viewport-1",
        element_id="forged",
        event_id="event-3",
    )
    _write_synthesis(
        tmp_path,
        corpus_refs=(reference,),
        finding_refs=(reference,),
        finding_title="Forged model estimate must not render",
    )

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True
    assert all(
        finding["title"] != "Forged model estimate must not render"
        for finding in synthesis["findings"]
    )


def test_renderer_rejects_ranked_element_target_with_unrecorded_element(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_saliency_replay_evidence(tmp_path, "run-1")
    reference = EvidenceRef(
        "ranked-element:run-1:inference-1:3s:forged",
        "ranked-element",
        "run-1",
        viewport_id="inference-1",
        element_id="forged",
    )
    _write_synthesis(
        tmp_path,
        corpus_refs=(reference,),
        finding_refs=(reference,),
        finding_title="Forged ranked element must not render",
    )

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True
    assert all(
        finding["title"] != "Forged ranked element must not render"
        for finding in synthesis["findings"]
    )


def test_renderer_rejects_unsupported_target_fields(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    reference = EvidenceRef(
        "scenario:run-1",
        "scenario",
        "run-1",
        viewport_id="forged-viewport",
    )
    _write_synthesis(
        tmp_path,
        corpus_refs=(reference,),
        finding_refs=(reference,),
        finding_title="Unsupported target fields must not render",
    )

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True
    assert all(
        finding["title"] != "Unsupported target fields must not render"
        for finding in synthesis["findings"]
    )


@pytest.mark.parametrize("attempt_status", [None, SynthesisStatus.UNAVAILABLE])
def test_renderer_missing_or_unavailable_synthesis_uses_deterministic_fallback(
    tmp_path: Path,
    attempt_status: SynthesisStatus | None,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    if attempt_status is not None:
        _write_synthesis(tmp_path, status=attempt_status)

    context = renderer._report_context(renderer._load_experiment(tmp_path))
    synthesis = context["synthesis"]

    assert synthesis["synthesis_status"] == (
        "missing" if attempt_status is None else "unavailable"
    )
    assert synthesis["using_fallback"] is True
    finding = next(
        finding
        for finding in synthesis["findings"]
        if finding["finding_id"] == "run-1:weak-scent"
    )
    assert finding["evidence_refs"] == [
        {
            "evidence_id": "run-1:discovery-cost",
            "available": True,
            "target": {
                "kind": "metric",
                "run_id": "run-1",
                "metric_id": "discovery-cost",
            },
        }
    ]
    assert finding["evidence_targets"] == [
        {
            "kind": "metric",
            "run_id": "run-1",
            "metric_id": "discovery-cost",
        }
    ]


def test_renderer_split_fallback_preserves_scope_findings_and_limitations(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)

    experiment = renderer._load_experiment(tmp_path)
    index_context = renderer._report_context(
        experiment,
        include_run_payload=False,
        run_links={"run-1": "report-runs/run-1.html"},
    )
    run_context = renderer._report_context(
        experiment,
        run_links={"run-1": "report-runs/run-1.html"},
        run_scope=frozenset({"run-1"}),
    )

    assert index_context["synthesis"]["assessment"]
    assert index_context["synthesis"]["tested_scope"] == {
        "run_ids": ["run-1"],
        "scenario_ids": ["invite"],
        "version_ids": ["defective"],
        "persona_ids": ["persona"],
    }
    assert index_context["synthesis"]["limitations"]
    assert index_context["synthesis"]["findings"]
    assert run_context["synthesis"]["findings"]


@pytest.mark.parametrize(
    "status",
    (SynthesisStatus.ACCEPTED, SynthesisStatus.NO_ISSUES),
)
def test_renderer_rejects_selected_synthesis_for_stale_run_set(
    tmp_path: Path,
    status: SynthesisStatus,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_synthesis(tmp_path, status=status)
    _write_run(tmp_path, "run-2", version="improved", discovery_cost=2)

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True


def test_renderer_rejects_selected_synthesis_for_changed_run_identity(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_synthesis(tmp_path, status=SynthesisStatus.NO_ISSUES)
    run = tmp_path / "runs" / "run-1"
    result_path = run / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["spec"]["application_version"]["id"] = "improved"
    result["spec"]["application_version"]["label"] = "Improved"
    result["metrics"]["application_version_id"] = "improved"
    _write_json(result_path, result)
    _write_checksums(run)

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True


@pytest.mark.parametrize(
    "status",
    (SynthesisStatus.ACCEPTED, SynthesisStatus.NO_ISSUES),
)
def test_renderer_rejects_legacy_selected_synthesis_without_run_identities(
    tmp_path: Path,
    status: SynthesisStatus,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_synthesis(tmp_path, status=status, include_scope_identity=False)

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True


@pytest.mark.parametrize(
    "invalid_state",
    (
        "accepted-empty",
        "no-issues-candidates",
        "no-issues-blocker",
        "duplicate-final-ids",
        "final-absent-from-candidates",
        "duplicate-objection-ids",
        "orphan-objection",
        "forged-resolved-blocker",
        "changed-core-claim",
        "missing-candidate-evidence",
        "boolean-replay-sequence",
    ),
)
def test_renderer_rejects_selected_synthesis_with_hostile_publication_state(
    tmp_path: Path,
    invalid_state: str,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    references = (
        _synthesis_ref("event", sequence=1, viewport_id=None),
        _synthesis_ref("replay"),
    )
    _write_synthesis(
        tmp_path,
        status=SynthesisStatus.ACCEPTED,
        corpus_refs=references,
        finding_refs=references,
    )
    baseline = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]
    assert baseline["synthesis_status"] == "accepted"

    def make_hostile(value: dict[str, object]) -> None:
        if invalid_state == "accepted-empty":
            value["final_findings"] = []
        elif invalid_state == "no-issues-candidates":
            value["status"] = "no-issues"
            value["final_findings"] = []
        elif invalid_state == "no-issues-blocker":
            value["status"] = "no-issues"
            value["candidates"] = []
            value["final_findings"] = []
            value["objections"] = [
                {
                    "objection_id": "blocking-objection",
                    "finding_id": "synthesis-finding",
                    "severity": "blocking",
                    "message": "Recorded evidence contradicts publication.",
                    "evidence_refs": [],
                    "reviewer_role": "report-evidence-auditor",
                    "resolved": False,
                    "resolution": None,
                    "resolved_by_role": None,
                    "resolution_evidence_refs": [],
                }
            ]
        elif invalid_state == "duplicate-final-ids":
            value["final_findings"].append(dict(value["final_findings"][0]))
        elif invalid_state == "final-absent-from-candidates":
            value["candidates"] = []
        elif invalid_state in {"duplicate-objection-ids", "orphan-objection"}:
            objection = {
                "objection_id": "material-objection",
                "finding_id": (
                    "unknown-finding"
                    if invalid_state == "orphan-objection"
                    else "synthesis-finding"
                ),
                "severity": "material",
                "message": "The severity needs review.",
                "evidence_refs": [],
                "reviewer_role": "report-evidence-auditor",
                "resolved": False,
                "resolution": None,
                "resolved_by_role": None,
                "resolution_evidence_refs": [],
            }
            value["objections"] = (
                [objection, dict(objection)]
                if invalid_state == "duplicate-objection-ids"
                else [objection]
            )
        elif invalid_state == "forged-resolved-blocker":
            value["objections"] = [
                {
                    "objection_id": "blocking-objection",
                    "finding_id": "synthesis-finding",
                    "severity": "blocking",
                    "message": "Recorded evidence contradicts publication.",
                    "evidence_refs": [],
                    "reviewer_role": "report-evidence-auditor",
                    "resolved": True,
                    "resolution": "The adjudicator resolved the objection.",
                    "resolved_by_role": "report-adjudicator",
                    "resolution_evidence_refs": [
                        {
                            "evidence_id": "event:run-1:999",
                            "kind": "event",
                            "run_id": "run-1",
                            "viewport_id": None,
                            "element_id": None,
                            "event_id": "event-999",
                            "metric_id": None,
                            "artifact_path": None,
                            "replay_sequence": 999,
                            "sha256": None,
                        }
                    ],
                }
            ]
        elif invalid_state == "changed-core-claim":
            value["final_findings"][0]["issue"] = (
                "A corrupted artifact replaced the reviewed claim."
            )
        elif invalid_state == "boolean-replay-sequence":
            value["final_findings"][0]["evidence_refs"][0]["replay_sequence"] = True
        else:
            value["final_findings"][0]["evidence_refs"] = value["final_findings"][0][
                "evidence_refs"
            ][:1]

    _rewrite_selected_synthesis(tmp_path, make_hostile)

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True


@pytest.mark.parametrize(
    ("kind", "relative_path", "viewport_id", "max_bytes"),
    (
        ("screenshot", "artifacts/screenshot.png", "viewport-1", 16 * 1024 * 1024),
        (
            "heatmap",
            "saliency/inference-1/3s-heatmap.png",
            "inference-1",
            8 * 1024 * 1024,
        ),
        ("native-map", "saliency/inference-1/3s.npz", "inference-1", 64 * 1024 * 1024),
    ),
)
def test_renderer_rejects_oversized_synthesis_visual_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    relative_path: str,
    viewport_id: str,
    max_bytes: int,
) -> None:
    provider = "heuristic" if kind == "screenshot" else "foveacast"
    _write_run(
        tmp_path,
        "run-1",
        version="defective",
        discovery_cost=8,
        prominence_provider_id=provider,
    )
    if kind != "screenshot":
        _write_saliency_replay_evidence(tmp_path, "run-1")
    digest = "0" * 64
    evidence_id = (
        f"screenshot:run-1:{digest}"
        if kind == "screenshot"
        else f"{kind}:run-1:{viewport_id}:3s"
    )
    reference = EvidenceRef(
        evidence_id,
        kind,
        "run-1",
        viewport_id=viewport_id,
        artifact_path=f"runs/run-1/{relative_path}",
        sha256=digest,
    )
    _write_synthesis(
        tmp_path,
        corpus_refs=(reference,),
        finding_refs=(reference,),
        finding_title="Oversized synthesis visual must not render",
    )
    loaded_run = renderer._load_run(tmp_path / "runs" / "run-1")
    artifact_path = tmp_path / "runs" / "run-1" / relative_path
    with artifact_path.open("wb") as handle:
        handle.truncate(max_bytes + 1)
    secure_read = renderer.secure_read_bytes
    observed_limits: list[int | None] = []

    def assert_bounded_read(
        path: Path,
        label: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        if path == artifact_path:
            observed_limits.append(max_bytes)
            assert max_bytes == expected_max_bytes
        return secure_read(path, label, max_bytes=max_bytes)

    expected_max_bytes = max_bytes
    monkeypatch.setattr(renderer, "secure_read_bytes", assert_bounded_read)

    synthesis, _ = renderer._load_synthesis(tmp_path, (loaded_run,))

    assert observed_limits == [expected_max_bytes]
    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True


def test_renderer_rejects_forged_synthesis_event_viewport(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    forged_ref = _synthesis_ref("event", viewport_id="forged-viewport")
    _write_synthesis(
        tmp_path,
        corpus_refs=(forged_ref,),
        finding_refs=(forged_ref,),
        finding_title="Forged viewport finding must not render",
    )

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True
    assert all(
        finding["title"] != "Forged viewport finding must not render"
        for finding in synthesis["findings"]
    )


def test_renderer_sorts_fallback_findings_by_severity() -> None:
    runs = (
        {
            "run_id": "run-1",
            "findings": [
                {"finding_id": "low", "severity": "low"},
                {"finding_id": "critical", "severity": "critical"},
                {"finding_id": "medium", "severity": "medium"},
                {"finding_id": "high", "severity": "high"},
            ],
        },
    )

    findings = renderer._deterministic_fallback_findings(runs)

    assert [finding["finding_id"] for finding in findings] == [
        "critical",
        "high",
        "medium",
        "low",
    ]


def test_renderer_scopes_no_issues_copy_to_tested_scenarios(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_synthesis(tmp_path, status=SynthesisStatus.NO_ISSUES)

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "no-issues"
    assert synthesis["using_fallback"] is False
    assert synthesis["findings"] == []
    assert synthesis["assessment"] == (
        "No supported UX issues were established in the tested scenarios."
    )
    assert "issue-free" not in synthesis["assessment"]


def test_renderer_rejects_forged_synthesis_evidence_references(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    valid_ref = _synthesis_ref("event")
    forged_ref = _synthesis_ref("event", sequence=999)
    _write_synthesis(
        tmp_path,
        corpus_refs=(valid_ref,),
        finding_refs=(forged_ref,),
        finding_title="Forged finding must not render",
    )

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True
    assert all(
        finding["title"] != "Forged finding must not render"
        for finding in synthesis["findings"]
    )


def test_renderer_rejects_synthesis_copied_beside_changed_finalized_bundle(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_synthesis(tmp_path)
    result_path = tmp_path / "runs" / "run-1" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["metrics"]["discovery_cost"] = 2
    _write_json(result_path, result)
    _write_checksums(result_path.parent)

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True
    assert all(
        finding["title"] != "Accepted synthesis finding"
        for finding in synthesis["findings"]
    )


def test_renderer_does_not_promote_rejected_attempt_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_synthesis(
        tmp_path,
        status=SynthesisStatus.REJECTED,
        corpus_refs=(_synthesis_ref("event"),),
        finding_refs=(_synthesis_ref("event"),),
        finding_title="Rejected attempt finding",
    )
    monkeypatch.setattr(renderer, "_deterministic_fallback_findings", lambda runs: [])

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "rejected"
    assert synthesis["using_fallback"] is True
    assert synthesis["findings"] == []
    assert all(
        finding["title"] != "Rejected attempt finding"
        for finding in synthesis["findings"]
    )

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )
    normalized_html = " ".join(html.split())

    assert 'data-synthesis-status="rejected"' in normalized_html
    assert "Model review rejected; recorded evidence available" in (normalized_html)
    assert "Recorded signals requiring manual review" in normalized_html
    assert "Priority findings" not in normalized_html
    assert "Check first" not in normalized_html
    assert "Candidate findings are not publishable." in normalized_html
    assert "the evidence review is unavailable" not in normalized_html


def test_renderer_uses_global_sequence_for_latest_unselected_attempt(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_synthesis(
        tmp_path,
        status=SynthesisStatus.UNAVAILABLE,
        sequence=1,
        corpus_marker="sequence-10",
        include_scope_identity=False,
        limitations=("Older unavailable attempt.",),
    )
    _write_synthesis(
        tmp_path,
        status=SynthesisStatus.REJECTED,
        sequence=2,
        corpus_marker="sequence-2",
        include_scope_identity=False,
        limitations=("Latest rejected attempt.",),
    )

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "rejected"
    assert synthesis["limitations"] == ["Latest rejected attempt."]


def test_renderer_does_not_promote_accepted_attempt_without_index_pointer(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_synthesis(tmp_path)
    index_path = tmp_path / "synthesis" / "index.json"
    index = json.loads(index_path.read_text(encoding="ascii"))
    index["accepted_attempt_id"] = None
    index_path.write_bytes(
        (
            json.dumps(index, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")
    )

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "unavailable"
    assert synthesis["using_fallback"] is True


def test_renderer_reads_lockless_external_synthesis_without_writing(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_synthesis(tmp_path)
    lock_path = tmp_path / "synthesis" / ".publication.lock"
    lock_path.unlink()

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "accepted"
    assert synthesis["using_fallback"] is False
    assert not lock_path.exists()


def test_renderer_reads_only_latest_unselected_attempt_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    for sequence in range(1, 13):
        _write_synthesis(
            tmp_path,
            status=SynthesisStatus.UNAVAILABLE,
            sequence=sequence,
            limitations=(f"Unavailable attempt {sequence}.",),
        )

    calls = {"attempt_ids": 0, "bundles": 0}
    original_attempt_ids = SynthesisArtifactStore._attempt_ids
    original_read_bundle = SynthesisArtifactStore._read_attempt_bundle

    def counted_attempt_ids(store: SynthesisArtifactStore) -> tuple[str, ...]:
        calls["attempt_ids"] += 1
        return original_attempt_ids(store)

    def counted_read_bundle(
        store: SynthesisArtifactStore,
        attempt_id: str,
    ) -> tuple[SynthesisAttempt, bytes, bytes]:
        calls["bundles"] += 1
        return original_read_bundle(store, attempt_id)

    monkeypatch.setattr(SynthesisArtifactStore, "_attempt_ids", counted_attempt_ids)
    monkeypatch.setattr(
        SynthesisArtifactStore,
        "_read_attempt_bundle",
        counted_read_bundle,
    )

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "unavailable"
    assert synthesis["limitations"] == ["Unavailable attempt 12."]
    assert calls == {"attempt_ids": 1, "bundles": 1}


@pytest.mark.parametrize("oversized_artifact", ("index", "corpus"))
def test_renderer_bounds_synthesis_json_reads_during_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    oversized_artifact: str,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_synthesis(tmp_path)
    store = SynthesisArtifactStore(tmp_path)
    attempt = store.accepted_attempt
    assert attempt is not None
    if oversized_artifact == "index":
        hostile_path = store.index_path
    else:
        hostile_path = store.attempts_root / attempt.attempt_id / "corpus-manifest.json"
        monkeypatch.setattr(
            SynthesisArtifactStore,
            "report_attempt",
            property(lambda self: attempt),
        )
    max_bytes = MAX_SYNTHESIS_JSON_BYTES
    with hostile_path.open("wb") as handle:
        handle.truncate(max_bytes + 1)
    secure_read = renderer.secure_read_bytes
    observed_limits: list[int | None] = []

    def assert_bounded_read(
        path: Path,
        label: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        if path == hostile_path:
            observed_limits.append(max_bytes)
            assert max_bytes == expected_max_bytes
        return secure_read(path, label, max_bytes=max_bytes)

    expected_max_bytes = max_bytes
    monkeypatch.setattr(renderer, "secure_read_bytes", assert_bounded_read)

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert synthesis["synthesis_status"] == "invalid"
    assert synthesis["using_fallback"] is True
    assert observed_limits
    assert set(observed_limits) == {max_bytes}


def test_renderer_labels_boundary_rejection_separately_from_unavailable(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)

    synthesis = renderer._fallback_synthesis(
        "rejected",
        renderer._load_experiment(tmp_path)["runs"],
        [],
        "A synthesis retrieval request could not be resolved through the evidence boundary.",
    )

    assert synthesis["model_review_status"] == "rejected"
    assert "evidence boundary" in synthesis["assessment"]
    assert "unavailable" not in synthesis["assessment"].casefold()


def test_renderer_preserves_boundary_limitation_from_rejected_attempt(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_synthesis(
        tmp_path,
        status=SynthesisStatus.REJECTED,
        limitations=("The synthesis evidence boundary rejected a retrieval request.",),
    )

    synthesis = renderer._report_context(renderer._load_experiment(tmp_path))[
        "synthesis"
    ]

    assert "evidence boundary" in synthesis["assessment"]
    assert "unavailable" not in synthesis["assessment"].casefold()


def test_render_experiment_report_is_offline_and_does_not_call_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)

    def fail_model_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("report rendering must not call a model")

    monkeypatch.setattr(
        renderer, "create_structured_model_client", fail_model_call, raising=False
    )
    monkeypatch.setattr(
        renderer, "ReportSynthesisService", fail_model_call, raising=False
    )

    output = render_experiment_report(tmp_path, tmp_path / "report.html")

    assert output.is_file()


def test_renderer_split_index_keeps_accepted_findings_and_counts_synthesis_bytes(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run.active", version="defective", discovery_cost=8)
    _write_run(tmp_path, "run_active", version="improved", discovery_cost=3)
    _write_synthesis(
        tmp_path,
        corpus_refs=(_synthesis_ref("event", run_id="run.active"),),
        finding_refs=(_synthesis_ref("event", run_id="run.active"),),
        run_id="run.active",
        scope_run_ids=("run.active", "run_active"),
    )

    experiment = renderer._load_experiment(tmp_path)
    without_artifact_bytes = {
        key: value
        for key, value in experiment.items()
        if key != "_synthesis_artifact_bytes"
    }
    assert renderer._estimated_full_report_bytes(
        experiment
    ) > renderer._estimated_full_report_bytes(without_artifact_bytes)

    output = render_experiment_report(
        tmp_path,
        tmp_path / "report.html",
        max_single_file_bytes=100,
    )
    html = output.read_text(encoding="utf-8")
    run_pages = tuple((tmp_path / "report-runs").glob("*.html"))

    assert "Accepted synthesis finding" in html
    assert len(run_pages) == 2
    assert all(path.name in html for path in run_pages)


def test_renderer_places_conclusions_before_comparison_and_orders_severity(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    findings = tuple(
        SynthesisFinding(
            finding_id=f"finding-{severity}",
            title=title,
            issue=f"Issue for {severity} priority.",
            impact=f"Impact for {severity} priority.",
            root_cause=f"Cause for {severity} priority.",
            fixes=(f"Fix {severity} priority first.",),
            severity=severity,
            confidence=0.9,
            evidence_refs=(ref,),
            affected_surfaces=("Invite flow",),
            principles=("mental-models",),
            counterevidence=(),
            limitations=("Fixture limitation.",),
            reviewer_state="accepted",
            severity_justification="Recorded event evidence supports this priority.",
        )
        for severity, title, ref in (
            ("low", "Low priority finding", _synthesis_ref("event", sequence=6)),
            (
                "critical",
                "Critical priority finding",
                _synthesis_ref("event", sequence=8),
            ),
            ("high", "High priority finding", _synthesis_ref("event", sequence=7)),
        )
    )
    _write_synthesis(tmp_path, findings=findings)

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    section_positions = [
        html.index('id="analysis-summary"'),
        html.index('id="priority-findings"'),
        html.index('id="fix-first"'),
        html.index('id="evidence-workspace"'),
        html.index('id="comparison-table"'),
    ]
    assert section_positions == sorted(section_positions)
    priority_html = html[
        html.index('id="priority-findings"') : html.index('id="fix-first"')
    ]
    assert priority_html.index("Critical priority finding") < priority_html.index(
        "High priority finding"
    )
    assert priority_html.index("High priority finding") < priority_html.index(
        "Low priority finding"
    )
    assert "Accepted findings" in html
    assert "Tested scope" in html
    assert "Evidence review complete" in html
    assert html.count("Verify evidence") == 3
    assert "Affected surfaces" in html
    assert "mental-models" in html


def test_renderer_exposes_no_issue_and_fallback_conclusion_states(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="improved", discovery_cost=3)
    _write_synthesis(tmp_path, status=SynthesisStatus.NO_ISSUES)

    no_issue_html = render_experiment_report(
        tmp_path, tmp_path / "no-issues.html"
    ).read_text(encoding="utf-8")

    assert "No supported UX issues were established in the tested scenarios." in (
        no_issue_html
    )
    assert "Accepted findings" in no_issue_html
    assert 'data-synthesis-status="no-issues"' in no_issue_html

    fallback_root = tmp_path / "fallback"
    _write_run(fallback_root, "run-1", version="defective", discovery_cost=8)
    fallback_html = render_experiment_report(
        fallback_root, fallback_root / "report.html"
    ).read_text(encoding="utf-8")

    assert "1 recorded signal is linked directly to evidence" in fallback_html
    assert 'data-synthesis-status="missing"' in fallback_html
    priority_html = fallback_html[
        fallback_html.index('id="priority-findings"') : fallback_html.index(
            'id="evidence-workspace"'
        )
    ]
    assert "Target wording gives weak goal cues" not in priority_html
    assert "Invite" in fallback_html
    assert "Invite a teammate to the workspace" in fallback_html
    assert "Workspace administrator" in fallback_html
    assert "run-1" in fallback_html
    assert "verified-success" in fallback_html
    assert "Model review unavailable; recorded evidence available" in fallback_html
    assert "Recorded signals requiring manual review" in fallback_html
    assert "Priority findings" not in fallback_html
    assert "Check first" in fallback_html
    assert "What to change" in fallback_html
    assert "Recorded High" not in fallback_html
    assert "Recorded Medium" not in fallback_html
    assert "in the linked replay and make its label or nearby cue name the goal" in (
        fallback_html
    )
    fallback_copy = " ".join(fallback_html.split()).lower()
    assert (
        "model review did not complete. recorded deterministic evidence remains available. "
        "ux principles, counterevidence, and reviewer status are unavailable."
        in fallback_copy
    )
    assert "Recorded deterministic evidence only" in fallback_html
    assert "1 recorded" in fallback_html
    assert (
        "Unavailable: model review did not complete, so synthesis principles were not recorded."
        in fallback_html
    )
    assert (
        "Unavailable: model review did not complete, so counterevidence was not recorded."
        in fallback_html
    )
    assert 'data-report-navigation="true"' in fallback_html
    assert "max-width: 1440px" in fallback_html
    assert "padding-inline: clamp(" in fallback_html
    assert 'data-evidence-target="{&#34;kind&#34;: &#34;metric&#34;' in fallback_html


@pytest.mark.parametrize("status", ("missing", "unavailable", "rejected", "invalid"))
def test_renderer_fallback_statuses_remove_accepted_priority_hierarchy(
    tmp_path: Path,
    status: str,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    if status == "unavailable":
        _write_synthesis(tmp_path, status=SynthesisStatus.UNAVAILABLE)
    elif status == "rejected":
        _write_synthesis(tmp_path, status=SynthesisStatus.REJECTED)
    elif status == "invalid":
        _write_synthesis(tmp_path)

        def invalidate_claim(value: dict[str, object]) -> None:
            findings = cast(list[dict[str, object]], value["final_findings"])
            findings[0]["issue"] = "Changed after review."

        _rewrite_selected_synthesis(tmp_path, invalidate_claim)

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert f'data-synthesis-status="{status}"' in html
    assert "Recorded signals requiring manual review" in html
    assert "Recorded signal" in html
    assert "Priority findings" not in html
    assert "Check first" in html


def test_renderer_omits_prescriptive_hierarchy_without_reviewed_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_run(tmp_path, "run-1", version="improved", discovery_cost=3)
    monkeypatch.setattr(renderer, "_deterministic_fallback_findings", lambda runs: [])

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )
    normalized_html = " ".join(html.split())

    assert "Recorded signals requiring manual review" in normalized_html
    assert "Fix first" not in normalized_html
    assert "Priority findings" not in normalized_html
    assert (
        "No fix is prioritized because no supported issue was established"
        not in normalized_html
    )


def test_renderer_fallback_finding_is_self_contained(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)

    context = renderer._report_context(renderer._load_experiment(tmp_path))
    finding = context["synthesis"]["findings"][0]

    assert finding["source"] == "deterministic-fallback"
    assert finding["fallback_context"] == {
        "run_id": "run-1",
        "scenario": "Invite",
        "persona": "Workspace administrator",
        "goal": "Invite a teammate to the workspace",
        "version": "Defective",
        "target": "target",
        "action": "interact-with-element on target: succeeded",
        "outcome": "verified-success",
        "verification": "verified",
    }
    assert "does not clearly signal the task goal" in finding["fallback_title"]
    assert "model review" in finding["fallback_issue"].lower()
    assert finding["evidence_refs"][0]["available"] is True


def test_renderer_omits_fallback_claim_without_resolvable_evidence(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    runs = renderer._load_experiment(tmp_path)["runs"]
    runs[0]["findings"][0]["evidence_ids"] = ["unknown-evidence"]

    synthesis = renderer._fallback_synthesis(
        "missing",
        runs,
        renderer._deterministic_fallback_findings(runs),
        "No persisted report synthesis is available.",
    )

    assert synthesis["findings"] == []
    assert any(
        "omitted because no canonical evidence target could be resolved" in limitation
        for limitation in synthesis["limitations"]
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_renderer_browser_fallback_navigation_context_and_evidence_on_mobile(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    report_path = render_experiment_report(tmp_path, tmp_path / "report.html")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto(report_path.resolve().as_uri())

        assert await page.locator('[data-report-navigation="true"]').count() == 1
        assert "Model review unavailable" in (
            await page.locator("#analysis-summary").text_content() or ""
        )
        assert "Invite a teammate to the workspace" in (
            await page.locator("#priority-findings").text_content() or ""
        )
        assert await page.locator('a[href="#playback-workspace"]').count() == 1

        await page.locator("summary", has_text="Verify evidence").first.press("Enter")
        await page.locator('[data-evidence-id="run-1:discovery-cost"]').click()
        assert "evidence=run-1%3Adiscovery-cost" in page.url
        metric = page.locator('tr[data-run-id="run-1"] [data-metric="discovery-cost"]')
        assert await metric.get_attribute("data-viewing-evidence") == "true"
        assert await metric.evaluate("node => document.activeElement === node")
        assert await page.locator("#evidence-context").text_content() == (
            "Evidence opened in the workspace below."
        )
        assert page.url.endswith("#comparison-overview")

        await page.set_viewport_size({"width": 360, "height": 800})
        dimensions = await page.evaluate(
            "({scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth})"
        )
        assert dimensions["scrollWidth"] <= dimensions["innerWidth"]
        assert await page.locator('[data-report-navigation="true"]').evaluate(
            "node => node.getBoundingClientRect().right <= window.innerWidth"
        )
        await browser.close()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_renderer_browser_preserves_deep_link_state_on_reload_and_back(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    _write_synthesis(tmp_path, finding_refs=(_synthesis_ref("event"),))
    report_path = render_experiment_report(tmp_path, tmp_path / "report.html")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(report_path.resolve().as_uri() + "#analysis-summary")
        await page.locator("summary", has_text="Verify evidence").click()
        await page.locator('[data-evidence-id="event:run-1:7"]').click()

        assert page.url.endswith("#playback-workspace")
        assert "evidence=event%3Arun-1%3A7" in page.url
        assert "event=event-7" in page.url
        assert await page.locator("#current-event-card").evaluate(
            "node => document.activeElement === node"
        )
        await page.reload()
        assert "Event 7 /" in (
            await page.locator("#playback-position").text_content() or ""
        )
        assert await page.locator("#evidence-context").text_content() == (
            "Evidence opened in the workspace below."
        )
        await page.go_back()
        assert page.url.endswith("#analysis-summary")
        assert "Event 1 /" in (
            await page.locator("#playback-position").text_content() or ""
        )
        assert await page.locator("#evidence-context").is_hidden()
        await page.go_forward()
        assert page.url.endswith("#playback-workspace")
        assert "Event 7 /" in (
            await page.locator("#playback-position").text_content() or ""
        )
        assert await page.locator("#evidence-context").text_content() == (
            "Evidence opened in the workspace below."
        )
        await browser.close()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_renderer_browser_routes_non_replay_evidence_to_exact_detail(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    references = (
        EvidenceRef("verification:run-1", "verification", "run-1"),
        EvidenceRef("expectation:run-1", "expectation", "run-1"),
        EvidenceRef(
            "model-estimate:run-1:prominence:3:target",
            "model-estimate",
            "run-1",
            viewport_id="viewport-1",
            element_id="target",
            event_id="event-3",
        ),
    )
    _write_synthesis(tmp_path, corpus_refs=references, finding_refs=references)
    report_path = render_experiment_report(tmp_path, tmp_path / "report.html")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(report_path.resolve().as_uri())
        await page.locator("summary", has_text="Verify evidence").click()

        for evidence_id in ("verification:run-1", "expectation:run-1"):
            await page.locator(f'[data-evidence-id="{evidence_id}"]').click()
            detail = page.locator("#evidence-detail")
            assert await detail.get_attribute("data-evidence-id") == evidence_id
            assert await detail.evaluate("node => document.activeElement === node")
            assert page.url.endswith("#evidence-detail")

        await page.locator(
            '[data-evidence-id="model-estimate:run-1:prominence:3:target"]'
        ).click()
        assert "Event 3 /" in (
            await page.locator("#playback-position").text_content() or ""
        )
        assert (
            await page.locator("#selected-element-evidence").get_attribute(
                "data-selected-element-id"
            )
            == "target"
        )
        await browser.close()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_renderer_browser_keyboard_tabs_rankings_and_table_semantics(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-1",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-1")
    report_path = render_experiment_report(tmp_path, tmp_path / "report.html")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(report_path.resolve().as_uri())

        row = page.locator('tr[data-run-id="run-1"]')
        assert await row.get_attribute("role") is None
        assert await row.get_attribute("tabindex") is None
        assert await row.get_by_role("link", name="Open replay").count() == 1

        tabs = page.get_by_role("tab")
        assert await tabs.count() == 3
        assert await tabs.nth(0).get_attribute("tabindex") == "0"
        assert await tabs.nth(1).get_attribute("tabindex") == "-1"
        assert await tabs.nth(0).get_attribute("aria-controls") == "saliency-detail"
        assert (
            await page.locator("#saliency-detail").get_attribute("role") == "tabpanel"
        )
        await tabs.nth(0).focus()
        await tabs.nth(0).press("ArrowRight")
        assert await tabs.nth(1).get_attribute("aria-selected") == "true"
        assert await tabs.nth(1).evaluate("node => document.activeElement === node")
        await tabs.nth(1).press("End")
        assert await tabs.nth(2).get_attribute("aria-selected") == "true"

        ranked = page.locator(".ranked-element-button")
        assert await ranked.count() >= 1
        await ranked.first.focus()
        await ranked.first.press("Space")
        assert (
            await page.locator("#selected-element-evidence").get_attribute(
                "data-selected-element-id"
            )
            == "target"
        )
        await browser.close()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_renderer_browser_mobile_workspace_is_reachable_without_overflow(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-1", version="defective", discovery_cost=8)
    report_path = render_experiment_report(tmp_path, tmp_path / "report.html")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(report_path.resolve().as_uri())

        for width in (320, 360, 390, 414):
            await page.set_viewport_size({"width": width, "height": 720})
            dimensions = await page.evaluate(
                "({scrollWidth: document.documentElement.scrollWidth, "
                "innerWidth: window.innerWidth})"
            )
            offenders = await page.evaluate(
                "Array.from(document.querySelectorAll('*')).map(node => ({"
                "tag: node.tagName, id: node.id, className: node.className, "
                "right: node.getBoundingClientRect().right, width: node.scrollWidth"
                "})).filter(item => item.right > window.innerWidth || "
                "item.width > window.innerWidth).sort((left, right) => "
                "right.right - left.right).slice(0, 12)"
            )
            assert dimensions["scrollWidth"] <= dimensions["innerWidth"], offenders
            assert await page.locator("#selected-element-evidence").evaluate(
                "node => getComputedStyle(node).overflowY === 'visible'"
            )
            assert await page.locator("#report-limitations").count() == 0
            await page.locator("#playback-workspace").scroll_into_view_if_needed()
            assert await page.locator("#playback-workspace").is_visible()
        await browser.close()


def test_renderer_no_issues_lists_named_scope_with_run_links(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-1", version="improved", discovery_cost=3)
    _write_synthesis(tmp_path, status=SynthesisStatus.NO_ISSUES)

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )
    scope = html[
        html.index('data-no-issues-scope="true"') : html.index('id="priority-findings"')
    ]

    assert "Invite / Improved / Workspace administrator" in scope
    assert "Invite [invite]" not in scope
    assert ">run-1<" not in scope
    assert 'href="?run=run-1#playback-workspace"' in scope


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_renderer_browser_opens_finding_evidence_and_wraps_on_mobile(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-1",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-1")
    _write_synthesis(
        tmp_path,
        finding_refs=(_synthesis_ref("event"), _synthesis_ref("heatmap")),
    )
    report_path = render_experiment_report(tmp_path, tmp_path / "report.html")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto(report_path.resolve().as_uri())
        await page.screenshot(path=str(tmp_path / "report-desktop.png"))
        await page.screenshot(path=str(tmp_path / "report-full.png"), full_page=True)

        verify = page.locator("summary", has_text="Verify evidence").first
        await verify.focus()
        await verify.press("Enter")
        assert await page.locator(".finding-verification[open]").count() == 1

        event_ref = page.locator('[data-evidence-id="event:run-1:7"]')
        await event_ref.click()
        assert "Event 7 /" in (
            await page.locator("#playback-position").text_content() or ""
        )

        heatmap_ref = page.locator('[data-evidence-id="heatmap:run-1:viewport-1:3s"]')
        await heatmap_ref.click()
        assert "run=run-1" in page.url
        assert "evidence=heatmap%3Arun-1%3Aviewport-1%3A3s" in page.url
        assert (
            await page.locator('#saliency-tabs [aria-selected="true"]').text_content()
            == "3s"
        )
        await page.screenshot(path=str(tmp_path / "report-evidence.png"))

        await page.set_viewport_size({"width": 360, "height": 800})
        await page.evaluate("window.scrollTo(0, 0)")
        await page.screenshot(path=str(tmp_path / "report-mobile.png"))
        dimensions = await page.evaluate(
            "({scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth})"
        )
        assert dimensions["scrollWidth"] <= dimensions["innerWidth"]
        assert await heatmap_ref.evaluate(
            "node => node.getBoundingClientRect().right <= window.innerWidth"
        )
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.evaluate("window.scrollTo(0, 0)")
        await page.screenshot(path=str(tmp_path / "report-mobile-390.png"))
        await browser.close()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_renderer_split_index_evidence_button_opens_run_page(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run.active", version="defective", discovery_cost=8)
    _write_run(tmp_path, "run_active", version="improved", discovery_cost=3)
    _write_synthesis(
        tmp_path,
        finding_refs=(_synthesis_ref("event", run_id="run.active"),),
        run_id="run.active",
        scope_run_ids=("run.active", "run_active"),
    )
    report_path = render_experiment_report(
        tmp_path, tmp_path / "report.html", max_single_file_bytes=100
    )

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(report_path.resolve().as_uri())
        await page.locator("summary", has_text="Verify evidence").first.click()
        evidence_button = page.locator('[data-evidence-id="event:run.active:7"]')
        assert await evidence_button.text_content() == "Show on screenshot"
        await evidence_button.click()
        assert "report-runs" in page.url
        assert "run.active" in page.url or "run.active-" in page.url
        assert "evidence=event%3Arun.active%3A7" in page.url
        assert page.url.endswith("#playback-workspace")
        await browser.close()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_renderer_split_index_populates_controls_and_opens_replay(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run.active", version="defective", discovery_cost=8)
    _write_run(tmp_path, "run_active", version="improved", discovery_cost=3)
    report_path = render_experiment_report(
        tmp_path, tmp_path / "report.html", max_single_file_bytes=100
    )

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(report_path.resolve().as_uri())
        assert await page.locator("#scenario-select option").count() > 1
        assert await page.locator("#run-select option").count() == 2
        run_labels = await page.locator("#run-select option").all_text_contents()
        assert all("run.active" not in label for label in run_labels)
        await page.locator("#play-pause").click()
        assert "report-runs" in page.url
        assert page.url.endswith("#playback-workspace")
        await browser.close()


def test_renderer_embeds_sanitized_replay_evidence_and_controls(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "run-1",
        version="defective",
        discovery_cost=8,
    )

    output = render_experiment_report(tmp_path, tmp_path / "report.html")

    html = output.read_text(encoding="utf-8")
    assert output == tmp_path / "report.html"
    assert "Comparison overview" in html
    assert "User actions" in html
    assert "Observations" in html
    assert "Discovery cost" in html
    assert "deterministic-fact" in html
    assert "model-estimate" in html
    assert "unsupported-human-claim" in html
    assert "Run workspace" in html
    assert "Recorded timeline" in html
    assert "play-pause" in html
    assert "restart-playback" in html
    assert "Prominence contributions" in html
    assert "Element evidence" in html
    assert 'data-metric="model-trial">2</td>' in html
    assert "model-dependent (attention seed 7, model trial 2)" in html
    assert "Observation" in html
    assert "Terminal / failure" in html
    assert "Scent" in html
    assert "Model decision and reason" in html
    assert "Action and result" in html
    assert "Verification" in html
    assert "Memory" in html
    assert "Prompt version" in html
    assert "Schema version" in html
    assert "Sanitized request summary" in html
    assert '"bundle_path":"runs\\u002frun-1"' in html
    assert '"integrity_status":"trusted"' in html
    assert "Trust boundaries and limitations" not in html
    assert 'href="#report-limitations"' not in html
    assert '"width":800' in html
    assert "secret-token" not in html
    assert "data-testid=secret" not in html
    assert "<img src=x onerror" not in html
    assert "fetch(" not in html
    assert '<link rel="stylesheet"' not in html
    assert "<script src=" not in html


def test_renderer_distinguishes_model_trials_in_run_aggregate_and_gate_views(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-defective",
        version="defective",
        discovery_cost=8,
        model_trial=2,
    )
    _write_run(
        tmp_path,
        "run-improved",
        version="improved",
        discovery_cost=3,
        model_trial=2,
    )

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert html.count('data-metric="model-trial">2</td>') >= 4
    assert html.count("model-dependent (attention seed 7, model trial 2)") >= 2


def test_renderer_keeps_prominence_providers_separate_in_rows_and_gates(
    tmp_path: Path,
) -> None:
    for provider in ("heuristic", "foveacast"):
        _write_run(
            tmp_path,
            f"run-defective-{provider}",
            version="defective",
            discovery_cost=8,
            prominence_provider_id=provider,
        )
        _write_run(
            tmp_path,
            f"run-improved-{provider}",
            version="improved",
            discovery_cost=3,
            prominence_provider_id=provider,
        )

    experiment = renderer._load_experiment(tmp_path)

    assert {run["prominence_provider_id"] for run in experiment["runs"]} == {
        "heuristic",
        "foveacast",
    }
    assert {row["prominence_provider_id"] for row in experiment["run_rows"]} == {
        "heuristic",
        "foveacast",
    }
    assert {row["prominence_provider_id"] for row in experiment["comparison_rows"]} == {
        "heuristic",
        "foveacast",
    }
    assert {row["prominence_provider_id"] for row in experiment["gate_rows"]} == {
        "heuristic",
        "foveacast",
    }

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )
    assert html.count('data-metric="prominence-provider">foveacast</td>') >= 3


def test_renderer_pairs_provider_heatmaps_and_recorded_action_paths(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-heuristic",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="heuristic",
    )
    _write_run(
        tmp_path,
        "run-foveacast",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-foveacast")

    experiment = renderer._load_experiment(tmp_path)
    comparisons = experiment["provider_comparisons"]

    assert len(comparisons) == 1
    providers = {item["provider_id"]: item for item in comparisons[0]["providers"]}
    assert set(providers) == {"heuristic", "foveacast"}
    assert providers["foveacast"]["heatmaps"][0]["heatmap"].startswith(
        "data:image/png;base64,"
    )
    assert providers["foveacast"]["heatmaps"][0]["heatmap_path"].endswith(
        "1s-heatmap.png"
    )
    assert providers["heuristic"]["heatmaps"] == []
    assert providers["heuristic"]["prominence"][0]["rankings"][0]["label"]
    assert providers["heuristic"]["action_path"][0]["element_label"]

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )
    assert "Provider method comparison" in html
    assert "Open exact generated heatmap" in html
    assert "Recorded action path" in html


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_renderer_split_index_populates_provider_comparison(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-heuristic",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="heuristic",
    )
    _write_run(
        tmp_path,
        "run-foveacast",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-foveacast")
    report_path = render_experiment_report(
        tmp_path, tmp_path / "report.html", max_single_file_bytes=100
    )

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(report_path.resolve().as_uri())
        assert await page.locator("#provider-comparison-select option").count() == 1
        assert await page.locator("#provider-comparison-duration option").count() > 0
        assert (
            await page.locator("#provider-comparison-output .provider-card").count()
            == 2
        )
        await browser.close()


def test_overview_counts_executed_actions_and_formats_discovery_cost(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-effort",
        version="improved",
        discovery_cost=4.199999999999999,
    )
    run = tmp_path / "runs" / "run-effort"
    events = [
        json.loads(line)
        for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events.insert(
        -1,
        {
            "sequence": 10,
            "kind": "decision-recorded",
            "action": {"kind": "interact-with-element", "element_id": "target"},
            "reason": "Duplicate representation of the proposed action.",
        },
    )
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    _write_checksums(run)

    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert 'data-metric="user-actions">1</td>' in html
    assert 'data-metric="observations">1</td>' in html
    assert 'data-metric="discovery-cost">4.2</td>' in html
    assert 'data-metric="estimated-task-seconds">2.5 s</td>' in html
    assert 'data-metric="model-calls">1</td>' in html
    assert 'data-metric="analysis-latency">125 ms</td>' in html
    assert 'data-metric="analysis-tokens">19</td>' in html
    assert "Estimated task time combines recorded observations and actions" in html
    assert "Model processing cost is shown separately" in html


def test_renderer_replays_saliency_profiles_and_separates_inference_cost(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-saliency",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-saliency")

    experiment = renderer._load_experiment(tmp_path)
    run = experiment["runs"][0]

    assert run["saliency_runtime"]["total_inference_ms"] == pytest.approx(6.0)
    assert run["analysis_cost"]["latency_ms"] == 125
    assert run["saliency"][0]["entries"][0]["overlay_available"] is False
    assert run["saliency"][0]["entries"][0]["profiles"][0]["element_id"] == "target"
    assert run["saliency"][0]["profile_event_ids"]
    assert run["saliency"][0]["operational_event_ids"]
    assert run["saliency"][0]["profiles"][0]["prediction_provenance"]
    assert run["saliency_stage_timeline"][0]["search_stage"] == "initial"
    assert run["saliency_fallbacks"] == []

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )
    assert "Saliency evidence" in html
    assert "Overlay unavailable due redaction" in html
    assert "Heatmap-only artifact" in html
    assert "Aggregation components" in html
    assert "CPUExecutionProvider" in html


def test_renderer_accepts_production_saliency_source_event_chain(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-production-linkage",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-production-linkage")
    run = tmp_path / "runs" / "run-production-linkage"
    events = [
        json.loads(line)
        for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    inference = next(
        item for item in events if item["kind"] == "saliency-inference-recorded"
    )
    profiles = next(
        item for item in events if item["kind"] == "saliency-profiles-recorded"
    )
    inference["source_event_id"] = "event-1"
    profiles["source_event_id"] = f"event-{inference['sequence']}"
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_checksums(run)

    loaded = renderer._load_experiment(tmp_path)["runs"][0]

    assert loaded["trusted"] is True
    assert loaded["saliency"][0]["replay_available"] is True


@pytest.mark.parametrize(
    ("source_event_id", "source_viewport_id"),
    (
        ("event-12", "viewport-1"),
        ("event-2", "viewport-1"),
        ("event-11", "viewport-2"),
    ),
)
def test_renderer_rejects_forged_saliency_source_links(
    tmp_path: Path,
    source_event_id: str,
    source_viewport_id: str,
) -> None:
    _write_run(tmp_path, "run-linkage", version="improved", discovery_cost=3)
    _write_saliency_replay_evidence(tmp_path, "run-linkage")
    run = tmp_path / "runs" / "run-linkage"
    events = [
        json.loads(line)
        for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prominence = next(item for item in events if item["kind"] == "prominence-recorded")
    prominence["source_event_id"] = source_event_id
    prominence["source_viewport_id"] = source_viewport_id
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_checksums(run)

    loaded = renderer._load_experiment(tmp_path)["runs"][0]

    assert loaded["saliency"][0]["replay_available"] is False


def test_renderer_rejects_saliency_metadata_provider_forgery(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-provider-forgery",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-provider-forgery")
    metadata_path = (
        tmp_path
        / "runs"
        / "run-provider-forgery"
        / "saliency"
        / "inference-1"
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for prediction in metadata["predictions"]:
        prediction["metadata"]["provider_id"] = "heuristic"
    metadata["saliency_metadata"]["provider_manifests"][0]["provider_id"] = "heuristic"
    metadata_path.write_bytes(
        canonicalize_saliency_artifact_content(
            SaliencyArtifactKind.METADATA,
            json.dumps(metadata).encode(),
            expected_viewport_id="inference-1",
        )
    )
    _write_checksums(metadata_path.parents[2])

    loaded = renderer._load_experiment(tmp_path)["runs"][0]

    assert loaded["trusted"] is False
    assert loaded["saliency"][0]["replay_available"] is False


@pytest.mark.parametrize("mutation", ("duplicate", "after-terminal"))
def test_renderer_rejects_malformed_timeline_order(
    tmp_path: Path, mutation: str
) -> None:
    _write_run(tmp_path, "run-order", version="improved", discovery_cost=3)
    run = tmp_path / "runs" / "run-order"
    events = [
        json.loads(line)
        for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if mutation == "duplicate":
        events[1]["sequence"] = events[0]["sequence"]
    else:
        events.append({"sequence": 11, "kind": "run-started"})
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_checksums(run)

    loaded = renderer._load_experiment(tmp_path)["runs"][0]

    assert loaded["trusted"] is False
    assert any("timeline" in failure for failure in loaded["integrity_failures"])


def test_renderer_rejects_duplicate_bundle_json_fields(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-duplicate-json", version="improved", discovery_cost=3)
    run = tmp_path / "runs" / "run-duplicate-json"
    manifest = (run / "manifest.json").read_text(encoding="utf-8")
    manifest = manifest.replace(
        '"run_id": "run-duplicate-json",',
        '"run_id": "run-duplicate-json", "run_id": "run-duplicate-json",',
        1,
    )
    (run / "manifest.json").write_text(manifest, encoding="utf-8")
    _write_checksums(run)

    loaded = renderer._load_experiment(tmp_path)["runs"][0]

    assert loaded["trusted"] is False


def test_renderer_rejects_malformed_native_map_from_saliency_replay(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-native-map",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-native-map")
    native_map = (
        tmp_path / "runs" / "run-native-map" / "saliency" / "inference-1" / "1s.npz"
    )
    native_map.write_bytes(b"not-a-native-map")
    _write_checksums(native_map.parents[2])

    loaded = renderer._load_experiment(tmp_path)["runs"][0]

    assert loaded["saliency"][0]["replay_available"] is False
    assert loaded["comparison_valid"] is False


def test_renderer_rejects_out_of_range_saliency_aggregate(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-range", version="improved", discovery_cost=3)
    _write_saliency_replay_evidence(tmp_path, "run-range")
    profiles_path = (
        tmp_path / "runs" / "run-range" / "saliency" / "inference-1" / "profiles.json"
    )
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    profiles[0]["aggregates"][0]["density"] = 2.0
    profiles_path.write_text(json.dumps(profiles), encoding="utf-8")
    _write_checksums(profiles_path.parents[2])

    loaded = renderer._load_experiment(tmp_path)["runs"][0]

    assert loaded["saliency"][0]["replay_available"] is False


def test_renderer_binds_bundle_directory_to_embedded_run_id(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-directory", version="improved", discovery_cost=3)
    run = tmp_path / "runs" / "run-directory"
    for name in ("manifest.json", "result.json"):
        value = json.loads((run / name).read_text(encoding="utf-8"))
        value["run_id"] = "run-embedded"
        if name == "result.json":
            value["metrics"]["run_id"] = "run-embedded"
        _write_json(run / name, value)
    _write_checksums(run)

    loaded = renderer._load_experiment(tmp_path)["runs"][0]

    assert loaded["trusted"] is False
    assert any(
        "bundle directory" in failure for failure in loaded["integrity_failures"]
    )


def test_renderer_ignores_symlinked_run_directory(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-real", version="improved", discovery_cost=3)
    try:
        (tmp_path / "runs" / "run-link").symlink_to(
            tmp_path / "runs" / "run-real", target_is_directory=True
        )
    except OSError:
        pytest.skip("symlink creation unavailable")

    directories = renderer._run_directories(tmp_path)

    assert all(path.name != "run-link" for path in directories)


def test_renderer_does_not_follow_hostile_report_symlink(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-report-link", version="improved", discovery_cost=3)
    outside = tmp_path / "outside-report.html"
    outside.write_text("outside report sentinel", encoding="utf-8")
    destination = tmp_path / "report.html"
    try:
        destination.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")

    with pytest.raises(BundleStateError, match="must not be a reparse point"):
        render_experiment_report(tmp_path, destination)

    assert outside.read_text(encoding="utf-8") == "outside report sentinel"
    assert destination.is_symlink()


def test_renderer_rejects_hostile_split_run_directory_link(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-page-link", version="improved", discovery_cost=3)
    outside = tmp_path / "outside-run-pages"
    outside.mkdir()
    run_directory = tmp_path / "report-runs"
    try:
        run_directory.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")

    with pytest.raises((OSError, RuntimeError, ValueError)):
        render_experiment_report(
            tmp_path,
            tmp_path / "report.html",
            max_single_file_bytes=100,
        )

    assert list(outside.iterdir()) == []


def test_renderer_rejects_saliency_artifact_path_traversal(tmp_path: Path) -> None:
    assert (
        renderer._saliency_artifact_data_uri(
            tmp_path,
            "saliency/../artifacts/screenshot.png",
        )
        is None
    )


def test_renderer_rejects_symlinked_saliency_ancestor(tmp_path: Path) -> None:
    real_root = tmp_path / "real-saliency"
    real_root.mkdir()
    try:
        (tmp_path / "saliency").symlink_to(real_root, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    heatmap = real_root / "viewport-1"
    heatmap.mkdir()
    image = BytesIO()
    Image.new("L", (2, 2), color=180).save(image, format="PNG")
    (heatmap / "1s-heatmap.png").write_bytes(image.getvalue())

    assert (
        renderer._saliency_artifact_data_uri(
            tmp_path, "saliency/viewport-1/1s-heatmap.png"
        )
        is None
    )


def test_renderer_rejects_saliency_namespace_traversal_before_bundle_reads(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_json(outside / "profiles.json", [])
    _write_json(outside / "metadata.json", {"warnings": ["OUTSIDE-DATA"]})

    replay = renderer._saliency_replay(
        tmp_path,
        [
            {
                "kind": "saliency-profiles-recorded",
                "viewport_id": "../outside",
                "artifact_namespace": "../outside",
            }
        ],
        [],
    )

    assert replay == []
    assert "OUTSIDE-DATA" not in repr(replay)


def test_renderer_omits_source_pixels_from_tampered_bundle(tmp_path: Path) -> None:
    source = BytesIO()
    Image.new("RGB", (2, 2), color=(20, 30, 40)).save(source, format="PNG")
    secret = b"TAMPERED-SCREENSHOT-SECRET"
    _write_run(
        tmp_path,
        "run-tampered-source",
        version="defective",
        discovery_cost=8,
        screenshot=source.getvalue(),
    )
    screenshot = (
        tmp_path / "runs" / "run-tampered-source" / "artifacts" / "screenshot.png"
    )
    screenshot.write_bytes(screenshot.read_bytes() + secret)

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert secret.decode() not in html
    assert base64.b64encode(secret).decode() not in html


def test_renderer_rejects_inconsistent_persisted_provider_from_scorecards(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-provider-mismatch",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    result_path = tmp_path / "runs" / "run-provider-mismatch" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["metrics"].update(
        {
            "run_id": "run-provider-mismatch",
            "prominence_provider_id": "heuristic",
            "comparison_valid": True,
        }
    )
    result["ux_sample_valid"] = True
    _write_json(result_path, result)
    _write_checksums(result_path.parent)

    experiment = renderer._load_experiment(tmp_path)

    assert experiment["runs"][0]["comparison_valid"] is False
    assert experiment["comparison_rows"] == []


def test_renderer_redacts_invalid_manifest_provider_identity(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "run-invalid-provider",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="heuristic",
    )
    manifest_path = tmp_path / "runs" / "run-invalid-provider" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prominence_provider_id"] = "token-provider-secret"
    _write_json(manifest_path, manifest)
    result_path = tmp_path / "runs" / "run-invalid-provider" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["metrics"]["prominence_provider_id"] = "token-provider-secret"
    _write_json(result_path, result)
    _write_checksums(manifest_path.parent)

    experiment = renderer._load_experiment(tmp_path)
    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert experiment["runs"][0]["comparison_valid"] is False
    assert "token-provider-secret" not in html


def test_renderer_canonicalizes_real_heuristic_provider_aliases(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-heuristic-alias",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="heuristic",
    )
    run = tmp_path / "runs" / "run-heuristic-alias"
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prominence_provider_id"] = "heuristic-prominence"
    manifest["provider_manifests"].append(
        {
            "provider_id": "heuristic-prominence",
            "role": "prominence",
            "model_id": None,
            "endpoint_origin": "internal",
            "version": "heuristic-prominence-v1",
        }
    )
    _write_json(manifest_path, manifest)
    result_path = run / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["metrics"]["prominence_provider_id"] = "heuristic-prominence"
    _write_json(result_path, result)
    _write_checksums(run)

    loaded = renderer._load_experiment(tmp_path)["runs"][0]

    assert loaded["trusted"] is True
    assert loaded["prominence_provider_id"] == "heuristic"


def test_renderer_marks_malformed_saliency_schema_unavailable(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-invalid-saliency", version="improved", discovery_cost=3)
    _write_saliency_replay_evidence(tmp_path, "run-invalid-saliency")
    profiles_path = (
        tmp_path
        / "runs"
        / "run-invalid-saliency"
        / "saliency"
        / "inference-1"
        / "profiles.json"
    )
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    profiles[0]["unexpected"] = "invalid"
    _write_json(profiles_path, profiles)
    _write_checksums(profiles_path.parents[2])

    run = renderer._load_experiment(tmp_path)["runs"][0]

    assert run["saliency"][0]["replay_available"] is False
    assert run["saliency"][0]["entries"] == []


def test_renderer_rejects_incomplete_saliency_profile_duration_coverage(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-incomplete-profile",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-incomplete-profile")
    profiles_path = (
        tmp_path
        / "runs"
        / "run-incomplete-profile"
        / "saliency"
        / "inference-1"
        / "profiles.json"
    )
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    profiles[0]["early"] = None
    profiles_path.write_bytes(
        canonicalize_saliency_artifact_content(
            SaliencyArtifactKind.PROFILES,
            json.dumps(profiles).encode(),
            expected_viewport_id="inference-1",
        )
    )
    _write_checksums(profiles_path.parents[2])

    run = renderer._load_experiment(tmp_path)["runs"][0]

    assert run["saliency"][0]["replay_available"] is False


def test_renderer_accepts_profile_with_all_timed_estimates_unavailable(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-unavailable-profile",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-unavailable-profile")
    profiles_path = (
        tmp_path
        / "runs"
        / "run-unavailable-profile"
        / "saliency"
        / "inference-1"
        / "profiles.json"
    )
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    for duration in ("immediate", "early", "eventual"):
        profiles[0][duration] = None
    profiles_path.write_bytes(
        canonicalize_saliency_artifact_content(
            SaliencyArtifactKind.PROFILES,
            json.dumps(profiles).encode(),
            expected_viewport_id="inference-1",
        )
    )
    _write_checksums(profiles_path.parents[2])

    run = renderer._load_experiment(tmp_path)["runs"][0]

    assert run["saliency"][0]["replay_available"] is True
    assert run["saliency"][0]["entries"]


def test_renderer_rejects_profile_prediction_provenance_mismatch(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-profile-provenance-mismatch",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-profile-provenance-mismatch")
    profiles_path = (
        tmp_path
        / "runs"
        / "run-profile-provenance-mismatch"
        / "saliency"
        / "inference-1"
        / "profiles.json"
    )
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    profiles[0]["prediction_provenance"][0]["metadata"]["model_checksum"] = "4" * 64
    profiles_path.write_bytes(
        canonicalize_saliency_artifact_content(
            SaliencyArtifactKind.PROFILES,
            json.dumps(profiles).encode(),
            expected_viewport_id="inference-1",
        )
    )
    _write_checksums(profiles_path.parents[2])

    run = renderer._load_experiment(tmp_path)["runs"][0]

    assert run["saliency"][0]["replay_available"] is False


def test_renderer_rejects_saliency_screenshot_digest_mismatch(
    tmp_path: Path,
) -> None:
    screenshot = BytesIO()
    Image.new("RGB", (2, 2), color="white").save(screenshot, format="PNG")
    _write_run(
        tmp_path,
        "run-screenshot-digest",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
        screenshot=screenshot.getvalue(),
    )
    _write_saliency_replay_evidence(tmp_path, "run-screenshot-digest")

    run = renderer._load_experiment(tmp_path)["runs"][0]

    assert run["saliency"][0]["replay_available"] is False


def test_renderer_rejects_oversized_saliency_heatmap(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-oversized-heatmap",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-oversized-heatmap")
    heatmap_path = (
        tmp_path
        / "runs"
        / "run-oversized-heatmap"
        / "saliency"
        / "inference-1"
        / "1s-heatmap.png"
    )
    heatmap_path.write_bytes(b"oversized" * (8 * 1024 * 1024 // 8 + 1))
    _write_checksums(heatmap_path.parents[2])

    run = renderer._load_experiment(tmp_path)["runs"][0]

    assert run["saliency"][0]["replay_available"] is False


def test_renderer_rejects_non_screenshot_artifact_path(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-screenshot-allowlist",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-screenshot-allowlist")
    run = tmp_path / "runs" / "run-screenshot-allowlist"
    events = [
        json.loads(line)
        for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    capture = next(item for item in events if item["kind"] == "viewport-captured")
    capture["snapshot"]["screenshot_artifact"] = "saliency/inference-1/1s-heatmap.png"
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_checksums(run)

    snapshot = renderer._load_experiment(tmp_path)["runs"][0]["snapshots"][0]

    assert snapshot["screenshot"] is None


def test_renderer_replays_content_addressed_verifier_screenshot(
    tmp_path: Path,
) -> None:
    source = BytesIO()
    image = Image.new("RGB", (2, 2), color="red")
    image.putpixel((0, 0), (0, 0, 255))
    image.save(source, format="PNG")
    screenshot = source.getvalue()
    digest = hashlib.sha256(screenshot).hexdigest()
    _write_run(
        tmp_path,
        "run-verifier-screenshot",
        version="improved",
        discovery_cost=3,
        screenshot=screenshot,
    )
    run = tmp_path / "runs" / "run-verifier-screenshot"
    artifact_path = run / "artifacts" / digest
    artifact_path.write_bytes(screenshot)
    (run / "artifacts" / "screenshot.png").unlink()
    events = [
        json.loads(line)
        for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    capture = next(event for event in events if event["kind"] == "viewport-captured")
    capture["snapshot"]["screenshot_artifact"] = f"artifacts/{digest}"
    verification = next(
        event for event in events if event["kind"] == "verification-recorded"
    )
    verification["result"]["evidence_ids"].append(f"screenshot:artifacts/{digest}")
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_checksums(run)

    snapshot = renderer._load_experiment(tmp_path)["runs"][0]["snapshots"][0]

    expected = base64.b64encode(screenshot).decode("ascii")
    assert snapshot["screenshot"] == f"data:image/png;base64,{expected}"


@pytest.mark.parametrize(
    "artifact",
    (
        f"artifacts/{'a' * 63}",
        f"artifacts/{'g' * 64}",
        "artifacts/screenshot",
        f"generated/{'a' * 64}",
        f"artifacts/../{'a' * 64}",
    ),
)
def test_renderer_rejects_non_content_addressed_extensionless_screenshot_paths(
    artifact: str,
) -> None:
    assert not renderer._is_allowed_screenshot_artifact(PurePosixPath(artifact))


def test_renderer_accepts_uppercase_content_addressed_screenshot_name() -> None:
    assert renderer._is_allowed_screenshot_artifact(
        PurePosixPath("artifacts", "A" * 64)
    )


def test_renderer_rejects_saliency_profile_element_without_snapshot_lineage(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-profile-lineage",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-profile-lineage")
    profiles_path = (
        tmp_path
        / "runs"
        / "run-profile-lineage"
        / "saliency"
        / "inference-1"
        / "profiles.json"
    )
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    profiles[0]["element_id"] = "not-in-snapshot"
    for aggregate in profiles[0]["aggregates"]:
        aggregate["element_id"] = "not-in-snapshot"
    profiles_path.write_bytes(
        canonicalize_saliency_artifact_content(
            SaliencyArtifactKind.PROFILES,
            json.dumps(profiles).encode(),
            expected_viewport_id="inference-1",
        )
    )
    _write_checksums(profiles_path.parents[2])

    run = renderer._load_experiment(tmp_path)["runs"][0]

    assert run["saliency"][0]["replay_available"] is False


def test_renderer_preserves_repeated_saliency_stage_history() -> None:
    group = {"stage_history": [], "warnings": [], "timings_ms": []}

    renderer._merge_saliency_event(
        group,
        {
            "kind": "saliency-profiles-recorded",
            "sequence": 4,
            "provider_id": "foveacast",
            "search_stage": "initial",
            "selected_mixture": [["1s", 1.0]],
            "timings_ms": [1.0, 2.0, 3.0],
        },
    )
    renderer._merge_saliency_event(
        group,
        {
            "kind": "prominence-recorded",
            "sequence": 8,
            "provider_id": "foveacast-prominence",
            "search_stage": "persistent",
            "selected_mixture": [["3s", 0.25], ["7s", 0.75]],
            "cache_state": "hit",
        },
    )

    assert [item["search_stage"] for item in group["stage_history"]] == [
        "initial",
        "persistent",
    ]
    assert group["search_stage"] == "persistent"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_renderer_browser_replays_saliency_tabs_and_selected_viewport(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-multi-viewport",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    run = tmp_path / "runs" / "run-multi-viewport"
    events = [
        json.loads(line)
        for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    terminal = events.pop()
    events.append(
        {
            "kind": "viewport-captured",
            "viewport_width": 800,
            "viewport_height": 600,
            "snapshot": {
                "id": "viewport-2",
                "screenshot_artifact": "artifacts/screenshot.png",
                "elements": [
                    {
                        "id": "target",
                        "role": "button",
                        "label": "Target",
                        "bounds": {"x": 90, "y": 80, "width": 180, "height": 40},
                        "visibility_fraction": 1,
                        "actionable": True,
                    }
                ],
                "regions": [],
            },
        }
    )
    events.append(terminal)
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_checksums(run)
    _write_saliency_replay_evidence(tmp_path, "run-multi-viewport")
    _write_saliency_replay_evidence(
        tmp_path,
        "run-multi-viewport",
        namespace="inference-2",
        source_viewport_id="viewport-2",
        source_event_id="event-10",
    )
    report_path = render_experiment_report(tmp_path, tmp_path / "report.html")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(report_path.resolve().as_uri())
        tabs = page.locator(".saliency-tab")
        assert await tabs.count() == 6, repr(
            [
                (
                    group["artifact_namespace"],
                    group["replay_available"],
                    group["replay_error"],
                    len(group["entries"]),
                    group.get("profile_event_ids"),
                    group.get("operational_event_ids"),
                    group.get("source_event_ids"),
                    [
                        (event.get("sequence"), event.get("kind"))
                        for event in json.loads(
                            "["
                            + ",".join(
                                (run / "timeline.jsonl")
                                .read_text(encoding="utf-8")
                                .splitlines()
                            )
                            + "]"
                        )
                        if event.get("sequence") in {12, 13, 15, 16}
                    ],
                )
                for group in renderer._load_experiment(tmp_path)["runs"][0]["saliency"]
            ]
        )
        await tabs.nth(3).click()
        assert await page.locator("#saliency-detail .saliency-heatmap").is_visible()
        await page.locator(".ranked-element-button").first.click()
        assert await page.locator("#selected-element-evidence").get_attribute(
            "data-selected-element-id"
        )
        await browser.close()


def test_renderer_reports_redacted_source_and_keeps_heatmap_replay(
    tmp_path: Path,
) -> None:
    blank = BytesIO()
    Image.new("RGBA", (2, 2), color=(0, 0, 0, 255)).save(blank, format="PNG")
    _write_run(
        tmp_path,
        "run-redacted-saliency",
        version="improved",
        discovery_cost=3,
        screenshot=blank.getvalue(),
        prominence_provider_id="foveacast",
    )
    _write_saliency_replay_evidence(tmp_path, "run-redacted-saliency")

    experiment = renderer._load_experiment(tmp_path)
    run = experiment["runs"][0]
    snapshot = run["snapshots"][0]

    assert snapshot["screenshot"] is None
    assert snapshot["screenshot_redacted"] is True
    assert run["saliency"][0]["entries"][0]["heatmap"]
    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )
    assert "Overlay unavailable due redaction" in html
    assert "Heatmap-only artifact" in html


def test_renderer_excludes_fallback_learned_run_from_report_scorecards(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-fallback",
        version="improved",
        discovery_cost=3,
        prominence_provider_id="foveacast",
    )
    run = tmp_path / "runs" / "run-fallback"
    events = [
        json.loads(line)
        for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    terminal = events.pop()
    events.append(
        {
            "kind": "saliency-fallback-recorded",
            "viewport_id": "viewport-1",
            "provider_id": "foveacast",
            "fallback_provider_id": "heuristic-prominence",
            "search_stage": "initial",
            "reason": "runtime unavailable",
            "cache_state": "fallback",
        }
    )
    events.append(terminal)
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_checksums(run)

    experiment = renderer._load_experiment(tmp_path)
    loaded = experiment["runs"][0]

    assert loaded["ux_sample_valid"] is False
    assert loaded["comparison_valid"] is False
    assert loaded["saliency_fallbacks"]
    assert experiment["comparison_rows"] == []


def test_renderer_exposes_safe_model_and_progress_diagnostics(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "run-diagnostics",
        version="improved",
        discovery_cost=3,
        outcome="model-failure",
        verified=False,
        terminal_reason="cognitive: inspect action requires element_id",
        ux_sample_valid=False,
        ux_sample_invalid_reason="model-failure: invalid cognitive response",
    )
    run = tmp_path / "runs" / "run-diagnostics"
    events = [
        json.loads(line)
        for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    terminal = events.pop()
    events.extend(
        [
            {
                "kind": "attention-selection-recorded",
                "viewport_id": "viewport-2",
                "selected_ids": ["email", "submit"],
                "recovery_selected_ids": ["submit"],
                "selection_mode": "recovery-forced",
                "region_id": None,
                "element_probabilities": {"email": 0.4, "submit": 0.1},
                "region_probabilities": {"form": 0.4, "actions": 0.1},
            },
            {
                "kind": "model-failure",
                "role": "cognitive",
                "reason": "inspect action requires element_id",
                "response_summary": {
                    "action": "inspect",
                    "element_id": None,
                    "api_key": "provider-secret",
                },
            },
            {
                "kind": "repeated-fixture-input",
                "element_id": "target",
                "fixture_key": "invite_email",
                "reason": "fixture input was already entered successfully",
            },
            {
                "kind": "repeated-action-detected",
                "action": {"kind": "interact-with-element", "element_id": "target"},
                "count": 3,
                "reason": "identical action repeated without sufficient progress",
            },
            {
                "kind": "no-progress-recovery",
                "action": {"kind": "scroll", "direction": "down"},
                "count": 2,
                "reason": "action succeeded but the recaptured interface did not change semantically",
            },
            {
                "kind": "no-progress-detected",
                "count": 3,
                "reason": "three consecutive actions produced no state progress",
            },
            {
                "kind": "repeated-action-cycle",
                "cycle_length": 2,
                "reason": "repeated semantic action cycle detected",
                "url": "https://fixture.test/invite?token=private-secret",
            },
        ]
    )
    events.append(terminal)
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    _write_checksums(run)

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert "inspect action requires element_id" in html
    assert '"ux_sample_valid":false' in html
    assert "model-failure: invalid cognitive response" in html
    assert '"recovery_selected_ids":["submit"]' in html
    assert "recovery-forced" in html
    assert '"role":"cognitive"' in html
    assert '"response_summary":{"action":"inspect","element_id":null}' in html
    assert "repeated-fixture-input" in html
    assert "invite_email" in html
    assert "repeated-action-detected" in html
    assert '"count":3' in html
    assert "no-progress-recovery" in html
    assert (
        '"kind":"no-progress-recovery",'
        '"action":{"kind":"scroll","direction":"down"},'
        '"count":2'
    ) in html
    assert "no-progress-detected" in html
    assert "repeated-action-cycle" in html
    assert '"cycle_length":2' in html
    assert "provider-secret" not in html
    assert "private-secret" not in html


def test_renderer_includes_all_failed_experiment_and_staging_crash(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".staging" / "run-crashed"
    staging.mkdir(parents=True)
    _write_json(staging / "manifest.json", {"run_id": "run-crashed", "seed": 7})
    (staging / "timeline.jsonl").write_text(
        json.dumps({"sequence": 1, "kind": "run-started", "run_id": "run-crashed"})
        + "\n"
        + '{"sequence":2,"kind":"viewport-captured"',
        encoding="utf-8",
    )
    _write_json(
        staging / "crash.marker",
        {"run_id": "run-crashed", "reason": "browser capture failed"},
    )
    _write_json(
        staging / "result.json",
        {
            "metrics": {
                "discovery_cost": {"total": 999999},
                "verified_completion": True,
            },
            "outcome": {"kind": "verified-success"},
        },
    )
    _write_json(
        tmp_path / "experiment.json",
        {
            "run_metrics": [],
            "cell_aggregates": [],
            "variant_comparisons": [],
            "findings": {},
            "failures": [
                {
                    "run_id": "run-crashed",
                    "error_type": "ProviderFailure",
                    "stage": "execution",
                    "terminal_state": "crashed",
                    "reason": "browser capture failed",
                    "scenario_id": "enable-2fa",
                    "application_version_id": "fixture-app-improved",
                    "persona_id": "impatient",
                    "policy": "progressive-prominence-scent",
                    "seed": 7,
                    "model_trial": 1,
                },
            ],
        },
    )

    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert output.is_file()
    assert "Comparison overview" in html
    assert 'data-run-id="run-crashed"' in html
    assert "browser capture failed" in html
    assert "run-crashed" in html
    assert "enable-2fa" in html
    assert "progressive-prominence-scent" in html
    assert 'data-metric="model-trial">1</td>' in html
    assert "999999" not in html


def test_renderer_does_not_score_incomplete_bundle_with_result_metrics(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".staging" / "run-incomplete"
    staging.mkdir(parents=True)
    _write_json(staging / "manifest.json", {"run_id": "run-incomplete", "seed": 8})
    (staging / "timeline.jsonl").write_text(
        json.dumps({"sequence": 1, "kind": "run-started"}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        staging / "result.json",
        {"metrics": {"discovery_cost": {"total": 888888}}},
    )
    _write_json(staging / "crash.marker", {"reason": "capture failed"})

    output = render_experiment_report(tmp_path, tmp_path / "report.html")

    assert "888888" not in output.read_text(encoding="utf-8")


def test_renderer_excludes_tampered_bundle_from_scorecards_and_gates(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-defective", version="defective", discovery_cost=8)
    _write_run(tmp_path, "run-improved", version="improved", discovery_cost=3)
    tampered = tmp_path / "runs" / "run-improved" / "result.json"
    tampered.write_text(
        tampered.read_text(encoding="utf-8").replace('"total": 3', '"total": 999999'),
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "experiment.json",
        {
            "variant_comparisons": [
                {
                    "baseline": {
                        "scenario_id": "invite",
                        "application_version_id": "defective",
                        "persona_id": "persona",
                        "policy": "progressive-prominence-scent",
                    },
                    "improved": {
                        "scenario_id": "invite",
                        "application_version_id": "improved",
                        "persona_id": "persona",
                        "policy": "progressive-prominence-scent",
                    },
                    "gate": {
                        "passed": True,
                        "paired_seed_count": 1,
                        "reasons": [],
                    },
                }
            ]
        },
    )

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert "checksum mismatch: result.json" in html
    assert "999999" not in html
    assert "All directional checks passed" not in html


def test_renderer_derives_directional_gate_from_all_trusted_bundles(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-defective", version="defective", discovery_cost=8)
    _write_run(tmp_path, "run-improved", version="improved", discovery_cost=3)

    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "All directional checks passed" in html
    assert "Directional gates unavailable" not in html


def test_renderer_excludes_active_bundle_from_scorecards_and_findings(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, "run-active", version="improved", discovery_cost=777777)
    active_run = tmp_path / "runs" / "run-active"
    (active_run / ".active").write_text('{"run_id":"run-active"}', encoding="utf-8")

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert "run-active" in html
    assert "active bundle marker present" in html
    assert "777777" not in html
    assert "Target wording gives weak goal cues" not in html


@pytest.mark.parametrize("filename", ("manifest.json", "timeline.jsonl", "result.json"))
def test_renderer_reports_missing_required_bundle_file_as_untrusted(
    tmp_path: Path, filename: str
) -> None:
    _write_run(tmp_path, "run-missing", version="defective", discovery_cost=8)
    (tmp_path / "runs" / "run-missing" / filename).unlink()

    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert f"missing required bundle file: {filename}" in html
    assert "run-missing" in html


def test_renderer_never_embeds_sensitive_fixture_artifacts(tmp_path: Path) -> None:
    invite_email = "invitee@example.test"
    totp_code = "246810"
    writer = FilesystemRunBundleWriter.start(
        tmp_path,
        BundleManifest(
            run_id="run-sensitive",
            seed=1,
            config_digest="config-sha",
            endpoint_origin="https://llm.example.test",
        ),
        redaction=RedactionPolicy(exact_values=(invite_email, totp_code)),
    )
    screenshot = writer.write_artifact(
        "screenshot.png",
        b"\x89PNG\r\n\x1a\n" + invite_email.encode() + totp_code.encode(),
    )
    writer.append_event(RunStarted(run_id="run-sensitive"))
    writer.append_event(
        {
            "kind": "viewport-captured",
            "snapshot": {
                "id": "viewport-1",
                "screenshot_artifact": screenshot.path,
                "elements": [],
            },
        }
    )
    writer.append_event(
        {
            "kind": "run-terminated",
            "outcome": {"kind": "agent-abandoned"},
            "fixture_inputs": {
                "invite_email": invite_email,
                "totp_code": totp_code,
            },
        }
    )
    final_path = writer.finalize(
        {
            "outcome": {"kind": "agent-abandoned"},
            "invite_email": invite_email,
            "totp_code": totp_code,
        }
    )

    output = render_experiment_report(tmp_path, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert invite_email not in html
    assert totp_code not in html
    assert base64.b64encode(invite_email.encode()).decode() not in html
    assert base64.b64encode(totp_code.encode()).decode() not in html
    artifact_bytes = b"".join(
        path.read_bytes() for path in (final_path / "artifacts").iterdir()
    )
    assert invite_email.encode() not in artifact_bytes
    assert totp_code.encode() not in artifact_bytes


def test_renderer_builds_comparison_and_splits_large_experiment(tmp_path: Path) -> None:
    _write_run(tmp_path, "run-defective", version="defective", discovery_cost=8)
    _write_run(tmp_path, "run-improved", version="improved", discovery_cost=3)
    _write_json(
        tmp_path / "experiment.json",
        {
            "variant_comparisons": [
                {
                    "baseline": {
                        "scenario_id": "invite",
                        "application_version_id": "defective",
                        "persona_id": "persona",
                        "policy": "progressive-prominence-scent",
                    },
                    "improved": {
                        "scenario_id": "invite",
                        "application_version_id": "improved",
                        "persona_id": "persona",
                        "policy": "progressive-prominence-scent",
                    },
                    "gate": {
                        "passed": True,
                        "paired_seed_count": 1,
                        "reasons": [],
                    },
                }
            ]
        },
    )

    output = render_experiment_report(
        tmp_path,
        tmp_path / "experiment.html",
        max_single_file_bytes=100,
    )

    html = output.read_text(encoding="utf-8")
    run_pages = tmp_path / "experiment-runs"
    assert output == tmp_path / "experiment.html"
    assert "run-defective.html" in html
    assert "run-improved.html" in html
    assert (run_pages / "run-defective.html").is_file()
    assert (run_pages / "run-improved.html").is_file()
    assert "Defective" in html
    assert "Improved" in html
    assert "discovery-cost" in html
    assert "directional gates" in html
    assert "All directional checks passed" in html


def test_renderer_preflights_threshold_before_full_aggregate_render(
    tmp_path: Path, monkeypatch
) -> None:
    _write_run(
        tmp_path,
        "run-large",
        version="defective",
        discovery_cost=8,
        screenshot=b"x" * 50_000,
    )
    aggregate_payloads: list[bool] = []
    original_render = renderer._render_html

    def track_render(context, title):
        aggregate_payloads.append(any("timeline" in run for run in context["runs"]))
        return original_render(context, title)

    monkeypatch.setattr(renderer, "_render_html", track_render)

    render_experiment_report(
        tmp_path,
        tmp_path / "report.html",
        max_single_file_bytes=20_000,
    )

    assert aggregate_payloads[0] is False


def test_renderer_replaces_oversized_run_page_with_bounded_notice(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-oversized",
        version="defective",
        discovery_cost=8,
        screenshot=b"x" * 100_000,
    )
    threshold = 20_000

    render_experiment_report(
        tmp_path,
        tmp_path / "report.html",
        max_single_file_bytes=threshold,
    )

    run_page = tmp_path / "report-runs" / "run-oversized.html"
    html = run_page.read_text(encoding="utf-8")
    assert run_page.stat().st_size <= threshold
    assert (
        "Detailed replay omitted because run page exceeds configured size limit."
        in html
    )


def test_renderer_default_split_keeps_complete_run_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(
        tmp_path,
        "run-complete",
        version="defective",
        discovery_cost=8,
        screenshot=b"x" * 100_000,
    )
    monkeypatch.setattr(
        renderer,
        "_estimated_full_report_bytes",
        lambda _experiment: renderer.DEFAULT_SINGLE_FILE_THRESHOLD + 1,
    )

    render_experiment_report(tmp_path, tmp_path / "report.html")

    run_page = tmp_path / "report-runs" / "run-complete.html"
    html = run_page.read_text(encoding="utf-8")
    assert "Run workspace" in html
    assert "Detailed replay omitted" not in html


def test_renderer_streams_checksum_verification_for_unreferenced_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(tmp_path, "run-streamed", version="defective", discovery_cost=8)
    artifact = tmp_path / "runs" / "run-streamed" / "artifacts" / "large.bin"
    artifact.write_bytes(b"x" * 100_000)
    _write_checksums(artifact.parents[1])
    original_read_bytes = Path.read_bytes

    def reject_whole_file_read(path: Path) -> bytes:
        if path == artifact:
            raise AssertionError("checksum verification read whole artifact")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)

    output = render_experiment_report(tmp_path, tmp_path / "report.html")

    assert output.is_file()


def test_renderer_split_index_is_concise_and_uses_collision_safe_run_links(
    tmp_path: Path,
) -> None:
    for run_id in ("run active", "run_active"):
        _write_run(tmp_path, run_id, version="defective", discovery_cost=8)
        result_path = tmp_path / "runs" / run_id / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["limitations"] = ["x" * 120_000]
        _write_json(result_path, result)
        _write_checksums(result_path.parent)
    threshold = 80_000

    output = render_experiment_report(
        tmp_path,
        tmp_path / "report.html",
        max_single_file_bytes=threshold,
    )

    html = output.read_text(encoding="utf-8")
    run_pages = tuple((tmp_path / "report-runs").glob("*.html"))
    assert output.stat().st_size <= threshold
    assert "x" * 1_000 not in html
    assert len(run_pages) == 2
    assert len({path.name for path in run_pages}) == 2
    assert all(path.name in html for path in run_pages)


def test_renderer_overview_keeps_every_run_and_preserves_outcome_from_failure_stage(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-timeout",
        version="defective",
        discovery_cost=12,
        outcome="timed-out",
        verified=False,
        terminal_reason="run timeout exceeded",
    )
    _write_run(
        tmp_path,
        "run-evaluation",
        version="improved",
        discovery_cost=4,
        outcome="agent-abandoned",
        verified=False,
        terminal_reason="agent chose to stop",
        evaluation_failure_reason="evaluation evidence unavailable: target absent",
    )
    _write_json(
        tmp_path / "experiment.json",
        {
            "failures": [
                {
                    "run_id": "run-evaluation",
                    "error_type": "EvaluationFailure",
                    "stage": "evaluation",
                    "terminal_state": "finalized",
                    "reason": "evaluation evidence unavailable: target absent",
                    "scenario_id": "invite",
                    "application_version_id": "improved",
                    "persona_id": "persona",
                    "policy": "progressive-prominence-scent",
                    "seed": 7,
                }
            ]
        },
    )

    experiment = renderer._load_experiment(tmp_path)
    by_id = {run["run_id"]: run for run in experiment["runs"]}
    html = render_experiment_report(tmp_path, tmp_path / "report.html").read_text(
        encoding="utf-8"
    )

    assert len(experiment["runs"]) == 2
    assert by_id["run-timeout"]["outcome"] == "timed-out"
    assert by_id["run-timeout"]["terminal_reason"] == "run timeout exceeded"
    assert by_id["run-evaluation"]["outcome"] == "agent-abandoned"
    assert by_id["run-evaluation"]["stage"] == "evaluation"
    assert by_id["run-evaluation"]["evaluation_failure_reason"] == (
        "evaluation evidence unavailable: target absent"
    )
    assert 'data-run-id="run-timeout"' in html
    assert 'data-run-id="run-evaluation"' in html
    assert "timed-out" in html
    assert "agent-abandoned" in html
    assert "evaluation evidence unavailable: target absent" in html
    assert "Gate unavailable" in html


def test_split_report_renders_execution_failure_without_run_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_json(
        tmp_path / "experiment.json",
        {
            "failures": [
                {
                    "run_id": "run-execution-failure",
                    "error_type": "ProviderFailure",
                    "stage": "execution",
                    "terminal_state": "failed",
                    "reason": "browser capture failed",
                    "scenario_id": "invite",
                    "application_version_id": "defective",
                    "persona_id": "persona",
                    "policy": "full-list",
                    "seed": 7,
                },
                {
                    "run_id": "run-second-execution-failure",
                    "error_type": "ProviderFailure",
                    "stage": "execution",
                    "terminal_state": "failed",
                    "reason": "browser startup failed",
                    "scenario_id": "invite",
                    "application_version_id": "improved",
                    "persona_id": "persona",
                    "policy": "full-list",
                    "seed": 7,
                },
            ]
        },
    )

    monkeypatch.setattr(
        renderer,
        "_estimated_full_report_bytes",
        lambda _: renderer.DEFAULT_SINGLE_FILE_THRESHOLD + 1,
    )
    report = render_experiment_report(tmp_path, tmp_path / "report.html")

    html = report.read_text(encoding="utf-8")
    run_pages = tuple((tmp_path / "report-runs").glob("*.html"))
    assert 'data-run-id="run-execution-failure"' in html
    assert len(run_pages) == 2
    assert any(
        "browser capture failed" in page.read_text(encoding="utf-8")
        for page in run_pages
    )


def test_renderer_gate_does_not_hide_jointly_completed_regression() -> None:
    def run(
        seed: int,
        version: str,
        *,
        verified: bool,
        cost: float,
        wrong: float = 0,
        backtracks: float = 0,
    ) -> dict[str, object]:
        return {
            "seed": seed,
            "scenario_id": "invite",
            "scenario_label": "Invite",
            "persona_id": "persona",
            "persona_label": "Persona",
            "policy": "progressive-prominence-scent",
            "version_id": version,
            "version_label": version.title(),
            "verified": verified,
            "metrics": [
                {"name": "discovery-cost", "value": cost},
                {"name": "wrong-actions", "value": wrong},
                {"name": "backtracks", "value": backtracks},
            ],
        }

    gate = renderer._derived_gate_row(
        [
            run(7, "defective", verified=False, cost=1),
            run(7, "improved", verified=True, cost=2),
            run(8, "defective", verified=True, cost=1),
            run(8, "improved", verified=True, cost=4, wrong=1, backtracks=1),
        ]
    )

    assert gate is not None
    assert not gate["passed"]
    assert gate["reasons"] == [
        "paired median discovery cost did not decrease",
        "paired median wrong-action burden increased",
        "paired median backtrack burden increased",
    ]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_report_browser_workspace_replays_and_inspects_without_network(
    tmp_path: Path,
) -> None:
    _write_run(
        tmp_path,
        "run-timeout",
        version="defective",
        discovery_cost=12,
        outcome="timed-out",
        verified=False,
        terminal_reason="run timeout exceeded",
    )
    _write_run(
        tmp_path,
        "run-evaluation",
        version="improved",
        discovery_cost=4,
        outcome="agent-abandoned",
        verified=False,
        terminal_reason="agent chose to stop",
        evaluation_failure_reason="evaluation evidence unavailable: target absent",
    )
    _write_json(
        tmp_path / "experiment.json",
        {
            "failures": [
                {
                    "run_id": "run-evaluation",
                    "error_type": "EvaluationFailure",
                    "stage": "evaluation",
                    "terminal_state": "finalized",
                    "reason": "evaluation evidence unavailable: target absent",
                    "scenario_id": "invite",
                    "application_version_id": "improved",
                    "persona_id": "persona",
                    "policy": "progressive-prominence-scent",
                    "seed": 7,
                }
            ]
        },
    )
    report_path = render_experiment_report(tmp_path, tmp_path / "report.html")
    external_requests: list[str] = []

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(service_workers="block")

        async def block_external(route: Route) -> None:
            if route.request.url.startswith(("file:", "data:")):
                await route.continue_()
            else:
                external_requests.append(route.request.url)
                await route.abort()

        await context.route("**/*", block_external)
        page = await context.new_page()
        await page.goto(report_path.resolve().as_uri())

        overview_row = page.locator('tr[data-run-id="run-evaluation"]')
        assert (
            await overview_row.locator('[data-metric="user-actions"]').text_content()
            == "1"
        )
        assert (
            await overview_row.locator('[data-metric="observations"]').text_content()
            == "1"
        )
        assert (
            await overview_row.locator('[data-metric="discovery-cost"]').text_content()
            == "4.0"
        )
        await overview_row.get_by_role("link", name="Open replay").click()
        assert "run=run-evaluation" in page.url
        event_ids = await page.locator(".timeline-event").evaluate_all(
            "nodes => nodes.map(node => node.dataset.eventId)"
        )
        assert event_ids == [f"event-{sequence}" for sequence in range(1, 11)]
        failure = page.locator("#run-status-banner")
        assert "Agent Abandoned" in (await failure.text_content() or "")
        assert "evaluation" in (await failure.text_content() or "")
        assert "evaluation evidence unavailable: target absent" in (
            await failure.text_content() or ""
        )

        await page.locator("#play-pause").click()
        assert await page.locator("#play-pause").get_attribute("aria-pressed") == "true"
        await page.locator("#play-pause").click()
        assert (
            await page.locator("#play-pause").get_attribute("aria-pressed") == "false"
        )
        await page.locator("#step-forward").click()
        assert "Event 2 /" in (
            await page.locator("#playback-position").text_content() or ""
        )
        await page.locator("#restart-playback").click()
        assert "Event 1 /" in (
            await page.locator("#playback-position").text_content() or ""
        )
        assert "time unavailable" in (
            await page.locator("#playback-position").text_content() or ""
        )

        await page.locator('[data-event-kind="prominence-recorded"]').click()
        target = page.locator('[data-element-id="target"]').first
        await target.hover()
        panel = page.locator("#selected-element-evidence")
        assert await panel.get_attribute("data-selected-element-id") == "target"
        panel_text = await panel.text_content() or ""
        assert "0.2" in panel_text
        assert "7200" in panel_text
        assert "0.4" in panel_text
        assert "0.1" in panel_text
        assert "Blocked by other content" in panel_text
        assert "Local contrast" in panel_text
        assert "Linked decisions" in panel_text
        assert "Linked actions and results" in panel_text
        assert "Linked findings" in panel_text

        await page.locator('[data-event-kind="action-proposed"]').click()
        event_text = await page.locator("#current-event-card").text_content() or ""
        assert "Target matches goal." in event_text
        assert "Interact With Element" in event_text
        await page.locator('[data-event-kind="model-call-recorded"]').click()
        model_text = await page.locator("#current-event-card").text_content() or ""
        assert "safe request" in model_text
        assert "safe response" in model_text
        assert "cognitive-v1" in model_text
        assert "125" in model_text
        assert "2" in model_text

        await (
            page.locator('tr[data-run-id="run-timeout"]')
            .get_by_role("link", name="Open replay")
            .click()
        )
        timeout_text = await page.locator("#run-status-banner").text_content() or ""
        assert "Timed Out" in timeout_text
        assert "run timeout exceeded" in timeout_text
        await page.set_viewport_size({"width": 390, "height": 844})
        dimensions = await page.evaluate(
            "({scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth})"
        )
        assert dimensions["scrollWidth"] <= dimensions["innerWidth"]
        assert await page.locator("#play-pause").is_visible()
        assert await page.locator("#selected-element-evidence").is_visible()
        await browser.close()

    assert not external_requests
