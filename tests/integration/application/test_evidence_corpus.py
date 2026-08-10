from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ux_analyzer.application.checkpoint import finalized_bundle_failures
from ux_analyzer.application.evidence_corpus import (
    EvidenceCorpus,
    EvidenceCorpusBuilder,
    EvidenceEntry,
    EvidenceResolver,
)
from ux_analyzer.application.experiment import ExperimentResult
from ux_analyzer.application.run_agent import RunResult
from ux_analyzer.domain.expectations import ExpectationKey, FrozenExpectation
from ux_analyzer.domain.findings import EvidenceClass
from ux_analyzer.domain.run import VerificationResult, VerifiedSuccess
from ux_analyzer.domain.synthesis import EvidenceRef
from ux_analyzer.ports.artifacts import (
    SaliencyArtifactKind,
    canonicalize_saliency_artifact_content,
    required_saliency_artifact_paths,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


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


def _screenshot_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), color=(200, 200, 200)).save(output, format="PNG")
    return output.getvalue()


def _heatmap_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("L", (2, 2), color=160).save(output, format="PNG")
    return output.getvalue()


def _native_map_bytes() -> bytes:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        values=np.full((2, 2), 0.5, dtype=np.float32),
        geometry=np.asarray(
            [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            dtype=np.float64,
        ),
    )
    return output.getvalue()


def _spec() -> SimpleNamespace:
    return SimpleNamespace(
        run_id="run-saliency",
        seed=7,
        model_trial=1,
        config_digest="fixture-config",
        policy=SimpleNamespace(value="progressive-prominence-scent"),
        prominence_provider_id="foveacast",
        scenario=SimpleNamespace(
            id="invite", name="Invite teammate", goal="Invite a teammate"
        ),
        application_version=SimpleNamespace(id="improved", label="Improved"),
        persona=SimpleNamespace(id="first-time", name="First-time teammate"),
    )


def _expectations() -> dict[ExpectationKey, FrozenExpectation]:
    key = ExpectationKey("improved", "invite", "first-time")
    return {
        key: FrozenExpectation(
            expectation_id="invite-first-time-v1",
            schema_version="frozen-expectation-v1",
            key=key,
            desired_outcomes=("A teammate receives a valid invitation.",),
        )
    }


def _write_valid_saliency_experiment(tmp_path: Path) -> tuple[ExperimentResult, Path]:
    spec = _spec()
    run = tmp_path / "runs" / spec.run_id
    saliency_root = run / "saliency" / "inference-1"
    artifacts_root = run / "artifacts"
    saliency_root.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    screenshot = _screenshot_bytes()
    (artifacts_root / "screenshot.png").write_bytes(screenshot)
    namespace = "inference-1"
    source_viewport_id = "viewport-1"
    checksums = ["1" * 64, "2" * 64, "3" * 64]
    geometry = {
        "geometry_version": "saliency-geometry-v1",
        "source_dimensions": [2, 2],
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
            "immediate": {"kind": "predicted", "score": 0.9, "source": "foveacast"},
            "early": {"kind": "predicted", "score": 0.7, "source": "foveacast"},
            "eventual": {"kind": "predicted", "score": 0.5, "source": "foveacast"},
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
                    "clipped_area": 1.0,
                    "visibility_fraction": 1.0,
                    "occlusion_fraction": 0.0,
                    "raw_score": 0.5,
                    "adjusted_score": 0.5,
                }
                for duration in ("1s", "3s", "7s")
            ],
            "aggregation_version": "element-saliency-aggregation-v1",
            "prediction_provenance": [
                {"duration": duration, "metadata": prediction_metadata(index)}
                for index, duration in enumerate(("1s", "3s", "7s"))
            ],
        }
    ]
    cache_key = {
        "viewport_id": namespace,
        "screenshot_sha256": hashlib.sha256(screenshot).hexdigest(),
        "screenshot_dimensions": [2, 2],
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
                }
            ],
            "aggregation_version": "element-saliency-aggregation-v1",
            "warnings": ["overlay-redacted"],
        },
        "warnings": ["overlay-redacted"],
        "artifact_paths": list(required_saliency_artifact_paths(namespace)),
    }
    (saliency_root / "profiles.json").write_bytes(
        canonicalize_saliency_artifact_content(
            SaliencyArtifactKind.PROFILES,
            json.dumps(profiles, sort_keys=True, separators=(",", ":")).encode(),
            expected_viewport_id=namespace,
        )
    )
    (saliency_root / "metadata.json").write_bytes(
        canonicalize_saliency_artifact_content(
            SaliencyArtifactKind.METADATA,
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(),
            expected_viewport_id=namespace,
        )
    )
    for duration in ("1s", "3s", "7s"):
        (saliency_root / f"{duration}.npz").write_bytes(_native_map_bytes())
        (saliency_root / f"{duration}-heatmap.png").write_bytes(_heatmap_bytes())

    snapshot = {
        "id": source_viewport_id,
        "screenshot_artifact": "artifacts/screenshot.png",
        "elements": [
            {
                "id": "target",
                "role": "button",
                "label": "Invite teammate",
                "bounds": {"x": 10, "y": 20, "width": 100, "height": 30},
                "visibility_fraction": 1.0,
                "occlusion_fraction": 0.0,
                "actionable": True,
                "disabled": False,
            }
        ],
        "regions": [],
    }
    artifact_ids = list(required_saliency_artifact_paths(namespace))
    events = [
        {
            "sequence": 1,
            "kind": "viewport-captured",
            "viewport_width": 2,
            "viewport_height": 2,
            "snapshot": snapshot,
        },
        {
            "sequence": 2,
            "kind": "saliency-inference-recorded",
            "viewport_id": namespace,
            "source_viewport_id": source_viewport_id,
            "artifact_namespace": namespace,
            "source_event_id": "event-1",
            "provider_id": "foveacast",
            "search_stage": "initial",
            "model_checksums": checksums,
            "execution_provider": "CPUExecutionProvider",
            "cache_state": "miss",
            "artifact_ids": artifact_ids,
            "warnings": ["overlay-redacted"],
        },
        {
            "sequence": 3,
            "kind": "saliency-profiles-recorded",
            "viewport_id": namespace,
            "source_viewport_id": source_viewport_id,
            "artifact_namespace": namespace,
            "source_event_id": "event-2",
            "provider_id": "foveacast",
            "search_stage": "initial",
            "model_checksums": checksums,
            "execution_provider": "CPUExecutionProvider",
            "cache_state": "miss",
            "artifact_ids": artifact_ids,
            "warnings": ["overlay-redacted"],
        },
        {
            "sequence": 4,
            "kind": "prominence-recorded",
            "viewport_id": source_viewport_id,
            "source_viewport_id": source_viewport_id,
            "artifact_namespace": namespace,
            "source_event_id": "event-3",
            "provider_id": "foveacast-prominence",
            "active_provider_id": "foveacast",
            "search_stage": "initial",
            "selected_mixture": [["1s", 1.0]],
            "selected_element_ids": ["target"],
            "cache_state": "miss",
        },
        {
            "sequence": 5,
            "kind": "action-executed",
            "viewport_id": source_viewport_id,
            "action": {"kind": "interact-with-element", "element_id": "target"},
            "succeeded": True,
        },
        {"sequence": 6, "kind": "run-terminated", "outcome": {"kind": "success"}},
    ]
    (run / "timeline.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    raw_spec = {
        "run_id": spec.run_id,
        "seed": spec.seed,
        "model_trial": spec.model_trial,
        "policy": spec.policy.value,
        "prominence_provider_id": spec.prominence_provider_id,
        "config_digest": "fixture-config",
        "scenario": {
            "id": spec.scenario.id,
            "name": spec.scenario.name,
            "goal": spec.scenario.goal,
        },
        "application_version": {
            "id": spec.application_version.id,
            "label": spec.application_version.label,
        },
        "persona": {"id": spec.persona.id, "name": spec.persona.name},
    }
    _write_json(
        run / "manifest.json",
        {
            "run_id": spec.run_id,
            "seed": spec.seed,
            "model_trial": spec.model_trial,
            "config_digest": "fixture-config",
            "scenario_id": spec.scenario.id,
            "application_version_id": spec.application_version.id,
            "persona_id": spec.persona.id,
            "policy": spec.policy.value,
            "prominence_provider_id": "foveacast",
            "endpoint_origin": "internal",
        },
    )
    _write_json(
        run / "result.json",
        {
            "run_id": spec.run_id,
            "state": {"spec": raw_spec},
            "outcome": {"kind": "verified-success"},
            "verification": {"verified": True, "evidence_ids": ["verify-1"]},
            "metrics": {
                "run_id": spec.run_id,
                "seed": spec.seed,
                "model_trial": spec.model_trial,
                "scenario_id": spec.scenario.id,
                "application_version_id": spec.application_version.id,
                "persona_id": spec.persona.id,
                "policy": spec.policy.value,
                "config_digest": "fixture-config",
                "verified_completion": True,
                "wrong_actions": 0,
                "prominence_provider_id": "foveacast",
            },
        },
    )
    _write_checksums(run)
    _write_json(
        tmp_path / "experiment.json",
        {
            "run_metrics": [
                {
                    "run_id": spec.run_id,
                    "seed": spec.seed,
                    "model_trial": spec.model_trial,
                    "scenario_id": spec.scenario.id,
                    "application_version_id": spec.application_version.id,
                    "persona_id": spec.persona.id,
                    "policy": spec.policy.value,
                    "prominence_provider_id": "foveacast",
                    "config_digest": "fixture-config",
                }
            ],
            "findings": [{"title": "prior finding prose sentinel"}],
        },
    )
    result = RunResult(
        run_id=spec.run_id,
        outcome=VerifiedSuccess(),
        verification=VerificationResult(verified=True, evidence_ids=("verify-1",)),
        agent_claimed_success=True,
        state=SimpleNamespace(spec=spec),
        bundle_path=run,
        findings=(),
    )
    return ExperimentResult(specs=(spec,), results=(result,), failures=()), run


def test_validated_heatmap_and_native_map_are_bounded_retrieval_entries(
    tmp_path: Path,
) -> None:
    experiment, _ = _write_valid_saliency_experiment(tmp_path)
    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())

    heatmap = next(entry for entry in corpus.entries if entry.ref.kind == "heatmap")
    native_map = next(
        entry for entry in corpus.entries if entry.ref.kind == "native-map"
    )
    assert heatmap.payload["namespace"] == "inference-1"
    assert (
        heatmap.payload["source_screenshot_sha256"]
        == hashlib.sha256(_screenshot_bytes()).hexdigest()
    )
    assert heatmap.payload["replay_linkage"] is True
    assert native_map.payload["provider_id"] == "foveacast"

    resolved = EvidenceResolver().resolve(
        corpus,
        (heatmap.ref.evidence_id,),
        max_entries=1,
        max_attachment_bytes=1_000_000,
    )
    assert resolved.entries[0].attachment_path is not None
    assert resolved.entries[0].attachment_path.as_posix().endswith("1s-heatmap.png")


def test_forged_saliency_linkage_is_not_published_as_heatmap_evidence(
    tmp_path: Path,
) -> None:
    experiment, run = _write_valid_saliency_experiment(tmp_path)
    timeline_path = run / "timeline.jsonl"
    events = [json.loads(line) for line in timeline_path.read_text().splitlines()]
    prominence = next(item for item in events if item["kind"] == "prominence-recorded")
    prominence["source_event_id"] = "event-1"
    timeline_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    _write_checksums(run)

    assert (
        finalized_bundle_failures(
            run,
            expected_run_id="run-saliency",
            expected_prominence_provider_id="foveacast",
        )
        == []
    )

    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())
    assert not [entry for entry in corpus.entries if entry.ref.kind == "heatmap"]
    assert not [entry for entry in corpus.entries if entry.ref.kind == "native-map"]


def test_resolver_rejects_symlink_and_unsupported_media(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "run-a"
    artifact_root = run / "artifacts"
    artifact_root.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("not an image", encoding="utf-8")
    (artifact_root / "outside.txt").write_text("not an image", encoding="utf-8")
    media = artifact_root / "media.txt"
    media.write_text("different non-image", encoding="utf-8")
    linked = artifact_root / "linked.txt"
    try:
        linked.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    symlink_entry = EvidenceEntry(
        ref=EvidenceRef(
            f"screenshot:run-a:{hashlib.sha256(outside.read_bytes()).hexdigest()}",
            "screenshot",
            "run-a",
            artifact_path="runs/run-a/artifacts/linked.txt",
            sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
        ),
        evidence_class=EvidenceClass.DETERMINISTIC_FACT,
        summary="linked file",
        payload={"media_type": "image/png"},
        attachment_path=Path("runs/run-a/artifacts/linked.txt"),
    )
    media_entry = EvidenceEntry(
        ref=EvidenceRef(
            f"screenshot:run-a:{hashlib.sha256(media.read_bytes()).hexdigest()}",
            "screenshot",
            "run-a",
            artifact_path="runs/run-a/artifacts/media.txt",
            sha256=hashlib.sha256(media.read_bytes()).hexdigest(),
        ),
        evidence_class=EvidenceClass.DETERMINISTIC_FACT,
        summary="text file",
        payload={"media_type": "text/plain"},
        attachment_path=Path("runs/run-a/artifacts/media.txt"),
    )
    corpus = EvidenceCorpus(output_root=tmp_path, entries=(symlink_entry, media_entry))
    resolver = EvidenceResolver()
    with pytest.raises(ValueError, match="symlink|reparse|link"):
        resolver.resolve(
            corpus,
            (symlink_entry.ref.evidence_id,),
            max_entries=1,
            max_attachment_bytes=1_000_000,
        )
    with pytest.raises(ValueError, match="media|image"):
        resolver.resolve(
            corpus,
            (media_entry.ref.evidence_id,),
            max_entries=1,
            max_attachment_bytes=1_000_000,
        )


def test_resolver_rejects_oversized_heatmap_attachment(tmp_path: Path) -> None:
    experiment, _ = _write_valid_saliency_experiment(tmp_path)
    corpus = EvidenceCorpusBuilder().build(experiment, tmp_path, _expectations())
    heatmap = next(entry for entry in corpus.entries if entry.ref.kind == "heatmap")

    with pytest.raises(ValueError, match="attachment"):
        EvidenceResolver().resolve(
            corpus,
            (heatmap.ref.evidence_id,),
            max_entries=1,
            max_attachment_bytes=1,
        )
