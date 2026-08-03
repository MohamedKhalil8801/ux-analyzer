from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import zipfile
import zlib
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

from ux_analyzer.application.run_agent import (
    RunProfiler,
    _ProfiledRunBundleWriter,
)
from ux_analyzer.domain.run import ProviderManifest, RunStarted
from ux_analyzer.ports.artifacts import (
    ArtifactReference,
    BundleAlreadyFinalizedError,
    BundleManifest,
    BundleStateError,
    RedactionPolicy,
    RunBundleWriter,
    SaliencyArtifactKind,
    SaliencyCacheHitEvent,
    sanitize_artifact_content,
)
from ux_analyzer.storage.run_bundle import FilesystemRunBundleWriter


def _png_with_trailing_data(data: bytes) -> bytes:
    raw = b"\x00" + b"\xff\x00\x00\xff"
    compressed = zlib.compress(raw)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
        + data
    )


def _native_map_bytes() -> bytes:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        values=np.asarray([[0.0, 0.25], [0.75, 1.0]], dtype=np.float32),
        geometry=np.asarray(
            [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            dtype=np.float64,
        ),
    )
    return output.getvalue()


def _heatmap_bytes(*, mode: str = "L") -> bytes:
    pixels = np.asarray([[0, 255]], dtype=np.uint8)
    if mode == "RGB":
        pixels = np.stack((pixels, pixels, pixels), axis=-1)
    output = io.BytesIO()
    Image.fromarray(pixels, mode=mode).save(output, format="PNG")
    return output.getvalue()


def _saliency_profile_bytes(
    *,
    viewport_id: str = "viewport-1",
    element_id: str = "button-1",
) -> bytes:
    metadata = {
        "provider_id": "foveacast",
        "model_id": "foveacast-v0.2.0",
        "provider_version": "foveacast-adapter-v1",
        "model_version": "v0.2.0",
        "model_checksum": "1" * 64,
        "input_dimensions": [2, 2],
        "output_dimensions": [2, 2],
        "geometry": {
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
        },
        "preprocessing_version": "foveacast-preprocess-v1",
        "inference_duration_ms": 1.0,
        "execution_provider": "CPUExecutionProvider",
        "warnings": [],
        "cache_state": "miss",
    }
    aggregate = {
        "viewport_id": viewport_id,
        "element_id": element_id,
        "duration": "1s",
        "density": 0.4,
        "robust_peak": 0.9,
        "raw_mass": 1.2,
        "mass_share": 0.3,
        "clipped_area": 4.0,
        "visibility_fraction": 1.0,
        "occlusion_fraction": 0.0,
        "raw_score": 0.6,
        "adjusted_score": 0.6,
    }
    profile = {
        "viewport_id": viewport_id,
        "element_id": element_id,
        "immediate": {"kind": "predicted", "score": 0.8, "source": "foveacast"},
        "early": None,
        "eventual": None,
        "general": None,
        "aggregates": [aggregate],
        "aggregation_version": "element-saliency-aggregation-v1",
        "prediction_provenance": [{"duration": "1s", "metadata": metadata}],
    }
    return json.dumps([profile]).encode("utf-8")


def _saliency_metadata_bytes() -> bytes:
    profile = json.loads(_saliency_profile_bytes())[0]
    base_prediction_metadata = profile["prediction_provenance"][0]["metadata"]
    checksums = ("1" * 64, "2" * 64, "3" * 64)
    predictions = [
        {
            "duration": duration,
            "metadata": {
                **base_prediction_metadata,
                "model_checksum": checksum,
            },
        }
        for duration, checksum in zip(("1s", "3s", "7s"), checksums, strict=True)
    ]
    cache_key = {
        "viewport_id": "viewport-1",
        "screenshot_sha256": "a" * 64,
        "screenshot_dimensions": [2, 2],
        "device_pixel_ratio": 1.0,
        "zoom": 1.0,
        "model_checksums": list(checksums),
        "preprocessing_version": "foveacast-preprocess-v1",
        "precision": "fp16",
        "execution_provider": "CPUExecutionProvider",
        "aggregation_version": "element-saliency-aggregation-v1",
        "geometry_version": "saliency-geometry-v1",
    }
    cache_key_digest = hashlib.sha256(
        json.dumps(cache_key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "cache_version": "saliency-cache-v1",
        "cache_key": cache_key,
        "cache_key_digest": cache_key_digest,
        "viewport_id": "viewport-1",
        "aggregation_version": "element-saliency-aggregation-v1",
        "predictions": predictions,
        "saliency_metadata": {
            "provider_manifests": [
                {
                    "provider_id": "foveacast",
                    "role": "prominence",
                    "model_id": "foveacast-v0.2.0",
                    "endpoint_origin": "foveacast.invalid",
                    "version": "v0.2.0",
                    "prompt_version": None,
                    "schema_version": None,
                }
            ],
            "aggregation_version": "element-saliency-aggregation-v1",
            "warnings": [],
        },
        "warnings": [],
        "artifact_paths": [
            "saliency/viewport-1/1s.npz",
            "saliency/viewport-1/3s.npz",
            "saliency/viewport-1/7s.npz",
            "saliency/viewport-1/1s-heatmap.png",
            "saliency/viewport-1/3s-heatmap.png",
            "saliency/viewport-1/7s-heatmap.png",
            "saliency/viewport-1/profiles.json",
            "saliency/viewport-1/metadata.json",
        ],
    }
    return json.dumps(metadata).encode("utf-8")


def _write_complete_saliency_artifacts(
    writer: RunBundleWriter,
) -> tuple[ArtifactReference, ...]:
    references = [
        writer.write_saliency_artifact(
            f"saliency/viewport-1/{duration}.npz",
            _native_map_bytes(),
            SaliencyArtifactKind.NATIVE_MAP,
        )
        for duration in ("1s", "3s", "7s")
    ]
    references.extend(
        writer.write_saliency_artifact(
            f"saliency/viewport-1/{duration}-heatmap.png",
            _heatmap_bytes(),
            SaliencyArtifactKind.HEATMAP,
        )
        for duration in ("1s", "3s", "7s")
    )
    references.append(
        writer.write_saliency_artifact(
            "saliency/viewport-1/profiles.json",
            _saliency_profile_bytes(),
            SaliencyArtifactKind.PROFILES,
        )
    )
    references.append(
        writer.write_saliency_artifact(
            "saliency/viewport-1/metadata.json",
            _saliency_metadata_bytes(),
            SaliencyArtifactKind.METADATA,
        )
    )
    return tuple(references)


def _append_complete_saliency_event(
    writer: RunBundleWriter,
    *,
    warnings: tuple[str, ...] = (),
) -> tuple[ArtifactReference, ...]:
    references = _write_complete_saliency_artifacts(writer)
    writer.append_saliency_event(
        _saliency_event(warnings=warnings, artifact_checksums=references)
    )
    return references


def bundle_manifest(run_id: str = "run-1") -> BundleManifest:
    return BundleManifest(
        run_id=run_id,
        seed=17,
        model_trial=2,
        config_digest="config-sha256",
        endpoint_origin="https://llm.example.test/v1",
        model_ids={"scent": "scent-model", "cognitive": "cognitive-model"},
        prompt_versions={"scent": "scent-v1", "cognitive": "cognitive-v1"},
        package_version="0.1.0",
        provider_versions={"observation": "fixture-1", "model": "adapter-2"},
        provider_manifests=(
            ProviderManifest(
                provider_id="fixture-provider",
                role="observation",
                model_id=None,
                endpoint_origin="fixture.invalid",
                version="fixture-1",
            ),
        ),
    )


def _saliency_event(
    *,
    endpoint_origin: str = "foveacast.invalid",
    warnings: tuple[str, ...] = (),
    artifact_path: str | None = None,
    artifact_checksums: tuple[ArtifactReference, ...] | None = None,
) -> SaliencyCacheHitEvent:
    artifact_paths = (
        (artifact_path,)
        if artifact_path is not None
        else (
            "saliency/viewport-1/1s.npz",
            "saliency/viewport-1/3s.npz",
            "saliency/viewport-1/7s.npz",
            "saliency/viewport-1/1s-heatmap.png",
            "saliency/viewport-1/3s-heatmap.png",
            "saliency/viewport-1/7s-heatmap.png",
            "saliency/viewport-1/profiles.json",
            "saliency/viewport-1/metadata.json",
        )
    )
    return SaliencyCacheHitEvent(
        cache_key="a" * 64,
        viewport_id="viewport-1",
        execution_provider="CPUExecutionProvider",
        model_checksums=("1" * 64, "2" * 64, "3" * 64),
        preprocessing_version="foveacast-preprocess-v1",
        precision="fp16",
        provider_manifests=(
            ProviderManifest(
                provider_id="foveacast",
                role="prominence",
                model_id="foveacast-v0.2.0",
                endpoint_origin=endpoint_origin,
                version="v0.2.0",
            ),
        ),
        artifact_checksums=(
            artifact_checksums
            if artifact_checksums is not None
            else tuple(
                ArtifactReference(
                    path=path,
                    sha256=f"{index:x}" * 64,
                    size=index,
                    name=path,
                )
                for index, path in enumerate(artifact_paths, start=1)
            )
        ),
        warnings=warnings,
    )


def test_manifest_persists_default_model_trial_zero() -> None:
    manifest = BundleManifest(
        run_id="run-default-trial",
        seed=17,
        config_digest="config-sha256",
        endpoint_origin="https://llm.example.test/v1",
    )

    assert manifest.to_dict()["model_trial"] == 0


def test_mapping_manifest_preserves_prominence_provider_and_legacy_default(
    tmp_path: Path,
) -> None:
    common = {
        "seed": 17,
        "config_digest": "config-sha256",
        "endpoint_origin": "https://llm.example.test/v1",
    }
    provider_writer = FilesystemRunBundleWriter.start(
        tmp_path,
        {
            **common,
            "run_id": "run-foveacast",
            "prominence_provider_id": "foveacast",
        },
    )
    provider_manifest = json.loads(
        (provider_writer.staging_path / "manifest.json").read_text()
    )
    provider_writer.abort("test complete")

    legacy_writer = FilesystemRunBundleWriter.start(
        tmp_path,
        {**common, "run_id": "run-legacy"},
    )
    legacy_manifest = json.loads(
        (legacy_writer.staging_path / "manifest.json").read_text()
    )
    legacy_writer.abort("test complete")

    assert provider_manifest["prominence_provider_id"] == "foveacast"
    assert legacy_manifest["prominence_provider_id"] == "heuristic"


def test_start_uses_atomic_staging_and_finalize_creates_required_files(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    assert writer.staging_path == tmp_path / ".staging" / "run-1"
    assert writer.final_path == tmp_path / "runs" / "run-1"
    assert writer.staging_path.is_dir()
    assert not writer.final_path.exists()

    writer.append_event(RunStarted(run_id="run-1"))
    final_path = writer.finalize({"outcome": "verified-success"})

    assert final_path == tmp_path / "runs" / "run-1"
    assert not writer.staging_path.exists()
    assert (final_path / "manifest.json").is_file()
    assert (final_path / "timeline.jsonl").is_file()
    assert (final_path / "artifacts").is_dir()
    assert (final_path / "checksums.sha256").is_file()
    assert (final_path / "result.json").is_file()


def test_start_rejects_symlinked_staging_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".staging").symlink_to(outside, target_is_directory=True)

    with pytest.raises(BundleStateError, match="symlink"):
        FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())


def test_start_rejects_symlinked_output_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    redirected = tmp_path / "redirected"
    try:
        redirected.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(BundleStateError, match="symlink"):
        FilesystemRunBundleWriter.start(
            redirected / "experiment-output", bundle_manifest()
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point check")
def test_start_rejects_windows_reparse_output_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    redirected = tmp_path / "redirected"
    try:
        redirected.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Windows reparse fixture unavailable: {error}")

    with pytest.raises(BundleStateError, match="reparse|symlink"):
        FilesystemRunBundleWriter.start(
            redirected / "experiment-output", bundle_manifest()
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction check")
def test_start_rejects_windows_junction_output_ancestor(tmp_path: Path) -> None:
    import subprocess

    outside = tmp_path / "outside-junction-target"
    outside.mkdir()
    redirected = tmp_path / "redirected-junction"
    created = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(redirected), str(outside)],
        capture_output=True,
        check=False,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip(f"Windows junction fixture unavailable: {created.stderr.strip()}")

    try:
        with pytest.raises(BundleStateError, match="reparse|symlink"):
            FilesystemRunBundleWriter.start(
                redirected / "experiment-output", bundle_manifest()
            )
    finally:
        redirected.rmdir()


def test_events_are_flushed_redacted_and_assigned_monotonic_sequences(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(
        tmp_path,
        bundle_manifest(),
        redaction=RedactionPolicy(
            exact_values=("fixture-secret", "api-secret"),
            keys=frozenset({"password", "api_key"}),
        ),
    )

    writer.append_event(
        {
            "kind": "run-started",
            "password": "fixture-secret",
            "nested": {"token": "api-secret"},
        }
    )
    assert writer.timeline_path.read_text(encoding="utf-8").count("\n") == 1
    writer.append_event({"kind": "checkpoint", "sequence": 999})

    events = [
        json.loads(line)
        for line in writer.timeline_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["sequence"] for event in events] == [1, 2]
    assert "fixture-secret" not in writer.timeline_path.read_text(encoding="utf-8")
    assert "api-secret" not in writer.timeline_path.read_text(encoding="utf-8")
    assert events[0]["password"] == "[REDACTED]"


def test_untyped_saliency_event_cannot_enter_jsonl(tmp_path: Path) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    with pytest.raises((TypeError, ValueError)):
        writer.append_event(
            {
                "kind": "saliency-cache-hit",
                "api_key": "api-secret",
                "selector": "[data-secret]",
                "raw_map": [0.1, 0.2],
            }
        )


def test_typed_saliency_event_is_rejected_before_generic_serialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    def fail_serialization(value: object) -> object:
        del value
        raise AssertionError("typed saliency event reached generic serializer")

    monkeypatch.setattr(
        "ux_analyzer.storage.run_bundle._json_value", fail_serialization
    )

    with pytest.raises(TypeError, match="typed append_saliency_event"):
        writer.append_event(
            _saliency_event(
                warnings=("Bearer fixture-secret", "/Users/fixture-secret/path")
            )
        )
    assert writer.timeline_path.read_text(encoding="utf-8") == ""


def test_typed_saliency_event_rejects_arbitrary_artifact_path() -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        _saliency_event(artifact_path="saliency/viewport-1/arbitrary-name.npz")


def test_typed_saliency_event_rejects_selector_artifact_path() -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        _saliency_event(artifact_path="saliency/[data-secret]/1s.npz")


def test_typed_saliency_event_rejects_cross_viewport_artifact_path() -> None:
    with pytest.raises(ValueError, match="viewport"):
        _saliency_event(artifact_path="saliency/viewport-2/1s.npz")


def test_typed_saliency_event_requires_complete_viewport_artifact_set() -> None:
    with pytest.raises(ValueError, match="complete|exact"):
        _saliency_event(artifact_path="saliency/viewport-1/1s.npz")


def test_typed_saliency_event_rejects_missing_bundle_artifact_before_append(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    with pytest.raises(BundleStateError, match="missing|exists|artifact"):
        writer.append_saliency_event(_saliency_event())

    assert writer.timeline_path.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("corrupted_field", ("sha256", "size"))
def test_typed_saliency_event_rejects_forged_bundle_reference(
    tmp_path: Path,
    corrupted_field: str,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    references = _write_complete_saliency_artifacts(writer)
    first = references[0]
    forged = replace(
        first,
        sha256="f" * 64 if corrupted_field == "sha256" else first.sha256,
        size=first.size + 1 if corrupted_field == "size" else first.size,
    )
    event = _saliency_event(
        artifact_checksums=(forged,) + references[1:],
    )

    with pytest.raises(BundleStateError, match=corrupted_field):
        writer.append_saliency_event(event)


def test_finalize_rejects_tampered_saliency_reference(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    references = _write_complete_saliency_artifacts(writer)
    writer.append_saliency_event(_saliency_event(artifact_checksums=references))
    tampered_path = writer.staging_path / references[0].path
    tampered_path.write_bytes(tampered_path.read_bytes() + b"tampered")

    with pytest.raises(BundleStateError, match="sha256|size|tamper"):
        writer.finalize({"status": "complete"})


@pytest.mark.parametrize(
    ("forged_field", "expected_message"),
    (
        ("cache_key_digest", "digest"),
        ("source_dimensions", "dimensions"),
        ("model_checksum", "checksum"),
        ("zoom", "geometry"),
        ("duration", "duration"),
        ("content_dimensions", "geometry"),
        ("scale", "geometry"),
        ("output_dimensions", "dimensions"),
        ("inference_duration_ms", "duration"),
        ("cache_state", "cache_state"),
    ),
)
def test_typed_saliency_metadata_rejects_forged_provenance(
    tmp_path: Path,
    forged_field: str,
    expected_message: str,
) -> None:
    metadata = json.loads(_saliency_metadata_bytes())
    if forged_field == "cache_key_digest":
        metadata[forged_field] = "b" * 64
    elif forged_field == "source_dimensions":
        metadata["predictions"][0]["metadata"]["geometry"][forged_field] = [3, 2]
    elif forged_field == "model_checksum":
        metadata["predictions"][1]["metadata"][forged_field] = "4" * 64
    elif forged_field == "zoom":
        metadata["predictions"][0]["metadata"]["geometry"][forged_field] = 2.0
    elif forged_field == "content_dimensions":
        for prediction in metadata["predictions"]:
            prediction["metadata"]["geometry"][forged_field] = [1, 2]
    elif forged_field == "scale":
        for prediction in metadata["predictions"]:
            prediction["metadata"]["geometry"][forged_field] = 0.0
    elif forged_field == "output_dimensions":
        metadata["predictions"][1]["metadata"][forged_field] = [3, 2]
    elif forged_field == "inference_duration_ms":
        metadata["predictions"][0]["metadata"][forged_field] = -1.0
    elif forged_field == "cache_state":
        metadata["predictions"][0]["metadata"][forged_field] = "hit"
    else:
        metadata["predictions"][1][forged_field] = "1s"

    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    with pytest.raises(ValueError, match=expected_message):
        writer.write_saliency_artifact(
            "saliency/viewport-1/metadata.json",
            json.dumps(metadata).encode("utf-8"),
            SaliencyArtifactKind.METADATA,
        )


def test_typed_saliency_event_rejects_credentialed_endpoint() -> None:
    with pytest.raises(ValueError, match="credentials"):
        _saliency_event(endpoint_origin="https://user:password@example.test")


def test_saliency_event_warnings_remove_credentials_and_sensitive_paths(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(
        tmp_path,
        bundle_manifest(),
        redaction=RedactionPolicy(exact_values=("fixture-secret",)),
    )

    _append_complete_saliency_event(
        writer,
        warnings=(
            "Bearer bearer-secret",
            "API key: api-secret",
            "https://user:password@example.test/private",
            "/Users/fixture-secret/private-cache",
            "selector=[data-secret]",
        ),
    )

    timeline = writer.timeline_path.read_text(encoding="utf-8")
    assert "bearer-secret" not in timeline
    assert "api-secret" not in timeline
    assert "user:password@" not in timeline
    assert "/Users/fixture-secret/private-cache" not in timeline
    assert "[data-secret]" not in timeline
    assert "[REDACTED]" in timeline


def test_default_saliency_warning_redaction_catches_unlabelled_secrets(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    _append_complete_saliency_event(
        writer,
        warnings=(
            "API_KEY=unlabelled-api-key",
            "Bearer unlabelled-bearer-token",
            "https://user:unlabelled-password@example.test/private",
            "fixture-secret",
            "ToKeN=unlabelled-token",
            "PASSWORD = unlabelled-password",
            "secret: unlabelled-secret",
            "AUTHORIZATION=unlabelled-authorization",
        ),
    )
    writer.append_event(
        {
            "kind": "diagnostic",
            "API_KEY": "unlabelled-api-key",
            "Fixture_Secret": "fixture-secret",
        }
    )

    timeline = writer.timeline_path.read_text(encoding="utf-8")
    for secret in (
        "unlabelled-api-key",
        "unlabelled-bearer-token",
        "unlabelled-password",
        "unlabelled-token",
        "unlabelled-secret",
        "unlabelled-authorization",
        "fixture-secret",
    ):
        assert secret not in timeline


def test_default_saliency_warning_redaction_catches_empty_assignments(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    _append_complete_saliency_event(
        writer,
        warnings=(
            "token=",
            "PASSWORD = ",
            "secret:",
            "AUTHORIZATION=, next",
        ),
    )

    timeline = writer.timeline_path.read_text(encoding="utf-8")
    assert "token=" not in timeline
    assert "PASSWORD =" not in timeline
    assert "secret:" not in timeline
    assert "AUTHORIZATION=" not in timeline


def test_saliency_warning_redaction_covers_spaced_and_quoted_assignments(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    _append_complete_saliency_event(
        writer,
        warnings=(
            "token = secret spaces",
            'PASSWORD : "quoted secret spaces"',
            "Api_Key = 'api key secret'",
        ),
    )

    timeline = writer.timeline_path.read_text(encoding="utf-8")
    for secret in ("secret spaces", "quoted secret spaces", "api key secret"):
        assert secret not in timeline


@pytest.mark.parametrize(
    ("assignment", "secret"),
    (
        ("token = secret spaces", "secret spaces"),
        ('password : "quoted password spaces"', "quoted password spaces"),
        ("secret = 'quoted secret spaces'", "quoted secret spaces"),
        ("authorization = Bearer authorization-secret", "authorization-secret"),
    ),
)
def test_default_artifact_sanitizer_redacts_sensitive_assignments(
    assignment: str,
    secret: str,
) -> None:
    sanitized = sanitize_artifact_content(
        "diagnostic.txt",
        assignment.encode("utf-8"),
        RedactionPolicy(),
    )

    assert secret.encode("utf-8") not in sanitized
    assert b"[REDACTED]" in sanitized


@pytest.mark.parametrize(
    "payload",
    (
        b'{" API_KEY ": "raw-json-secret"}',
        b'{"api key": "padded-json-secret"}',
        b'{" api---key ": "repeated-padding-secret"}',
        b'password = "first-line\nsecond-line-secret"',
        b"token = 'quoted\nmultiline-token-secret'",
    ),
)
def test_artifact_sanitizer_redacts_padded_keys_and_multiline_assignments(
    payload: bytes,
) -> None:
    sanitized = sanitize_artifact_content("diagnostic.txt", payload, RedactionPolicy())

    assert b"secret" not in sanitized
    assert b"[REDACTED]" in sanitized


def test_corrupt_zip_fallback_still_sanitizes_textual_secret() -> None:
    corrupt_zip = b"\xffPK\x03\x04\"password\" = 'corrupt-zip-secret'"

    sanitized = sanitize_artifact_content("trace.zip", corrupt_zip, RedactionPolicy())

    assert b"corrupt-zip-secret" not in sanitized
    assert b"[REDACTED]" in sanitized


def test_event_rejects_bytes_before_base64_serialization(tmp_path: Path) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    with pytest.raises(TypeError, match="bytes"):
        writer.append_event({"kind": "diagnostic", "api_key": b"api-secret"})

    assert writer.timeline_path.read_bytes() == b""


@pytest.mark.parametrize(
    "endpoint",
    (
        "https://llm.example.test/v1?api_key=raw-secret",
        "https://llm.example.test/v1#raw-secret",
        "foveacast.invalid?raw-secret",
    ),
)
def test_provider_manifest_rejects_endpoint_query_and_fragment(endpoint: str) -> None:
    with pytest.raises(ValueError, match="query|fragment"):
        _saliency_event(endpoint_origin=endpoint)


def test_typed_saliency_json_redacts_sensitive_element_identifiers(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    writer.write_saliency_artifact(
        "saliency/viewport-1/profiles.json",
        _saliency_profile_bytes(element_id='TOKEN = "secret spaces"'),
        SaliencyArtifactKind.PROFILES,
    )

    content = (writer.staging_path / "saliency/viewport-1/profiles.json").read_text(
        encoding="utf-8"
    )
    assert "secret spaces" not in content
    assert "[REDACTED]" in content


def test_typed_saliency_json_recursively_redacts_identifiers_sources_and_warnings(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    profiles = json.loads(_saliency_profile_bytes())
    profile = profiles[0]
    sensitive_element_id = "secret-element-internal-value"
    profile["element_id"] = sensitive_element_id
    profile["aggregates"][0]["element_id"] = sensitive_element_id
    profile["immediate"]["source"] = "artifacts/token/internal-source-value"
    profile["prediction_provenance"][0]["metadata"]["warnings"] = [
        "runtime error reading password internal-warning-value"
    ]

    writer.write_saliency_artifact(
        "saliency/viewport-1/profiles.json",
        json.dumps(profiles).encode("utf-8"),
        SaliencyArtifactKind.PROFILES,
    )

    content = (writer.staging_path / "saliency/viewport-1/profiles.json").read_text(
        encoding="utf-8"
    )
    for sensitive in (
        sensitive_element_id,
        "internal-source-value",
        "internal-warning-value",
    ):
        assert sensitive not in content


def test_typed_saliency_json_rejects_redacted_element_identifier_collision(
    tmp_path: Path,
) -> None:
    profiles = json.loads(_saliency_profile_bytes())
    second_profile = json.loads(_saliency_profile_bytes())[0]
    for profile, element_id in zip(
        (profiles[0], second_profile), ("element-one", "element-two"), strict=True
    ):
        profile["element_id"] = element_id
        for aggregate in profile["aggregates"]:
            aggregate["element_id"] = element_id

    writer = FilesystemRunBundleWriter.start(
        tmp_path,
        bundle_manifest(),
        redaction=RedactionPolicy(exact_values=("element-one", "element-two")),
    )

    with pytest.raises(ValueError, match="duplicate profiles"):
        writer.write_saliency_artifact(
            "saliency/viewport-1/profiles.json",
            json.dumps([profiles[0], second_profile]).encode("utf-8"),
            SaliencyArtifactKind.PROFILES,
        )

    assert not (writer.staging_path / "saliency/viewport-1/profiles.json").exists()


def test_typed_saliency_json_rejects_error_field_without_persisting_value(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    profiles = json.loads(_saliency_profile_bytes())
    profiles[0]["prediction_provenance"][0]["metadata"]["error"] = (
        "authorization = internal-error-secret"
    )

    with pytest.raises(ValueError, match="allowlisted"):
        writer.write_saliency_artifact(
            "saliency/viewport-1/profiles.json",
            json.dumps(profiles).encode("utf-8"),
            SaliencyArtifactKind.PROFILES,
        )

    assert not (writer.staging_path / "saliency/viewport-1/profiles.json").exists()


def test_typed_saliency_json_rejects_cross_viewport_profile_reference(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    with pytest.raises(ValueError, match="viewport"):
        writer.write_saliency_artifact(
            "saliency/viewport-1/profiles.json",
            _saliency_profile_bytes(viewport_id="viewport-2"),
            SaliencyArtifactKind.PROFILES,
        )


def test_generic_writers_reject_saliency_paths(tmp_path: Path) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    with pytest.raises(ValueError, match="typed saliency writer"):
        writer.write_artifact("saliency/viewport-1/metadata.json", b"arbitrary")
    with pytest.raises(ValueError, match="typed saliency writer"):
        writer.write_named_artifact(
            "saliency/viewport-1/metadata.json", b'{"source_pixels":"secret"}'
        )


@pytest.mark.parametrize(
    ("name", "kind"),
    (
        ("saliency/viewport-1/profiles.json", SaliencyArtifactKind.PROFILES),
        ("saliency/viewport-1/metadata.json", SaliencyArtifactKind.METADATA),
    ),
)
def test_typed_saliency_json_rejects_source_pixel_payload(
    tmp_path: Path,
    name: str,
    kind: SaliencyArtifactKind,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    with pytest.raises(ValueError, match="allowlisted"):
        writer.write_saliency_artifact(
            name,
            b'{"source_pixels":"raw screenshot bytes"}',
            kind,
        )


def test_finalize_writes_manifest_result_and_checksums_without_secrets(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(
        tmp_path,
        bundle_manifest(),
        redaction=RedactionPolicy(
            exact_values=("api-secret",), keys=frozenset({"api_key"})
        ),
    )
    writer.append_event({"kind": "run-started"})
    artifact = writer.write_artifact("trace.zip", b"trace-bytes")
    final_path = writer.finalize(
        {"status": "complete", "api_key": "api-secret", "artifact": artifact.path}
    )

    manifest_text = (final_path / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["seed"] == 17
    assert manifest["model_trial"] == 2
    assert manifest["config_digest"] == "config-sha256"
    assert manifest["endpoint_origin"] == "https://llm.example.test"
    assert manifest["model_ids"] == {
        "scent": "scent-model",
        "cognitive": "cognitive-model",
    }
    assert manifest["prompt_versions"] == {
        "scent": "scent-v1",
        "cognitive": "cognitive-v1",
    }
    assert manifest["package_version"] == "0.1.0"
    assert manifest["provider_versions"] == {
        "observation": "fixture-1",
        "model": "adapter-2",
    }
    assert "api-secret" not in manifest_text
    assert "api-secret" not in (final_path / "result.json").read_text(encoding="utf-8")

    checksum_lines = (
        (final_path / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    )
    checksums = {
        path: digest
        for digest, path in (line.split("  ", maxsplit=1) for line in checksum_lines)
    }
    assert "checksums.sha256" not in checksums
    for relative_path, digest in checksums.items():
        actual = hashlib.sha256((final_path / relative_path).read_bytes()).hexdigest()
        assert actual == digest


def test_identical_artifacts_share_content_hash_path(tmp_path: Path) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    first = writer.write_artifact("first.png", b"same-content")
    second = writer.write_artifact("second.trace", b"same-content")

    assert first.sha256 == second.sha256
    assert first.path == second.path
    assert len(list((writer.staging_path / "artifacts").iterdir())) == 1


def test_named_artifact_preserves_saliency_evidence_path_and_checksum(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    reference = writer.write_saliency_artifact(
        "saliency/viewport-1/1s.npz",
        _native_map_bytes(),
        SaliencyArtifactKind.NATIVE_MAP,
    )
    final_path = writer.finalize({"status": "complete"})

    assert reference.path == "saliency/viewport-1/1s.npz"
    assert (final_path / reference.path).read_bytes() == _native_map_bytes()
    checksum_lines = (final_path / "checksums.sha256").read_text().splitlines()
    assert f"{reference.sha256}  {reference.path}" in checksum_lines


def test_saliency_artifact_rejects_non_allowlisted_filename(tmp_path: Path) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    with pytest.raises(ValueError, match="allowlisted"):
        writer.write_saliency_artifact(
            "saliency/viewport-1/arbitrary-name.npz",
            b"native-map",
            SaliencyArtifactKind.NATIVE_MAP,
        )


def test_native_saliency_artifact_rejects_token_like_viewport_path(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    with pytest.raises(ValueError, match="sensitive"):
        writer.write_saliency_artifact(
            "saliency/token=internal-path-secret/1s.npz",
            _native_map_bytes(),
            SaliencyArtifactKind.NATIVE_MAP,
        )


def test_saliency_artifact_rejects_malformed_native_map(tmp_path: Path) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    with pytest.raises(ValueError, match="native saliency map"):
        writer.write_saliency_artifact(
            "saliency/viewport-1/1s.npz",
            b"arbitrary-bytes",
            SaliencyArtifactKind.NATIVE_MAP,
        )


def test_saliency_artifact_rejects_source_pixel_heatmap(tmp_path: Path) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    with pytest.raises(ValueError, match="heatmap"):
        writer.write_saliency_artifact(
            "saliency/viewport-1/1s-heatmap.png",
            _heatmap_bytes(mode="RGB"),
            SaliencyArtifactKind.HEATMAP,
        )


def test_saliency_artifact_accepts_valid_native_map_and_heatmap(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    native = writer.write_saliency_artifact(
        "saliency/viewport-1/1s.npz",
        _native_map_bytes(),
        SaliencyArtifactKind.NATIVE_MAP,
    )
    heatmap = writer.write_saliency_artifact(
        "saliency/viewport-1/1s-heatmap.png",
        _heatmap_bytes(),
        SaliencyArtifactKind.HEATMAP,
    )

    assert native.size > 0
    assert heatmap.size > 0


def test_binary_artifacts_redact_sensitive_values_before_persistence(
    tmp_path: Path,
) -> None:
    sensitive_email = "invitee@example.test"
    sensitive_totp = "246810"
    trace_buffer = io.BytesIO()
    with zipfile.ZipFile(trace_buffer, "w") as archive:
        archive.writestr(
            "trace.trace",
            json.dumps({"input": sensitive_email, "code": sensitive_totp}),
        )
        archive.writestr(
            "resources/page.png",
            _png_with_trailing_data(sensitive_email.encode()),
        )

    writer = FilesystemRunBundleWriter.start(
        tmp_path,
        bundle_manifest(),
        redaction=RedactionPolicy(exact_values=(sensitive_email, sensitive_totp)),
    )
    screenshot = writer.write_artifact(
        "screenshot.png", _png_with_trailing_data(sensitive_totp.encode())
    )
    trace = writer.write_artifact("trace.zip", trace_buffer.getvalue())

    screenshot_bytes = (writer.staging_path / screenshot.path).read_bytes()
    trace_bytes = (writer.staging_path / trace.path).read_bytes()
    assert sensitive_email.encode() not in screenshot_bytes
    assert sensitive_totp.encode() not in screenshot_bytes
    assert sensitive_email.encode() not in trace_bytes
    assert sensitive_totp.encode() not in trace_bytes
    with zipfile.ZipFile(io.BytesIO(trace_bytes)) as archive:
        assert all(
            sensitive_email not in archive.read(name).decode("utf-8", "ignore")
            and sensitive_totp not in archive.read(name).decode("utf-8", "ignore")
            for name in archive.namelist()
        )


def test_abort_leaves_crash_recovery_marker_in_staging_directory(
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())

    writer.abort("provider stopped unexpectedly")

    marker_path = writer.staging_path / "crash.marker"
    assert marker_path.is_file()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["reason"] == "provider stopped unexpectedly"
    assert not writer.final_path.exists()


def test_finalized_bundle_refuses_all_mutation(tmp_path: Path) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    writer.finalize({"status": "complete"})

    with pytest.raises(BundleAlreadyFinalizedError):
        writer.append_event({"kind": "late-event"})
    with pytest.raises(BundleAlreadyFinalizedError):
        writer.write_artifact("late.txt", b"late")
    with pytest.raises(BundleAlreadyFinalizedError):
        writer.finalize({"status": "again"})
    with pytest.raises(BundleAlreadyFinalizedError):
        writer.abort("too late")


def test_profiled_writer_preserves_typed_artifact_contract(tmp_path: Path) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    profiler = RunProfiler(tmp_path / "profile.json")
    profiled = _ProfiledRunBundleWriter(writer, profiler)

    assert profiled.manifest is writer.manifest
    _append_complete_saliency_event(profiled)
    profiled.write_named_artifact("notes.json", "{}")
    profiled.write_saliency_artifact(
        "saliency/viewport-1/1s.npz",
        _native_map_bytes(),
        SaliencyArtifactKind.NATIVE_MAP,
    )
    final_path = profiled.finalize({"status": "complete"})
    profiler.write(run_id=profiled.run_id)

    assert final_path.is_dir()
    assert (final_path / "saliency/viewport-1/1s.npz").is_file()
    assert (tmp_path / "profile.json").is_file()


def test_atomic_publish_failure_leaves_internal_error_recovery_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    writer.append_event({"kind": "run-terminated", "outcome": "verified-success"})

    def fail_publish(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("atomic publish failed")

    target = (
        "ux_analyzer.storage.run_bundle._replace_windows_handle_relative"
        if os.name == "nt"
        else "ux_analyzer.storage.run_bundle.os.replace"
    )
    monkeypatch.setattr(target, fail_publish)

    with pytest.raises(OSError, match="atomic publish failed"):
        writer.finalize({"outcome": "verified-success"})

    marker = json.loads(
        (writer.staging_path / "crash.marker").read_text(encoding="utf-8")
    )
    assert "finalization failed" in marker["reason"]
    assert "atomic publish failed" in marker["reason"]
    assert marker["outcome"] == "internal-error"
    assert not writer.final_path.exists()


def test_artifact_write_rejects_ancestor_swap_before_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    outside = tmp_path / "outside"
    outside.mkdir()
    artifacts_root = writer.staging_path / "artifacts"
    backup_root = writer.staging_path / "artifacts-real"
    original_open = os.open
    swapped = False

    def race_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        nonlocal swapped
        if not swapped and Path(path).parent == artifacts_root:
            artifacts_root.rename(backup_root)
            artifacts_root.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_open(path, flags, mode)

    monkeypatch.setattr("ux_analyzer.storage.run_bundle.os.open", race_open)

    try:
        with pytest.raises(BundleStateError, match="symlink|reparse"):
            writer.write_artifact("escape.txt", b"must-not-escape")
    finally:
        if artifacts_root.is_symlink():
            artifacts_root.unlink()
        if backup_root.exists():
            backup_root.rename(artifacts_root)

    assert not (outside / "escape.txt").exists()


def test_artifact_write_rejects_ancestor_swap_restored_after_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    outside = tmp_path / "outside"
    outside.mkdir()
    artifacts_root = writer.staging_path / "artifacts"
    backup_root = writer.staging_path / "artifacts-real"
    original_open = os.open
    swapped = False

    def race_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        nonlocal swapped
        if not swapped and Path(path).parent == artifacts_root:
            artifacts_root.rename(backup_root)
            try:
                artifacts_root.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                backup_root.rename(artifacts_root)
                pytest.skip(f"symlink race fixture unavailable: {error}")
            descriptor = original_open(path, flags, mode)
            artifacts_root.unlink()
            backup_root.rename(artifacts_root)
            swapped = True
            return descriptor
        return original_open(path, flags, mode)

    monkeypatch.setattr("ux_analyzer.storage.run_bundle.os.open", race_open)

    with pytest.raises(BundleStateError, match="containment|path|ancestor"):
        writer.write_artifact("escape.txt", b"must-not-escape")

    assert not any(outside.iterdir())


def test_finalize_rejects_destination_ancestor_swap_during_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    outside = tmp_path / "outside"
    outside.mkdir()
    runs_root = tmp_path / "runs"
    backup_root = tmp_path / "runs-real"
    original_replace = os.replace
    swapped = False

    def race_replace(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        **kwargs: object,
    ) -> None:
        nonlocal swapped
        if not swapped and Path(destination).name == writer.final_path.name:
            runs_root.rename(backup_root)
            try:
                runs_root.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                backup_root.rename(runs_root)
                pytest.skip(f"symlink race fixture unavailable: {error}")
            original_replace(
                source,
                destination,
                src_dir_fd=cast(int | None, kwargs.get("src_dir_fd")),
                dst_dir_fd=cast(int | None, kwargs.get("dst_dir_fd")),
            )
            runs_root.unlink()
            backup_root.rename(runs_root)
            swapped = True
            return
        original_replace(
            source,
            destination,
            src_dir_fd=cast(int | None, kwargs.get("src_dir_fd")),
            dst_dir_fd=cast(int | None, kwargs.get("dst_dir_fd")),
        )

    monkeypatch.setattr("ux_analyzer.storage.run_bundle.os.replace", race_replace)

    try:
        final_path = writer.finalize({"status": "complete"})
    except BundleStateError:
        final_path = None

    assert not (outside / writer.run_id).exists()
    if final_path is not None:
        assert final_path.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-relative replace race")
def test_finalize_windows_rejects_restored_destination_parent_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import ux_analyzer.storage.run_bundle as run_bundle_module

    writer = FilesystemRunBundleWriter.start(tmp_path, bundle_manifest())
    outside = tmp_path / "outside"
    outside.mkdir()
    runs_root = tmp_path / "runs"
    backup_root = tmp_path / "runs-real"
    real_replace = run_bundle_module._replace_windows_handle_relative
    swapped = False

    def swap_before_replace(source: Path, destination: Path, label: str) -> None:
        nonlocal swapped
        if not swapped and destination == writer.final_path:
            runs_root.rename(backup_root)
            try:
                runs_root.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                backup_root.rename(runs_root)
                pytest.skip(f"symlink race fixture unavailable: {error}")
            swapped = True
            try:
                real_replace(source, destination, label)
            finally:
                runs_root.unlink()
                backup_root.rename(runs_root)
            return
        real_replace(source, destination, label)

    monkeypatch.setattr(
        run_bundle_module,
        "_replace_windows_handle_relative",
        swap_before_replace,
    )

    with pytest.raises(BundleStateError, match="reparse|symlink|containment"):
        writer.finalize({"status": "complete"})

    assert not (outside / writer.run_id).exists()
