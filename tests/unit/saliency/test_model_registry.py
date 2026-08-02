from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ux_analyzer.saliency.model_registry import (
    ModelRegistry,
    ModelState,
    RuntimeState,
    RuntimeStatus,
    load_manifest,
    model_home,
)

EXPECTED_ARTIFACTS = {
    "foveacast-v3-1s-fp16.onnx": (
        "4b9fdc2734e36c612a120ab7b0050ae276723160ccc625c6f554e906dd6345d5"
    ),
    "foveacast-v3-3s-fp16.onnx": (
        "842a23f97908d146b8749e05f6b220bdb495eae76c75cc7252550825585ef76e"
    ),
    "foveacast-v3-7s-fp16.onnx": (
        "cf66388dc6fe5db4712c77cf3929d04380ed73ac963677b8c9de2b6380fb51e0"
    ),
    "foveacast-v3-1s-fp16.parity.json": (
        "2572f301f49bf49ec38bcff57fa1282380726e79f1ce4c9a5cd56f9f4894944b"
    ),
    "foveacast-v3-3s-fp16.parity.json": (
        "3ccbd9da648baaf7c0ddca9fce1e56d92ce71f94de2f2751b2dd2af41c276a20"
    ),
    "foveacast-v3-7s-fp16.parity.json": (
        "16cb94c448a6dc510e7d44d31147634750bb3d1801cfa55daf0bcf83e1d97cca"
    ),
}


def _runtime(state: RuntimeState) -> RuntimeStatus:
    return RuntimeStatus(state=state, provider="CPUExecutionProvider")


def test_manifest_pins_release_urls_checksums_and_attribution() -> None:
    manifest = load_manifest("foveacast-v0.2.0")

    assert len(manifest.artifacts) == 6
    assert {artifact.filename: artifact.sha256 for artifact in manifest.artifacts} == (
        EXPECTED_ARTIFACTS
    )
    assert all(
        artifact.url
        == "https://github.com/khawkins98/foveacast-training/releases/download/"
        + f"v0.2.0/{artifact.filename}"
        for artifact in manifest.artifacts
    )
    assert {license_item.license for license_item in manifest.licenses} == {
        "MIT",
        "CC BY 4.0",
    }
    assert "Jiang et al. 2023" in manifest.attribution_text


def test_store_uses_content_addressed_artifact_path(tmp_path: Path) -> None:
    manifest = load_manifest("foveacast-v0.2.0")
    registry = ModelRegistry(model_home=tmp_path, manifest=manifest)
    artifact = manifest.artifacts[0]

    assert registry.artifact_path(artifact) == (
        tmp_path / "foveacast" / "v0.2.0" / "fp16" / artifact.sha256 / artifact.filename
    )


@pytest.mark.parametrize(
    ("runtime_state", "expected"),
    [
        (RuntimeState.MISSING, ModelState.RUNTIME_MISSING),
        (RuntimeState.UNSUPPORTED, ModelState.UNSUPPORTED_PROVIDER),
    ],
)
def test_status_reports_runtime_diagnostics_before_artifact_state(
    tmp_path: Path,
    runtime_state: RuntimeState,
    expected: ModelState,
) -> None:
    registry = ModelRegistry(
        model_home=tmp_path,
        runtime_probe=lambda _provider: _runtime(runtime_state),
    )

    status = registry.status("foveacast-v0.2.0", provider="directml")

    assert status.state is expected
    assert status.diagnostics
    assert expected.value in status.diagnostics


def test_status_distinguishes_missing_and_bad_artifacts(tmp_path: Path) -> None:
    manifest = load_manifest("foveacast-v0.2.0")
    registry = ModelRegistry(
        model_home=tmp_path,
        manifest=manifest,
        runtime_probe=lambda _provider: _runtime(RuntimeState.READY),
    )

    missing = registry.status("foveacast-v0.2.0", provider="cpu")
    assert missing.state is ModelState.ARTIFACT_MISSING

    artifact = manifest.artifacts[0]
    path = registry.artifact_path(artifact)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong")
    mismatch = registry.status("foveacast-v0.2.0", provider="cpu")
    assert mismatch.state is ModelState.CHECKSUM_MISMATCH
    assert any(item.state == "checksum mismatch" for item in mismatch.artifacts)


def test_status_ready_requires_all_model_and_parity_artifacts(
    tmp_path: Path,
) -> None:
    manifest = load_manifest("foveacast-v0.2.0")
    registry = ModelRegistry(
        model_home=tmp_path,
        manifest=manifest,
        runtime_probe=lambda _provider: _runtime(RuntimeState.READY),
    )
    for artifact in manifest.artifacts:
        path = registry.artifact_path(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not the real artifact")

    assert registry.status("foveacast-v0.2.0", provider="cpu").state is (
        ModelState.CHECKSUM_MISMATCH
    )


def test_manifest_override_keeps_registry_api_local_for_tests(tmp_path: Path) -> None:
    manifest = load_manifest("foveacast-v0.2.0")
    local_manifest = replace(
        manifest,
        artifacts=tuple(
            replace(artifact, url=f"http://127.0.0.1/{artifact.filename}")
            for artifact in manifest.artifacts
        ),
    )

    registry = ModelRegistry(model_home=tmp_path, manifest=local_manifest)

    assert registry.manifest.artifacts[0].url.startswith("http://127.0.0.1/")


def test_model_home_honors_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UXA_MODEL_HOME", str(tmp_path / "custom-models"))

    assert model_home() == tmp_path / "custom-models"
