from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

import ux_analyzer.cli as cli
import ux_analyzer.saliency.model_registry as model_registry_module
from ux_analyzer.saliency.model_registry import (
    ChecksumMismatchError,
    ModelRegistry,
    ModelRegistryError,
    ModelState,
    RuntimeState,
    RuntimeStatus,
    load_manifest,
)

runner = CliRunner()


class _FixtureHandler(SimpleHTTPRequestHandler):
    requests: list[str] = []

    def do_GET(self) -> None:
        self.requests.append(self.path)
        super().do_GET()

    def log_message(self, _format: str, *_args: object) -> None:
        return


@pytest.fixture
def local_artifact_registry(tmp_path: Path) -> Iterator[ModelRegistry]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    _FixtureHandler.requests = []
    manifest = load_manifest("foveacast-v0.2.0")
    for index, artifact in enumerate(manifest.artifacts):
        (artifact_root / artifact.filename).write_bytes(
            f"fixture artifact {index}".encode()
        )

    def handler(*args: object, **kwargs: object) -> _FixtureHandler:
        return _FixtureHandler(*args, directory=str(artifact_root), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    local_manifest = manifest.__class__(
        model_id=manifest.model_id,
        provider=manifest.provider,
        version=manifest.version,
        precision=manifest.precision,
        artifacts=tuple(
            artifact.__class__(
                filename=artifact.filename,
                sha256=hashlib.sha256(
                    (artifact_root / artifact.filename).read_bytes()
                ).hexdigest(),
                url=f"{base_url}/{artifact.filename}",
                duration=artifact.duration,
                kind=artifact.kind,
            )
            for artifact in manifest.artifacts
        ),
        licenses=manifest.licenses,
        attribution=manifest.attribution,
    )
    registry = ModelRegistry(
        model_home=tmp_path / "models",
        manifest=local_manifest,
        runtime_probe=lambda _provider: RuntimeStatus(
            state=RuntimeState.READY,
            provider="CPUExecutionProvider",
        ),
    )
    try:
        yield registry
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_models_install_is_atomic_idempotent_and_retains_parity(
    local_artifact_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_registry_for", lambda _model_id: local_artifact_registry)

    installed = runner.invoke(cli.app, ["models", "install", "foveacast-v0.2.0"])
    assert installed.exit_code == 0, installed.stdout
    assert "CC BY 4.0" in installed.stdout
    assert local_artifact_registry.status("foveacast-v0.2.0").state is ModelState.READY

    requests_after_first_install = len(_FixtureHandler.requests)
    repeated = runner.invoke(cli.app, ["models", "install", "foveacast-v0.2.0"])
    assert repeated.exit_code == 0, repeated.stdout
    assert "already installed" in repeated.stdout
    assert len(_FixtureHandler.requests) == requests_after_first_install
    assert any(
        artifact.kind == "parity"
        for artifact in local_artifact_registry.manifest.artifacts
        if local_artifact_registry.artifact_path(artifact).exists()
    )
    assert not list(local_artifact_registry.model_home.rglob("*.partial"))


def test_models_install_rejects_bad_checksum_and_cleans_partial(
    local_artifact_registry: ModelRegistry,
) -> None:
    artifact = local_artifact_registry.manifest.artifacts[0]
    (
        local_artifact_registry.model_home.parent / "artifacts" / artifact.filename
    ).write_bytes(b"tampered fixture artifact")

    with pytest.raises(ChecksumMismatchError):
        local_artifact_registry.install("foveacast-v0.2.0")

    assert not local_artifact_registry.artifact_path(artifact).exists()
    assert not list(local_artifact_registry.model_home.rglob("*.partial"))


def test_models_install_second_checksum_failure_leaves_release_unpublished(
    local_artifact_registry: ModelRegistry,
) -> None:
    artifact = local_artifact_registry.manifest.artifacts[1]
    (
        local_artifact_registry.model_home.parent / "artifacts" / artifact.filename
    ).write_bytes(b"tampered second fixture artifact")

    with pytest.raises(ChecksumMismatchError):
        local_artifact_registry.install("foveacast-v0.2.0")

    assert all(
        not local_artifact_registry.artifact_path(item).exists()
        for item in local_artifact_registry.manifest.artifacts
    )
    release_root = (
        local_artifact_registry.model_home
        / local_artifact_registry.manifest.provider
        / local_artifact_registry.manifest.version
        / local_artifact_registry.manifest.precision
    )
    assert not (release_root / "manifest.json").exists()
    assert not list(local_artifact_registry.model_home.rglob("*.partial"))
    assert not list(local_artifact_registry.model_home.rglob(".staging-*"))


def test_publication_failure_preserves_existing_ready_release(
    local_artifact_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_artifact_registry.install("foveacast-v0.2.0")
    original_artifacts = {
        artifact.filename: local_artifact_registry.artifact_path(artifact).read_bytes()
        for artifact in local_artifact_registry.manifest.artifacts
    }
    release_root = (
        local_artifact_registry.model_home
        / local_artifact_registry.manifest.provider
        / local_artifact_registry.manifest.version
        / local_artifact_registry.manifest.precision
    )
    manifest_path = release_root / "manifest.json"
    original_manifest = manifest_path.read_bytes()

    monkeypatch.setattr(
        local_artifact_registry, "_release_manifest_is_valid", lambda: False
    )
    real_replace = model_registry_module.os.replace

    def fail_before_manifest_backup(source: Path, destination: Path) -> None:
        if source == manifest_path:
            raise OSError("injected publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        model_registry_module.os, "replace", fail_before_manifest_backup
    )

    with pytest.raises(ModelRegistryError, match="release publish failed"):
        local_artifact_registry.install("foveacast-v0.2.0")

    monkeypatch.undo()
    status = local_artifact_registry.status("foveacast-v0.2.0")
    assert status.state is ModelState.READY
    assert manifest_path.read_bytes() == original_manifest
    assert {
        artifact.filename: local_artifact_registry.artifact_path(artifact).read_bytes()
        for artifact in local_artifact_registry.manifest.artifacts
    } == original_artifacts


def test_models_status_and_remove_are_explicit_commands(
    local_artifact_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_registry_for", lambda _model_id: local_artifact_registry)

    before = runner.invoke(
        cli.app,
        ["models", "status", "foveacast-v0.2.0", "--provider", "cpu"],
    )
    assert before.exit_code == 0
    assert "artifact missing" in before.stdout
    assert _FixtureHandler.requests == []

    assert (
        runner.invoke(cli.app, ["models", "install", "foveacast-v0.2.0"]).exit_code == 0
    )
    removed = runner.invoke(
        cli.app,
        ["models", "remove", "foveacast-v0.2.0"],
    )
    assert removed.exit_code == 0, removed.stdout
    assert "removed" in removed.stdout
    assert local_artifact_registry.status("foveacast-v0.2.0").state is (
        ModelState.ARTIFACT_MISSING
    )


def test_models_status_reports_runtime_missing_without_downloading(
    local_artifact_registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_runtime = ModelRegistry(
        model_home=local_artifact_registry.model_home,
        manifest=local_artifact_registry.manifest,
        runtime_probe=lambda _provider: RuntimeStatus(
            state=RuntimeState.MISSING,
            provider="CPUExecutionProvider",
        ),
    )
    monkeypatch.setattr(cli, "_registry_for", lambda _model_id: missing_runtime)

    result = runner.invoke(
        cli.app,
        ["models", "status", "foveacast-v0.2.0", "--provider", "cpu"],
    )

    assert result.exit_code == 0
    assert "runtime missing" in result.stdout
    assert _FixtureHandler.requests == []
