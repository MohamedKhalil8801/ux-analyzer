from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ux_analyzer.adapters.saliency.foveacast import FoveacastSaliencyProvider
from ux_analyzer.domain.saliency import (
    AttentionDuration,
    SaliencyPredictionRequest,
    SaliencyRequestMetadata,
)
from ux_analyzer.saliency.model_registry import (
    ChecksumMismatchError,
    ModelArtifact,
    ModelManifest,
    ModelRegistry,
    ModelState,
    RuntimeState,
    RuntimeStatus,
    load_manifest,
)


@dataclass(frozen=True)
class FakeNode:
    name: str
    shape: tuple[int, ...]


class FakeSession:
    def __init__(self, duration: str, output_kind: str = "valid") -> None:
        self.duration = duration
        self.output_kind = output_kind
        self.calls: list[dict[str, np.ndarray]] = []
        self.run_started: list[str] = []

    def get_inputs(self) -> list[FakeNode]:
        return [FakeNode(name="input", shape=(1, 3, 240, 320))]

    def get_outputs(self) -> list[FakeNode]:
        return [FakeNode(name="output", shape=(1, 1, 240, 320))]

    def run(
        self, output_names: list[str], inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names == ["output"]
        self.calls.append(inputs)
        self.run_started.append(self.duration)
        if self.output_kind == "nan":
            return [np.full((1, 1, 240, 320), np.nan, dtype=np.float32)]
        if self.output_kind == "inf":
            return [np.full((1, 1, 240, 320), np.inf, dtype=np.float32)]
        if self.output_kind == "constant":
            return [np.full((1, 1, 240, 320), 0.5, dtype=np.float32)]
        if self.output_kind == "negative":
            return [np.full((1, 1, 240, 320), -0.1, dtype=np.float32)]
        if self.output_kind == "above-one":
            return [np.full((1, 1, 240, 320), 1.1, dtype=np.float32)]
        if self.output_kind == "wrong-shape":
            return [np.zeros((1, 1, 240, 319), dtype=np.float32)]

        output = np.zeros((1, 1, 240, 320), dtype=np.float32)
        duration_offset = {"1s": 0.25, "3s": 0.5, "7s": 0.75}[self.duration]
        output[0, 0, 120, 160] = duration_offset
        return [output]


class FakeOrt:
    def __init__(self, output_kind: str = "valid") -> None:
        self.output_kind = output_kind
        self.sessions: list[FakeSession] = []
        self.run_order: list[str] = []

    def InferenceSession(self, path: str, *, providers: list[str]) -> FakeSession:
        assert providers == ["CPUExecutionProvider"]
        stem = Path(path).stem
        duration = next(label for label in ("1s", "3s", "7s") if label in stem)
        session = FakeSession(duration, self.output_kind)
        session.run_started = self.run_order
        self.sessions.append(session)
        return session


def _screenshot() -> bytes:
    image = Image.new("RGB", (2, 1), color=(10, 20, 30))
    image.putpixel((1, 0), (110, 120, 130))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _request() -> SaliencyPredictionRequest:
    screenshot = _screenshot()
    return SaliencyPredictionRequest(
        screenshot=screenshot,
        metadata=SaliencyRequestMetadata(
            viewport_id="viewport-1",
            screenshot_sha256=hashlib.sha256(screenshot).hexdigest(),
            screenshot_width=2,
            screenshot_height=1,
            device_pixel_ratio=1.0,
            zoom=1.0,
            requested_durations=("7s", "1s", "3s"),
            model_set=("foveacast-v0.2.0",),
            precision="fp16",
            execution_provider_preference="cpu",
        ),
    )


def _registry(tmp_path: Path) -> ModelRegistry:
    manifest = load_manifest("foveacast-v0.2.0")
    artifacts: list[ModelArtifact] = []
    payloads: dict[str, bytes] = {}
    for artifact in manifest.artifacts:
        payload = f"fixture {artifact.filename}".encode()
        payloads[artifact.filename] = payload
        artifacts.append(
            ModelArtifact(
                filename=artifact.filename,
                sha256=hashlib.sha256(payload).hexdigest(),
                url=f"http://127.0.0.1/{artifact.filename}",
                duration=artifact.duration,
                kind=artifact.kind,
            )
        )
    local_manifest = ModelManifest(
        model_id=manifest.model_id,
        provider=manifest.provider,
        version=manifest.version,
        precision=manifest.precision,
        artifacts=tuple(artifacts),
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
    for artifact in local_manifest.artifacts:
        path = registry.artifact_path(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[artifact.filename])
    return registry


def _provider(fake_ort: FakeOrt, registry: ModelRegistry) -> FoveacastSaliencyProvider:
    return FoveacastSaliencyProvider(
        registry,
        ort_module=fake_ort,
    )


def test_cpu_provider_reuses_sessions_and_runs_durations_sequentially(
    tmp_path: Path,
) -> None:
    fake_ort = FakeOrt()
    provider = _provider(fake_ort, _registry(tmp_path))

    first = provider.predict(_request())
    second = provider.predict(_request())

    assert len(fake_ort.sessions) == 3
    assert fake_ort.run_order == [
        "1s",
        "3s",
        "7s",
        "1s",
        "3s",
        "7s",
    ]
    assert tuple(prediction.duration for prediction in first.predictions) == (
        AttentionDuration.ONE_SECOND,
        AttentionDuration.THREE_SECONDS,
        AttentionDuration.SEVEN_SECONDS,
    )
    assert second.predictions[0].plane.values == first.predictions[0].plane.values
    assert all(
        prediction.plane.width == 320 and prediction.plane.height == 240
        for prediction in first.predictions
    )
    assert all(
        0.0 <= value <= 1.0
        for prediction in first.predictions
        for value in prediction.plane.float_values()
    )

    input_tensor = fake_ort.sessions[0].calls[0]["input"]
    assert input_tensor.shape == (1, 3, 240, 320)
    assert input_tensor.dtype == np.dtype(np.float32)
    assert input_tensor[0, :, 40, 0].tolist() == [10.0, 20.0, 30.0]
    assert input_tensor[0, :, 0, 0].tolist() == [126.0, 126.0, 126.0]

    assert provider.last_metadata is not None
    assert provider.last_metadata.geometry.source_dimensions == (2, 1)
    assert provider.last_metadata.model_id == "foveacast-v0.2.0"
    assert provider.last_metadata.preprocessing_version == "foveacast-preprocess-v1"
    assert provider.last_metadata.execution_provider == "CPUExecutionProvider"
    assert provider.last_timing is not None
    assert provider.last_timing.cold_load_ms >= 0.0
    assert tuple(
        duration for duration, _ in provider.last_timing.warm_inference_ms
    ) == (
        AttentionDuration.ONE_SECOND,
        AttentionDuration.THREE_SECONDS,
        AttentionDuration.SEVEN_SECONDS,
    )
    assert all(
        duration_ms >= 0.0 for _, duration_ms in provider.last_timing.warm_inference_ms
    )
    assert provider.last_timing.total_inference_ms >= sum(
        duration_ms for _, duration_ms in provider.last_timing.warm_inference_ms
    )
    assert [
        prediction.metadata.inference_duration_ms for prediction in second.predictions
    ] == [
        pytest.approx(duration_ms)
        for _, duration_ms in provider.last_timing.warm_inference_ms
    ]


@pytest.mark.parametrize(
    "output_kind",
    ["nan", "inf", "constant", "negative", "above-one", "wrong-shape"],
)
def test_cpu_provider_rejects_malformed_model_outputs(
    tmp_path: Path, output_kind: str
) -> None:
    with pytest.raises(ValueError, match="(finite|constant|range|between|shape)"):
        _provider(FakeOrt(output_kind), _registry(tmp_path)).predict(_request())


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "input_name", "output_name", "message"),
    [
        ((1, 3, 240, 319), (1, 1, 240, 320), "input", "output", "input shape"),
        ((1, 3, 240, 320), (1, 1, 240, 319), "input", "output", "output shape"),
        ((1, 3, 240, 320), (1, 1, 240, 320), "", "output", "input name"),
        ((1, 3, 240, 320), (1, 1, 240, 320), "model_input", "output", "input name"),
        ((1, 3, 240, 320), (1, 1, 240, 320), "input", "saliency", "output name"),
    ],
)
def test_cpu_provider_validates_session_names_and_fixed_shapes(
    tmp_path: Path,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    input_name: str,
    output_name: str,
    message: str,
) -> None:
    class InvalidSession(FakeSession):
        def get_inputs(self) -> list[FakeNode]:
            return [FakeNode(name=input_name, shape=input_shape)]

        def get_outputs(self) -> list[FakeNode]:
            return [FakeNode(name=output_name, shape=output_shape)]

    class InvalidOrt(FakeOrt):
        def InferenceSession(
            self, path: str, *, providers: list[str]
        ) -> InvalidSession:
            session = InvalidSession(Path(path).stem)
            session.run_started = self.run_order
            self.sessions.append(session)
            return session

    with pytest.raises(ValueError, match=message):
        _provider(InvalidOrt(), _registry(tmp_path))


def test_cpu_provider_accepts_symbolic_spatial_session_dimensions(
    tmp_path: Path,
) -> None:
    class SymbolicSession(FakeSession):
        def get_inputs(self) -> list[FakeNode]:
            return [FakeNode(name="input", shape=(1, 3, "height", "width"))]  # type: ignore[arg-type]

        def get_outputs(self) -> list[FakeNode]:
            return [FakeNode(name="output", shape=(1, 1, "height", "width"))]  # type: ignore[arg-type]

    class SymbolicOrt(FakeOrt):
        def InferenceSession(
            self, path: str, *, providers: list[str]
        ) -> SymbolicSession:
            duration = next(label for label in ("1s", "3s", "7s") if label in path)
            session = SymbolicSession(duration)
            session.run_started = self.run_order
            self.sessions.append(session)
            return session

    provider = _provider(SymbolicOrt(), _registry(tmp_path))

    assert len(provider.predict(_request()).predictions) == 3


def test_cpu_provider_rejects_tampered_model_before_session_creation(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    model_artifact = next(
        artifact for artifact in registry.manifest.artifacts if artifact.kind == "model"
    )
    registry.artifact_path(model_artifact).write_bytes(b"tampered model")
    fake_ort = FakeOrt()

    with pytest.raises(ChecksumMismatchError, match="checksum mismatch"):
        _provider(fake_ort, registry)

    assert fake_ort.sessions == []


@pytest.mark.live
def test_pinned_foveacast_cpu_artifacts_are_stable_when_installed() -> None:
    registry = ModelRegistry()
    status = registry.status("foveacast-v0.2.0", provider="cpu")
    if status.state is not ModelState.READY:
        pytest.skip(
            "pinned Foveacast CPU artifacts/runtime unavailable: "
            + "; ".join(status.diagnostics)
        )

    provider = FoveacastSaliencyProvider(registry)
    first = provider.predict(_request())
    second = provider.predict(_request())

    assert tuple(prediction.duration for prediction in first.predictions) == (
        AttentionDuration.ONE_SECOND,
        AttentionDuration.THREE_SECONDS,
        AttentionDuration.SEVEN_SECONDS,
    )
    assert len(first.predictions) == 3
    for first_prediction, second_prediction in zip(
        first.predictions, second.predictions, strict=True
    ):
        assert (first_prediction.plane.width, first_prediction.plane.height) == (
            320,
            240,
        )
        values = np.asarray(first_prediction.plane.float_values(), dtype=np.float32)
        assert np.isfinite(values).all()
        assert np.all((values >= 0.0) & (values <= 1.0))
        assert first_prediction.plane.values == second_prediction.plane.values
        assert first_prediction.metadata.model_checksum == next(
            item.artifact.sha256
            for item in registry.verified_model_artifacts()
            if item.artifact.duration
            == AttentionDuration(first_prediction.duration).value
        )
        assert first_prediction.metadata.geometry.source_dimensions == (2, 1)
        assert first_prediction.metadata.geometry.native_dimensions == (320, 240)
        assert first_prediction.metadata.geometry.content_dimensions == (320, 160)
        assert first_prediction.metadata.geometry.pad_top == 40
