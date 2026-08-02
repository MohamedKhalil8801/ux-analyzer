from __future__ import annotations

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

        output = np.zeros((1, 1, 240, 320), dtype=np.float32)
        duration_offset = {"1s": 1.0, "3s": 2.0, "7s": 3.0}[self.duration]
        output[0, 0, 120, 160] = duration_offset
        return [output]


class FakeOrt:
    def __init__(self, output_kind: str = "valid") -> None:
        self.output_kind = output_kind
        self.sessions: list[FakeSession] = []
        self.run_order: list[str] = []

    def InferenceSession(self, path: str, *, providers: list[str]) -> FakeSession:
        assert providers == ["CPUExecutionProvider"]
        duration = Path(path).stem
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
    return SaliencyPredictionRequest(
        screenshot=_screenshot(),
        metadata=SaliencyRequestMetadata(
            viewport_id="viewport-1",
            screenshot_sha256="a" * 64,
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


def _provider(fake_ort: FakeOrt) -> FoveacastSaliencyProvider:
    return FoveacastSaliencyProvider(
        model_paths={
            AttentionDuration.ONE_SECOND: Path("1s.onnx"),
            AttentionDuration.THREE_SECONDS: Path("3s.onnx"),
            AttentionDuration.SEVEN_SECONDS: Path("7s.onnx"),
        },
        model_checksums={
            duration: f"checksum-{duration.value}"
            for duration in AttentionDuration
            if duration is not AttentionDuration.GENERAL
        },
        ort_module=fake_ort,
    )


def test_cpu_provider_reuses_sessions_and_runs_durations_sequentially() -> None:
    fake_ort = FakeOrt()
    provider = _provider(fake_ort)

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


@pytest.mark.parametrize("output_kind", ["nan", "inf", "constant"])
def test_cpu_provider_rejects_malformed_model_outputs(output_kind: str) -> None:
    with pytest.raises(ValueError, match="(finite|constant)"):
        _provider(FakeOrt(output_kind)).predict(_request())


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "input_name", "output_name", "message"),
    [
        ((1, 3, 240, 319), (1, 1, 240, 320), "input", "output", "input shape"),
        ((1, 3, 240, 320), (1, 1, 240, 319), "input", "output", "output shape"),
        ((1, 3, 240, 320), (1, 1, 240, 320), "", "output", "input name"),
    ],
)
def test_cpu_provider_validates_session_names_and_fixed_shapes(
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
        _provider(InvalidOrt())
