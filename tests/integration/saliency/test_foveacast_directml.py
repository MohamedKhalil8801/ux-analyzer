from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from ux_analyzer.adapters.saliency.foveacast import (
    CPU_EXECUTION_PROVIDER,
    DIRECTML_EXECUTION_PROVIDER,
    FoveacastSaliencyProvider,
)
from ux_analyzer.domain.saliency import (
    AttentionDuration,
    SaliencyPredictionRequest,
    SaliencyRequestMetadata,
)


@dataclass(frozen=True)
class FakeNode:
    name: str
    shape: tuple[int, ...]


class FakeSessionOptions:
    def __init__(self) -> None:
        self.execution_mode: object | None = None
        self.enable_mem_pattern: bool | None = None


class FakeExecutionMode:
    ORT_SEQUENTIAL = "ORT_SEQUENTIAL"


class FakeSession:
    def __init__(
        self,
        provider: str,
        run_order: list[str],
        duration: str,
        session_options: FakeSessionOptions | None,
    ) -> None:
        self.provider = provider
        self.run_order = run_order
        self.duration = duration
        self.session_options = session_options

    def get_inputs(self) -> list[FakeNode]:
        return [FakeNode(name="input", shape=(1, 3, 240, 320))]

    def get_outputs(self) -> list[FakeNode]:
        return [FakeNode(name="output", shape=(1, 1, 240, 320))]

    def get_providers(self) -> list[str]:
        return [self.provider, CPU_EXECUTION_PROVIDER]

    def get_provider_options(self) -> dict[str, dict[str, str]]:
        if self.provider == DIRECTML_EXECUTION_PROVIDER:
            return {self.provider: {"device_id": "3"}}
        return {self.provider: {}}

    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray[Any, Any]],
    ) -> list[np.ndarray[Any, Any]]:
        assert output_names == ["output"]
        assert inputs["input"].shape == (1, 3, 240, 320)
        self.run_order.append(self.duration)
        output = np.zeros((1, 1, 240, 320), dtype=np.float32)
        offset = {"1s": 0.1, "3s": 0.2, "7s": 0.3}[self.duration]
        output[0, 0, 120, 160] = offset
        output[0, 0, 20, 20] = 0.5
        return [output]


class FakeOrt:
    ExecutionMode = FakeExecutionMode

    def __init__(
        self,
        available_providers: tuple[str, ...],
        *,
        fail_directml: bool = False,
    ) -> None:
        self.available_providers = available_providers
        self.fail_directml = fail_directml
        self.sessions: list[FakeSession] = []
        self.run_order: list[str] = []
        self.session_options: list[FakeSessionOptions | None] = []

    def get_available_providers(self) -> list[str]:
        return list(self.available_providers)

    def SessionOptions(self) -> FakeSessionOptions:
        return FakeSessionOptions()

    def InferenceSession(
        self,
        path: str,
        *,
        providers: list[str],
        sess_options: FakeSessionOptions | None = None,
    ) -> FakeSession:
        provider = providers[0]
        if provider == DIRECTML_EXECUTION_PROVIDER and self.fail_directml:
            raise RuntimeError("fake DirectML initialization failure")
        if provider == DIRECTML_EXECUTION_PROVIDER:
            assert sess_options is not None
            assert sess_options.execution_mode == "ORT_SEQUENTIAL"
            assert sess_options.enable_mem_pattern is False
        else:
            assert sess_options is None
        session = FakeSession(
            provider,
            self.run_order,
            Path(path).stem,
            sess_options,
        )
        self.sessions.append(session)
        self.session_options.append(sess_options)
        return session


def _screenshot() -> bytes:
    image = Image.new("RGB", (2, 1), color=(10, 20, 30))
    image.putpixel((1, 0), (110, 120, 130))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _request(preference: str) -> SaliencyPredictionRequest:
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
            execution_provider_preference=preference,
        ),
    )


def _provider(
    fake_ort: FakeOrt,
    *,
    preference: str | None = None,
) -> FoveacastSaliencyProvider:
    kwargs: dict[str, object] = {"ort_module": fake_ort}
    if preference is not None:
        kwargs["execution_provider_preference"] = preference
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
        **kwargs,
    )


def _average_ranks(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.shape, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _spearman_rank_correlation(
    expected: np.ndarray[Any, Any],
    actual: np.ndarray[Any, Any],
) -> float:
    expected_ranks = _average_ranks(np.ravel(expected))
    actual_ranks = _average_ranks(np.ravel(actual))
    if expected_ranks.shape != actual_ranks.shape:
        raise ValueError("Spearman inputs must have matching shapes")
    expected_centered = expected_ranks - np.mean(expected_ranks)
    actual_centered = actual_ranks - np.mean(actual_ranks)
    denominator = float(
        np.sqrt(np.sum(expected_centered**2) * np.sum(actual_centered**2))
    )
    if denominator == 0.0:
        raise ValueError("Spearman correlation is undefined for constant input")
    return float(np.sum(expected_centered * actual_centered) / denominator)


def test_cpu_preference_records_cpu_provider_without_directml_options() -> None:
    fake_ort = FakeOrt((CPU_EXECUTION_PROVIDER,))
    provider = _provider(fake_ort)

    provider.predict(_request("cpu"))

    assert len(fake_ort.sessions) == 3
    assert [session.provider for session in fake_ort.sessions] == [
        CPU_EXECUTION_PROVIDER,
        CPU_EXECUTION_PROVIDER,
        CPU_EXECUTION_PROVIDER,
    ]
    assert provider.last_metadata is not None
    assert provider.last_metadata.requested_execution_provider == "cpu"
    assert provider.last_metadata.actual_execution_provider == CPU_EXECUTION_PROVIDER
    assert provider.last_metadata.fallback_reason is None
    assert provider.last_metadata.adapter_device_id is None
    assert provider.last_metadata.session_options == {}


def test_directml_preference_uses_three_sequential_configured_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    fake_ort = FakeOrt((DIRECTML_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER))
    provider = _provider(fake_ort, preference="directml")

    provider.predict(_request("directml"))

    assert len(fake_ort.sessions) == 3
    assert [session.provider for session in fake_ort.sessions] == [
        DIRECTML_EXECUTION_PROVIDER,
        DIRECTML_EXECUTION_PROVIDER,
        DIRECTML_EXECUTION_PROVIDER,
    ]
    assert fake_ort.run_order == ["1s", "3s", "7s"]
    assert provider.last_metadata is not None
    assert provider.last_metadata.requested_execution_provider == "directml"
    assert (
        provider.last_metadata.actual_execution_provider == DIRECTML_EXECUTION_PROVIDER
    )
    assert provider.last_metadata.fallback_reason is None
    assert provider.last_metadata.adapter_device_id == "3"
    assert provider.last_metadata.session_options == {
        "execution_mode": "ORT_SEQUENTIAL",
        "enable_mem_pattern": False,
    }


def test_auto_prefers_directml_when_available_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    fake_ort = FakeOrt((DIRECTML_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER))
    provider = _provider(fake_ort, preference="auto")

    provider.predict(_request("auto"))

    assert {session.provider for session in fake_ort.sessions} == {
        DIRECTML_EXECUTION_PROVIDER
    }
    assert provider.last_metadata is not None
    assert provider.last_metadata.requested_execution_provider == "auto"
    assert (
        provider.last_metadata.actual_execution_provider == DIRECTML_EXECUTION_PROVIDER
    )
    assert provider.last_metadata.fallback_reason is None


def test_auto_falls_back_to_cpu_with_reason_when_directml_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    fake_ort = FakeOrt((CPU_EXECUTION_PROVIDER,))
    provider = _provider(fake_ort, preference="auto")

    provider.predict(_request("auto"))

    assert {session.provider for session in fake_ort.sessions} == {
        CPU_EXECUTION_PROVIDER
    }
    assert provider.last_metadata is not None
    assert provider.last_metadata.requested_execution_provider == "auto"
    assert provider.last_metadata.actual_execution_provider == CPU_EXECUTION_PROVIDER
    assert provider.last_metadata.fallback_reason == (
        "DirectML unavailable: DmlExecutionProvider is not installed"
    )


def test_auto_falls_back_to_cpu_when_directml_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    fake_ort = FakeOrt(
        (DIRECTML_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER),
        fail_directml=True,
    )
    provider = _provider(fake_ort, preference="auto")

    provider.predict(_request("auto"))

    assert [session.provider for session in fake_ort.sessions] == [
        CPU_EXECUTION_PROVIDER,
        CPU_EXECUTION_PROVIDER,
        CPU_EXECUTION_PROVIDER,
    ]
    assert provider.last_metadata is not None
    assert provider.last_metadata.fallback_reason == (
        "DirectML initialization failed: RuntimeError"
    )


def test_explicit_directml_failure_is_not_silently_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    fake_ort = FakeOrt(
        (DIRECTML_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER),
        fail_directml=True,
    )
    provider = _provider(fake_ort)

    with pytest.raises(RuntimeError, match="DirectML initialization failed"):
        provider.predict(_request("directml"))


def test_explicit_directml_requires_windows_and_available_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    fake_ort = FakeOrt((DIRECTML_EXECUTION_PROVIDER, CPU_EXECUTION_PROVIDER))
    provider = _provider(fake_ort)

    with pytest.raises(RuntimeError, match="requires Windows"):
        provider.predict(_request("directml"))


def test_spearman_rejects_high_pearson_with_different_ties_and_ranking() -> None:
    group_count = 250
    expected = np.concatenate(
        (
            np.repeat(np.linspace(0.0, 0.001, group_count), 2),
            np.repeat(np.linspace(1.0, 1.001, group_count), 2),
        )
    )
    actual = np.concatenate(
        (
            expected[: group_count * 2][::-1],
            expected[group_count * 2 :][::-1],
        )
    )

    assert float(np.corrcoef(expected, actual)[0, 1]) >= 0.999
    assert _spearman_rank_correlation(expected, actual) < 0.999


def _hardware_model_paths() -> dict[AttentionDuration, Path] | None:
    names = {
        AttentionDuration.ONE_SECOND: "UXA_FOVEACAST_MODEL_1S",
        AttentionDuration.THREE_SECONDS: "UXA_FOVEACAST_MODEL_3S",
        AttentionDuration.SEVEN_SECONDS: "UXA_FOVEACAST_MODEL_7S",
    }
    paths = {
        duration: Path(os.environ[name])
        for duration, name in names.items()
        if name in os.environ
    }
    if len(paths) != len(names) or not all(path.is_file() for path in paths.values()):
        return None
    return paths


@pytest.mark.directml
def test_directml_hardware_parity_and_stability() -> None:
    if sys.platform != "win32":
        pytest.skip("DirectML hardware test requires Windows")
    paths = _hardware_model_paths()
    if paths is None:
        pytest.skip("set UXA_FOVEACAST_MODEL_1S/3S/7S to local model files")
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip("onnxruntime-directml is not installed")
    if DIRECTML_EXECUTION_PROVIDER not in ort.get_available_providers():
        pytest.skip("DirectML execution provider is unavailable")

    cpu = FoveacastSaliencyProvider(
        paths,
        execution_provider_preference="cpu",
        ort_module=ort,
    )
    directml = FoveacastSaliencyProvider(
        paths,
        execution_provider_preference="directml",
        ort_module=ort,
    )
    cpu_predictions = cpu.predict(_request("cpu"))
    directml_predictions = directml.predict(_request("directml"))
    repeated_predictions = directml.predict(_request("directml"))

    for cpu_prediction, directml_prediction, repeated_prediction in zip(
        cpu_predictions.predictions,
        directml_predictions.predictions,
        repeated_predictions.predictions,
        strict=True,
    ):
        cpu_map = np.asarray(cpu_prediction.plane.float_values(), dtype=np.float32)
        directml_map = np.asarray(
            directml_prediction.plane.float_values(), dtype=np.float32
        )
        repeated_map = np.asarray(
            repeated_prediction.plane.float_values(), dtype=np.float32
        )
        assert float(np.max(np.abs(cpu_map - directml_map))) <= 0.002
        assert _spearman_rank_correlation(cpu_map, directml_map) >= 0.999
        np.testing.assert_array_equal(directml_map, repeated_map)
