"""CPU ONNX Runtime adapter for pinned Foveacast saliency models."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

import numpy as np
from PIL import Image, UnidentifiedImageError

from ux_analyzer.domain.saliency import (
    AttentionDuration,
    SaliencyPlane,
    SaliencyPrediction,
    SaliencyPredictionMetadata,
    SaliencyPredictionRequest,
    SaliencyPredictionSet,
)

MODEL_WIDTH = 320
MODEL_HEIGHT = 240
PADDING_VALUE = 126.0
PREPROCESSING_VERSION = "foveacast-preprocess-v1"
PROVIDER_ID = "foveacast"
PROVIDER_VERSION = "foveacast-adapter-v1"
MODEL_ID = "foveacast-v0.2.0"
MODEL_VERSION = "v0.2.0"
CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"
_INPUT_SHAPE = (1, 3, MODEL_HEIGHT, MODEL_WIDTH)
_OUTPUT_SHAPES = {
    (MODEL_HEIGHT, MODEL_WIDTH),
    (1, MODEL_HEIGHT, MODEL_WIDTH),
    (1, 1, MODEL_HEIGHT, MODEL_WIDTH),
}
_DURATIONS = (
    AttentionDuration.ONE_SECOND,
    AttentionDuration.THREE_SECONDS,
    AttentionDuration.SEVEN_SECONDS,
)


class _OrtValueInfo(Protocol):
    name: object
    shape: Sequence[object]


class _OrtSession(Protocol):
    def get_inputs(self) -> Sequence[_OrtValueInfo]: ...

    def get_outputs(self) -> Sequence[_OrtValueInfo]: ...

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, np.ndarray[Any, Any]],
    ) -> Sequence[object]: ...


class _OrtModule(Protocol):
    def InferenceSession(
        self, path: str, *, providers: Sequence[str]
    ) -> _OrtSession: ...


@dataclass(frozen=True, slots=True)
class FoveacastGeometry:
    """Pixel-center transform between source pixels and native model pixels."""

    source_width: int
    source_height: int
    input_width: int
    input_height: int
    content_width: int
    content_height: int
    pad_left: int
    pad_top: int
    scale: float
    scale_x: float
    scale_y: float

    @classmethod
    def from_source_dimensions(
        cls,
        source_width: int,
        source_height: int,
        *,
        input_width: int = MODEL_WIDTH,
        input_height: int = MODEL_HEIGHT,
    ) -> FoveacastGeometry:
        if (
            isinstance(source_width, bool)
            or isinstance(source_height, bool)
            or source_width <= 0
            or source_height <= 0
        ):
            raise ValueError("source dimensions must be greater than zero")
        if (
            isinstance(input_width, bool)
            or isinstance(input_height, bool)
            or input_width <= 0
            or input_height <= 0
        ):
            raise ValueError("input dimensions must be greater than zero")

        scale = min(input_width / source_width, input_height / source_height)
        content_width = max(1, int(round(source_width * scale)))
        content_height = max(1, int(round(source_height * scale)))
        pad_left = (input_width - content_width) // 2
        pad_top = (input_height - content_height) // 2
        return cls(
            source_width=source_width,
            source_height=source_height,
            input_width=input_width,
            input_height=input_height,
            content_width=content_width,
            content_height=content_height,
            pad_left=pad_left,
            pad_top=pad_top,
            scale=scale,
            scale_x=content_width / source_width,
            scale_y=content_height / source_height,
        )

    @property
    def source_dimensions(self) -> tuple[int, int]:
        """Return source width and height."""

        return self.source_width, self.source_height

    @property
    def input_dimensions(self) -> tuple[int, int]:
        """Return native model width and height."""

        return self.input_width, self.input_height

    @property
    def resized_dimensions(self) -> tuple[int, int]:
        """Return fitted content width and height before padding."""

        return self.content_width, self.content_height

    @property
    def padding(self) -> tuple[int, int]:
        """Return left and top pad offsets."""

        return self.pad_left, self.pad_top

    @property
    def pad_right(self) -> int:
        """Return right-side padding width."""

        return self.input_width - self.content_right

    @property
    def pad_bottom(self) -> int:
        """Return bottom padding height."""

        return self.input_height - self.content_bottom

    @property
    def content_right(self) -> int:
        """Return exclusive right edge of fitted content."""

        return self.pad_left + self.content_width

    @property
    def content_bottom(self) -> int:
        """Return exclusive bottom edge of fitted content."""

        return self.pad_top + self.content_height

    def source_to_model(self, x: float, y: float) -> tuple[float, float]:
        """Map source pixel-center coordinates into native model coordinates."""

        return (
            self.pad_left + (x + 0.5) * self.scale_x - 0.5,
            self.pad_top + (y + 0.5) * self.scale_y - 0.5,
        )

    def model_to_source(self, x: float, y: float) -> tuple[float, float]:
        """Map native model pixel-center coordinates back to source coordinates."""

        return (
            (x - self.pad_left + 0.5) / self.scale_x - 0.5,
            (y - self.pad_top + 0.5) / self.scale_y - 0.5,
        )

    def crop_content(self, native_map: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Crop native map padding before source-coordinate projection."""

        if tuple(native_map.shape) != (self.input_height, self.input_width):
            raise ValueError(
                "native saliency map shape must match preprocessing input dimensions"
            )
        return native_map[
            self.pad_top : self.content_bottom,
            self.pad_left : self.content_right,
        ]


@dataclass(frozen=True, slots=True)
class PreprocessedScreenshot:
    """Model tensor and reversible source-to-native geometry."""

    tensor: np.ndarray[Any, Any]
    geometry: FoveacastGeometry


@dataclass(frozen=True, slots=True)
class FoveacastTiming:
    """Cold model-load time and warm sequential inference times."""

    cold_load_ms: float
    warm_inference_ms: tuple[tuple[AttentionDuration, float], ...]
    total_inference_ms: float


@dataclass(frozen=True, slots=True)
class FoveacastInferenceMetadata:
    """Adapter metadata not yet represented by the domain prediction contract."""

    viewport_id: str
    screenshot_dimensions: tuple[int, int]
    geometry: FoveacastGeometry
    timing: FoveacastTiming
    model_id: str
    model_version: str
    precision: str
    preprocessing_version: str
    execution_provider: str
    input_name: str
    output_name: str


def preprocess_image(image: Image.Image) -> PreprocessedScreenshot:
    """Convert RGB pixels to one padded NCHW float32 Foveacast input."""

    rgb = _load_rgb_image(image)
    source = np.asarray(rgb, dtype=np.float32)
    source_height, source_width, channels = source.shape
    if channels != 3:
        raise ValueError("screenshot must decode to RGB pixels")

    geometry = FoveacastGeometry.from_source_dimensions(source_width, source_height)
    resized = _resize_pixel_center_bilinear(
        source,
        width=geometry.content_width,
        height=geometry.content_height,
    )
    canvas = np.full((MODEL_HEIGHT, MODEL_WIDTH, 3), PADDING_VALUE, dtype=np.float32)
    canvas[
        geometry.pad_top : geometry.content_bottom,
        geometry.pad_left : geometry.content_right,
    ] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[np.newaxis, ...])
    return PreprocessedScreenshot(tensor=tensor, geometry=geometry)


def preprocess_screenshot(
    screenshot: bytes | Image.Image,
) -> PreprocessedScreenshot:
    """Decode screenshot bytes and prepare one Foveacast input tensor."""

    if isinstance(screenshot, Image.Image):
        return preprocess_image(screenshot)
    try:
        with Image.open(BytesIO(screenshot)) as image:
            image.load()
            return preprocess_image(image)
    except (TypeError, UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("screenshot must be a valid image") from error


def _load_rgb_image(image: object) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise ValueError("screenshot must be a valid image")
    try:
        rgb = image.convert("RGB")
        rgb.load()
    except (OSError, ValueError) as error:
        raise ValueError("screenshot must be a valid image") from error
    if rgb.width <= 0 or rgb.height <= 0:
        raise ValueError("screenshot must have positive dimensions")
    return rgb


def _resize_pixel_center_bilinear(
    source: np.ndarray[Any, Any], *, width: int, height: int
) -> np.ndarray[Any, Any]:
    source_height, source_width, _ = source.shape
    source_x = (np.arange(width, dtype=np.float32) + 0.5) * (source_width / width) - 0.5
    source_y = (np.arange(height, dtype=np.float32) + 0.5) * (
        source_height / height
    ) - 0.5
    source_x = np.clip(source_x, 0.0, source_width - 1.0)
    source_y = np.clip(source_y, 0.0, source_height - 1.0)
    x0 = np.floor(source_x).astype(np.intp)
    y0 = np.floor(source_y).astype(np.intp)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x_weight = source_x - x0
    y_weight = source_y - y0

    resized = np.empty((height, width, 3), dtype=np.float32)
    for channel in range(3):
        top_left = source[y0[:, np.newaxis], x0[np.newaxis, :], channel]
        top_right = source[y0[:, np.newaxis], x1[np.newaxis, :], channel]
        bottom_left = source[y1[:, np.newaxis], x0[np.newaxis, :], channel]
        bottom_right = source[y1[:, np.newaxis], x1[np.newaxis, :], channel]
        top = top_left + (top_right - top_left) * x_weight[np.newaxis, :]
        bottom = bottom_left + (bottom_right - bottom_left) * x_weight[np.newaxis, :]
        resized[:, :, channel] = top + (bottom - top) * y_weight[:, np.newaxis]
    return resized


def _duration_paths(
    model_paths: Mapping[AttentionDuration | str, Path | str] | Sequence[Path | str],
) -> dict[AttentionDuration, Path]:
    if isinstance(model_paths, Mapping):
        entries = tuple(model_paths.items())
    else:
        paths = tuple(model_paths)
        if len(paths) != len(_DURATIONS):
            raise ValueError("model_paths must contain one path for each duration")
        entries = tuple(zip(_DURATIONS, paths, strict=True))
    normalized: dict[AttentionDuration, Path] = {}
    for duration, path in entries:
        try:
            normalized_duration = AttentionDuration(duration)
        except ValueError as error:
            raise ValueError(f"unsupported Foveacast duration: {duration!r}") from error
        if normalized_duration is AttentionDuration.GENERAL:
            raise ValueError("Foveacast does not support general duration")
        if normalized_duration in normalized:
            raise ValueError(f"duplicate model path for {normalized_duration.value}")
        normalized[normalized_duration] = Path(path)
    if set(normalized) != set(_DURATIONS):
        raise ValueError("model_paths must contain 1s, 3s, and 7s models")
    return normalized


def _duration_checksums(
    checksums: Mapping[AttentionDuration | str, str] | None,
) -> dict[AttentionDuration, str]:
    if checksums is None:
        return {}
    normalized: dict[AttentionDuration, str] = {}
    for duration, checksum in checksums.items():
        normalized_duration = AttentionDuration(duration)
        if normalized_duration is AttentionDuration.GENERAL:
            raise ValueError("Foveacast does not support general duration")
        if normalized_duration in normalized:
            raise ValueError(
                f"duplicate model checksum for {normalized_duration.value}"
            )
        if not checksum.strip():
            raise ValueError("model checksum must not be empty")
        normalized[normalized_duration] = checksum
    return normalized


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _node_name(node: _OrtValueInfo, kind: str) -> str:
    name: object = node.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{kind} name must not be empty")
    return name


def _node_shape(node: _OrtValueInfo, kind: str) -> tuple[int, ...]:
    try:
        shape = tuple(node.shape)
    except TypeError as error:
        raise ValueError(f"{kind} shape must be fixed") from error
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in shape
    ):
        raise ValueError(f"{kind} shape must be fixed positive integers")
    return cast(tuple[int, ...], shape)


def _validate_session(session: _OrtSession) -> tuple[str, str, tuple[int, ...]]:
    inputs = tuple(session.get_inputs())
    outputs = tuple(session.get_outputs())
    if len(inputs) != 1:
        raise ValueError("Foveacast model must expose exactly one input")
    if len(outputs) != 1:
        raise ValueError("Foveacast model must expose exactly one output")
    input_name = _node_name(inputs[0], "input")
    output_name = _node_name(outputs[0], "output")
    input_shape = _node_shape(inputs[0], "input")
    output_shape = _node_shape(outputs[0], "output")
    if input_shape != _INPUT_SHAPE:
        raise ValueError(f"input shape must be {_INPUT_SHAPE}, got {input_shape}")
    if output_shape not in _OUTPUT_SHAPES:
        expected = ", ".join(str(shape) for shape in sorted(_OUTPUT_SHAPES))
        raise ValueError(f"output shape must be one of {expected}, got {output_shape}")
    return input_name, output_name, output_shape


def _normalize_output(
    output: object, output_shape: tuple[int, ...]
) -> np.ndarray[Any, Any]:
    try:
        array = np.asarray(output, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("saliency output must be numeric") from error
    if tuple(array.shape) != output_shape:
        raise ValueError(
            f"saliency output shape must be {output_shape}, got {array.shape}"
        )
    if not bool(np.isfinite(array).all()):
        raise ValueError("saliency output must contain finite values")
    if len(output_shape) == 4:
        saliency_map = array[0, 0]
    elif len(output_shape) == 3:
        saliency_map = array[0]
    else:
        saliency_map = array
    minimum = float(np.min(saliency_map))
    maximum = float(np.max(saliency_map))
    if minimum == maximum:
        raise ValueError("saliency output must not be constant")
    if 0.0 <= minimum and maximum <= 1.0:
        normalized = saliency_map
    else:
        normalized = (saliency_map - minimum) / (maximum - minimum)
    return np.ascontiguousarray(np.clip(normalized, 0.0, 1.0), dtype=np.float32)


class FoveacastSaliencyProvider:
    """Run three fixed-shape Foveacast CPU sessions sequentially per screenshot."""

    def __init__(
        self,
        model_paths: Mapping[AttentionDuration | str, Path | str]
        | Sequence[Path | str],
        *,
        model_id: str = MODEL_ID,
        model_version: str = MODEL_VERSION,
        provider_version: str = PROVIDER_VERSION,
        precision: str = "fp16",
        model_checksums: Mapping[AttentionDuration | str, str] | None = None,
        ort_module: _OrtModule | object | None = None,
    ) -> None:
        self.model_paths = _duration_paths(model_paths)
        self.model_checksums = _duration_checksums(model_checksums)
        self.model_id = model_id
        self.model_version = model_version
        self.provider_version = provider_version
        self.precision = precision
        self.execution_provider = CPU_EXECUTION_PROVIDER
        self._ort = _load_ort_module(ort_module)
        self._sessions: dict[AttentionDuration, _OrtSession] = {}
        self._io_names: dict[AttentionDuration, tuple[str, str, tuple[int, ...]]] = {}
        self.last_metadata: FoveacastInferenceMetadata | None = None
        self.last_timing: FoveacastTiming | None = None
        self._cold_load_sessions()

    def _cold_load_sessions(self) -> None:
        started = perf_counter()
        for duration in _DURATIONS:
            session = self._ort.InferenceSession(
                str(self.model_paths[duration]),
                providers=[CPU_EXECUTION_PROVIDER],
            )
            self._sessions[duration] = session
            self._io_names[duration] = _validate_session(session)
        cold_load_ms = (perf_counter() - started) * 1000.0
        self._cold_load_ms = cold_load_ms

    def predict(self, request: SaliencyPredictionRequest) -> SaliencyPredictionSet:
        """Infer requested durations in fixed 1s, 3s, 7s order."""

        metadata = request.metadata
        if self.model_id not in metadata.model_set:
            raise ValueError("request model_set does not include configured model")
        if metadata.precision != self.precision:
            raise ValueError(
                f"request precision must be {self.precision!r}, got {metadata.precision!r}"
            )

        processed = preprocess_screenshot(request.screenshot)
        if processed.geometry.source_dimensions != metadata.screenshot_dimensions:
            raise ValueError("screenshot dimensions do not match request metadata")
        requested = set(metadata.requested_durations)
        unsupported = requested.difference(_DURATIONS)
        if unsupported:
            raise ValueError(
                "Foveacast supports only 1s, 3s, and 7s requested durations"
            )

        predictions: list[SaliencyPrediction] = []
        warm_timings: list[tuple[AttentionDuration, float]] = []
        total_started = perf_counter()
        for duration in _DURATIONS:
            if duration not in requested:
                continue
            input_name, output_name, output_shape = self._io_names[duration]
            started = perf_counter()
            outputs = self._sessions[duration].run(
                [output_name], {input_name: processed.tensor}
            )
            inference_duration_ms = (perf_counter() - started) * 1000.0
            if len(outputs) != 1:
                raise ValueError("Foveacast model must return exactly one output")
            saliency_map = _normalize_output(outputs[0], output_shape)
            warm_timings.append((duration, inference_duration_ms))
            plane = SaliencyPlane(
                width=MODEL_WIDTH,
                height=MODEL_HEIGHT,
                values=saliency_map.astype("<f4", copy=False).tobytes(order="C"),
            )
            predictions.append(
                SaliencyPrediction(
                    viewport_id=metadata.viewport_id,
                    duration=duration,
                    plane=plane,
                    metadata=SaliencyPredictionMetadata(
                        provider_id=PROVIDER_ID,
                        model_id=self.model_id,
                        provider_version=self.provider_version,
                        model_version=self.model_version,
                        model_checksum=self._checksum_for(duration),
                        input_dimensions=(MODEL_WIDTH, MODEL_HEIGHT),
                        output_dimensions=(MODEL_WIDTH, MODEL_HEIGHT),
                        preprocessing_version=PREPROCESSING_VERSION,
                        inference_duration_ms=inference_duration_ms,
                        execution_provider=self.execution_provider,
                    ),
                )
            )
        total_inference_ms = (perf_counter() - total_started) * 1000.0
        timing = FoveacastTiming(
            cold_load_ms=self._cold_load_ms,
            warm_inference_ms=tuple(warm_timings),
            total_inference_ms=total_inference_ms,
        )
        self.last_timing = timing
        self.last_metadata = FoveacastInferenceMetadata(
            viewport_id=metadata.viewport_id,
            screenshot_dimensions=metadata.screenshot_dimensions,
            geometry=processed.geometry,
            timing=timing,
            model_id=self.model_id,
            model_version=self.model_version,
            precision=self.precision,
            preprocessing_version=PREPROCESSING_VERSION,
            execution_provider=self.execution_provider,
            input_name=self._io_names[_DURATIONS[0]][0],
            output_name=self._io_names[_DURATIONS[0]][1],
        )
        return SaliencyPredictionSet(
            viewport_id=metadata.viewport_id,
            predictions=tuple(predictions),
            request_metadata=metadata,
        )

    def _checksum_for(self, duration: AttentionDuration) -> str:
        configured = self.model_checksums.get(duration)
        if configured is not None:
            return configured
        checksum = _sha256_file(self.model_paths[duration])
        return checksum or f"unverified:{duration.value}"


def _load_ort_module(ort_module: _OrtModule | object | None) -> _OrtModule:
    if ort_module is not None:
        return cast(_OrtModule, ort_module)
    try:
        return cast(_OrtModule, import_module("onnxruntime"))
    except ImportError as error:
        raise RuntimeError(
            "ONNX Runtime missing; install ux-analyzer saliency-cpu extra"
        ) from error


__all__ = [
    "CPU_EXECUTION_PROVIDER",
    "FoveacastGeometry",
    "FoveacastInferenceMetadata",
    "FoveacastSaliencyProvider",
    "FoveacastTiming",
    "MODEL_HEIGHT",
    "MODEL_WIDTH",
    "PADDING_VALUE",
    "PREPROCESSING_VERSION",
    "PreprocessedScreenshot",
    "preprocess_image",
    "preprocess_screenshot",
]
