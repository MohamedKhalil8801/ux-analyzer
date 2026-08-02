"""ONNX Runtime adapters for model-dependent saliency providers."""

from ux_analyzer.adapters.saliency.foveacast import (
    FoveacastGeometry,
    FoveacastInferenceMetadata,
    FoveacastSaliencyProvider,
    FoveacastTiming,
    PreprocessedScreenshot,
    preprocess_image,
    preprocess_screenshot,
)

__all__ = [
    "FoveacastGeometry",
    "FoveacastInferenceMetadata",
    "FoveacastSaliencyProvider",
    "FoveacastTiming",
    "PreprocessedScreenshot",
    "preprocess_image",
    "preprocess_screenshot",
]
